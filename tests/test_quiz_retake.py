"""
tests/test_quiz_retake.py
──────────────────────────
Covers "Restart Quiz" (POST /quiz/sessions/{id}/retake) and how a retake
submission is graded but kept out of analytics/adaptive difficulty.

Restarting used to resubmit into the SAME session_id, which submit_quiz()
rejects with 409 once that session already has a QuizCompletion. Restart now
clones the session into a brand-new retake row (QuizSession.is_retake=True)
instead — this covers that the clone carries over the original questions,
that submitting it works (no more 409) and is still graded normally for the
user, but that none of it leaks into GET /analytics/me, GET /quiz/sessions,
or SubjectMastery/LessonMastery (the user already saw the correct answers by
the time they retake, so it must never look like real performance).
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.subject_mastery import SubjectMastery
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_question(db: AsyncSession, subject: str = "Mathematics", lesson: str = "Algebra") -> Question:
    q = Question(
        question="Q?", options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson=lesson, difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_completed_session(db: AsyncSession, user_id: int, question: Question) -> QuizSession:
    session = QuizSession(
        user_id=user_id,
        subject=question.subject,
        lesson=question.lesson,
        difficulty="easy",
        question_count=1,
        questions_snapshot=[{
            "id": question.id, "question": question.question, "options": question.options,
            "subject": question.subject, "lesson": question.lesson, "difficulty": question.difficulty,
            "correct_answer": question.correct_answer,
        }],
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def _submit(client, session_id: int, question_id: int, selected_answer: str = "A"):
    return await client.post(
        "/api/v1/quiz/submit",
        json={
            "session_id": session_id,
            "ended_by": "submitted",
            "answers": [
                {"question_id": question_id, "selected_answer": selected_answer, "response_time": 3.0},
            ],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Retaking a completed session no longer 409s, and clones its questions
# ─────────────────────────────────────────────────────────────────────────────

async def test_retake_clones_completed_session(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    session_id, question_id, questions_snapshot = session.id, question.id, session.questions_snapshot

    first_submit = await _submit(client, session_id, question_id)
    assert first_submit.status_code == 200

    # Resubmitting the SAME session is still rejected — that's the bug this
    # feature works around by cloning, not by weakening this guard.
    resubmit = await _submit(client, session_id, question_id)
    assert resubmit.status_code == 409

    retake_resp = await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")
    assert retake_resp.status_code == 201
    retake_id = retake_resp.json()["session_id"]
    assert retake_id != session_id

    db_session.expire_all()
    retake_row = (
        await db_session.execute(select(QuizSession).where(QuizSession.id == retake_id))
    ).scalar_one()
    assert retake_row.is_retake is True
    assert retake_row.retake_of_session_id == session_id
    assert retake_row.questions_snapshot == questions_snapshot

    retake_submit = await _submit(client, retake_id, question_id)
    assert retake_submit.status_code == 200
    assert retake_submit.json()["correct_count"] == 1


async def test_retaking_a_retake_chains_to_the_original(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    session_id, question_id = session.id, question.id

    retake_1_resp = await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")
    retake_1_id = retake_1_resp.json()["session_id"]
    await _submit(client, retake_1_id, question_id)

    retake_2_resp = await client.post(f"/api/v1/quiz/sessions/{retake_1_id}/retake")
    assert retake_2_resp.status_code == 201
    retake_2_id = retake_2_resp.json()["session_id"]

    db_session.expire_all()
    retake_2_row = (
        await db_session.execute(select(QuizSession).where(QuizSession.id == retake_2_id))
    ).scalar_one()
    # Chained to the ORIGINAL session, not the immediate parent retake.
    assert retake_2_row.retake_of_session_id == session_id


async def test_retake_of_someone_elses_session_is_404(client, db_session):
    from app.models.user import User

    other = User(clerk_id="someone-else", username="other")
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)

    question = await _make_question(db_session)
    other_session = await _make_completed_session(db_session, other.id, question)
    other_session_id = other_session.id

    resp = await client.post(f"/api/v1/quiz/sessions/{other_session_id}/retake")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 2. A retake's results are graded/returned normally...
# ─────────────────────────────────────────────────────────────────────────────

async def test_retake_submission_is_graded_and_persisted(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    session_id, question_id = session.id, question.id
    await _submit(client, session_id, question_id)

    retake_id = (await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")).json()["session_id"]
    resp = await _submit(client, retake_id, question_id)
    assert resp.status_code == 200
    body = resp.json()
    assert body["correct_count"] == 1
    assert body["accuracy"] == 100.0

    db_session.expire_all()
    attempts = (
        await db_session.execute(select(QuestionAttempt).where(QuestionAttempt.session_id == retake_id))
    ).scalars().all()
    assert len(attempts) == 1
    assert attempts[0].correct is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. ...but never shows up in the sessions list, the analytics dashboard, or
#    adaptive-difficulty mastery — only the original submission should.
# ─────────────────────────────────────────────────────────────────────────────

async def test_retake_hidden_from_sessions_list(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    session_id, question_id = session.id, question.id
    await _submit(client, session_id, question_id)

    retake_id = (await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")).json()["session_id"]
    await _submit(client, retake_id, question_id)

    list_resp = await client.get("/api/v1/quiz/sessions")
    assert list_resp.status_code == 200
    session_ids = [s["session_id"] for s in list_resp.json()]
    assert session_id in session_ids
    assert retake_id not in session_ids


async def test_retake_excluded_from_analytics_dashboard(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    session_id, question_id = session.id, question.id
    await _submit(client, session_id, question_id)

    before = (await client.get("/api/v1/analytics/me")).json()
    assert before["total_sessions"] == 1
    assert before["overall_accuracy"] == 100.0

    retake_id = (await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")).json()["session_id"]
    # Wrong on the retake — if this leaked into analytics, overall_accuracy
    # would drop from 100%. It must not.
    await _submit(client, retake_id, question_id, selected_answer="B")

    after = (await client.get("/api/v1/analytics/me")).json()
    assert after["total_sessions"] == 1
    assert after["overall_accuracy"] == 100.0


async def test_retake_does_not_move_adaptive_difficulty_mastery(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session)
    session = await _make_completed_session(db_session, user.id, question)
    user_id, session_id, question_id, subject = user.id, session.id, question.id, question.subject
    await _submit(client, session_id, question_id)

    db_session.expire_all()
    mastery_after_original = (await db_session.execute(
        select(SubjectMastery).where(SubjectMastery.user_id == user_id, SubjectMastery.subject == subject)
    )).scalar_one()
    evidence_after_original = mastery_after_original.evidence_count

    retake_id = (await client.post(f"/api/v1/quiz/sessions/{session_id}/retake")).json()["session_id"]
    await _submit(client, retake_id, question_id)

    db_session.expire_all()
    mastery_after_retake = (await db_session.execute(
        select(SubjectMastery).where(SubjectMastery.user_id == user_id, SubjectMastery.subject == subject)
    )).scalar_one()
    assert mastery_after_retake.evidence_count == evidence_after_original
