"""
tests/test_analytics_bkt.py
──────────────────────────────
Integration tests for bkt_mastery on subjects[].topics[] and
mastered_skill_count/total_skill_count on subjects[] in GET /analytics/me —
the BKT math itself is unit-tested in tests/test_bkt_service.py; these tests
verify it's wired up correctly end-to-end (real DB, real HTTP response).
"""
from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services import bkt_service
from app.services.analytics_service import update_analytics_after_submission
from app.services.difficulty_service import GradedAnswer
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _submit_quiz(
    db, user_id: int, subject: str, lesson: str, difficulty: str,
    question_count: int, correct_count: int, grade: int | None = None,
    clerk_id: str = TEST_CLERK_ID,
) -> None:
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty=difficulty,
        question_count=question_count, grade=grade,
    )
    db.add(session)
    await db.flush()

    graded_answers: list[GradedAnswer] = []
    for i in range(question_count):
        is_correct = i < correct_count
        question = Question(
            question=f"Q{i}?", options=["A", "B", "C", "D"], correct_answer="A",
            subject=subject, lesson=lesson, difficulty=difficulty,
        )
        db.add(question)
        await db.flush()
        db.add(QuestionAttempt(
            session_id=session.id, question_id=question.id,
            selected_answer="A" if is_correct else "B", correct=is_correct, response_time=5.0,
        ))
        graded_answers.append(GradedAnswer(
            subject=subject, lesson=lesson, difficulty=difficulty, correct=is_correct,
            response_time=5.0, fingerprint="",
        ))

    accuracy = round(correct_count / question_count * 100.0, 2)
    db.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=question_count * 5.0,
        score=float(correct_count), accuracy=accuracy,
        correct_count=correct_count, total_questions=question_count,
    ))
    await db.commit()

    # Mirrors the two update pathways quiz_service.submit_quiz drives for a
    # real submission: the legacy Analytics-table rollup (what makes the
    # subject appear in subjects[] at all) and BKT.
    await update_analytics_after_submission(db, clerk_id, session.id)
    await bkt_service.update_bkt_after_submission(
        db=db, user_id=user_id, grade=grade, graded_answers=graded_answers,
    )
    await db.commit()


def _subject(data: dict, subject: str) -> dict:
    return next(s for s in data["subjects"] if s["subject"] == subject)


def _topic(data: dict, subject: str, topic: str) -> dict:
    return next(t for t in _subject(data, subject)["topics"] if t["topic"] == topic)


async def test_bkt_mastery_populated_after_submission(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit_quiz(db_session, user.id, "Mathematics", "Percentages", "easy", 5, 5, grade=10)

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    topic = _topic(resp.json(), "Mathematics", "Percentages")

    assert topic["bkt_mastery"] is not None
    assert topic["bkt_mastery"]["opportunities"] == 5
    assert 0.0 <= topic["bkt_mastery"]["p_know"] <= 1.0
    assert topic["bkt_mastery"]["mastery_label"] in {"not_started", "learning", "mastered"}


async def test_bkt_mastery_none_for_topic_with_no_bkt_state(client, db_session):
    # A subject/topic can appear in analytics (via the Analytics-table
    # rollup) without ever having gone through the BKT update path in this
    # test's isolated DB — bkt_mastery must be None, not a fabricated
    # "0% known" row.
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = QuizSession(
        user_id=user.id, subject="Science", lesson="Kinematics", difficulty="easy", question_count=3,
    )
    db_session.add(session)
    await db_session.flush()
    for i in range(3):
        q = Question(
            question=f"Q{i}?", options=["A", "B"], correct_answer="A",
            subject="Science", lesson="Kinematics", difficulty="easy",
        )
        db_session.add(q)
        await db_session.flush()
        db_session.add(QuestionAttempt(session_id=session.id, question_id=q.id, selected_answer="A", correct=True, response_time=5.0))
    db_session.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=15.0,
        score=3.0, accuracy=100.0, correct_count=3, total_questions=3,
    ))
    await db_session.commit()
    await update_analytics_after_submission(db_session, TEST_CLERK_ID, session.id)
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/me")
    topic = _topic(resp.json(), "Science", "Kinematics")
    assert topic["bkt_mastery"] is None


async def test_mastered_skill_count_rollup_on_subject(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Enough correct opportunities to cross BKT_MASTERED_THRESHOLD.
    await _submit_quiz(db_session, user.id, "ICT", "Variables", "easy", 10, 10, grade=10)
    # Freshly created, only a couple of opportunities — should stay "learning" or "not_started".
    await _submit_quiz(db_session, user.id, "ICT", "Loops", "easy", 1, 0, grade=10)

    resp = await client.get("/api/v1/analytics/me")
    subj = _subject(resp.json(), "ICT")

    assert subj["total_skill_count"] == 2
    variables = _topic(resp.json(), "ICT", "Variables")
    assert variables["bkt_mastery"]["p_know"] >= settings.BKT_MASTERED_THRESHOLD
    assert subj["mastered_skill_count"] >= 1
