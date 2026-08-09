"""
tests/test_quiz_session_delete.py
──────────────────────────────────
Tests for DELETE /quiz/sessions/{id} soft-delete behavior.

Deleting a session must remove it from the user-facing list/detail
endpoints, but must NOT erase the row from the DB — analytics
(total_sessions, trend/growth/difficulty/repeated-question stats) and
future quiz generation (lesson-variety avoidance in
quiz_service._get_recent_lessons) both read session history directly,
not just the separate aggregate Analytics table.

Sessions are built directly against the test DB, same rationale as
test_analytics_completion.py: this is about delete/list/analytics
plumbing, not the real Groq generation pipeline.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import _get_recent_lessons, get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_question(db: AsyncSession, subject: str, lesson: str = "Topic") -> Question:
    q = Question(
        question="Q?", options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson=lesson, difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_session(
    db: AsyncSession,
    user_id: int,
    subject: str = "Mathematics",
    lesson: str = "Algebra",
) -> QuizSession:
    session = QuizSession(
        user_id=user_id,
        subject=subject,
        lesson=lesson,
        difficulty="easy",
        question_count=10,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# 1. Delete removes the session from the user-facing list and detail views
# ─────────────────────────────────────────────────────────────────────────────

async def test_delete_removes_session_from_list(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id)

    resp = await client.delete(f"/api/v1/quiz/sessions/{session.id}")
    assert resp.status_code == 204

    list_resp = await client.get("/api/v1/quiz/sessions")
    assert list_resp.status_code == 200
    assert list_resp.json() == []


async def test_delete_makes_session_detail_404(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id)

    resp = await client.delete(f"/api/v1/quiz/sessions/{session.id}")
    assert resp.status_code == 204

    detail_resp = await client.get(f"/api/v1/quiz/sessions/{session.id}")
    assert detail_resp.status_code == 404


async def test_deleting_already_deleted_session_is_404(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id)

    first = await client.delete(f"/api/v1/quiz/sessions/{session.id}")
    assert first.status_code == 204

    second = await client.delete(f"/api/v1/quiz/sessions/{session.id}")
    assert second.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 2. Delete does NOT hard-delete the row — the data survives in the DB
# ─────────────────────────────────────────────────────────────────────────────

async def test_delete_soft_deletes_not_hard_deletes(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id)
    session_id = session.id

    db_session.add(QuestionAttempt(
        session_id=session_id,
        question_id=1,
        selected_answer="A",
        correct=True,
        response_time=5.0,
    ))
    await db_session.commit()

    resp = await client.delete(f"/api/v1/quiz/sessions/{session_id}")
    assert resp.status_code == 204

    # db_session's identity map still holds the pre-delete `session` object
    # (conftest's session_factory uses expire_on_commit=False) — expire it so
    # the next query re-reads the row the other (request-scoped) session wrote.
    db_session.expire_all()

    # The QuizSession row itself must still exist, marked deleted.
    row = (
        await db_session.execute(select(QuizSession).where(QuizSession.id == session_id))
    ).scalar_one_or_none()
    assert row is not None
    assert row.deleted_at is not None

    # Its QuestionAttempt children must NOT have been cascade-deleted.
    attempts = (
        await db_session.execute(
            select(QuestionAttempt).where(QuestionAttempt.session_id == session_id)
        )
    ).scalars().all()
    assert len(attempts) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. The GET /analytics/me dashboard reflects only CURRENT (non-deleted)
#    sessions — total_sessions must go back down when a session is deleted,
#    same as it goes up when one is generated.
# ─────────────────────────────────────────────────────────────────────────────

async def test_total_sessions_decreases_after_delete(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session_a = await _make_session(db_session, user.id, subject="Mathematics")
    await _make_session(db_session, user.id, subject="Science")

    before = await client.get("/api/v1/analytics/me")
    assert before.json()["total_sessions"] == 2

    resp = await client.delete(f"/api/v1/quiz/sessions/{session_a.id}")
    assert resp.status_code == 204

    after = await client.get("/api/v1/analytics/me")
    assert after.json()["total_sessions"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Lesson-variety generation history keeps reading deleted sessions
# ─────────────────────────────────────────────────────────────────────────────

async def test_recent_lessons_include_deleted_sessions(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id, subject="Mathematics", lesson="Trigonometry")

    resp = await client.delete(f"/api/v1/quiz/sessions/{session.id}")
    assert resp.status_code == 204

    recent = await _get_recent_lessons(db_session, user.id, "Mathematics")
    assert "Trigonometry" in recent


# ─────────────────────────────────────────────────────────────────────────────
# 5. Deleted sessions' graded attempts drop out of overall/subject accuracy —
#    the dashboard numbers must match what's actually still in the list.
# ─────────────────────────────────────────────────────────────────────────────

async def test_overall_and_subject_accuracy_exclude_deleted_session(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    kept_question = await _make_question(db_session, subject="Mathematics")
    kept = await _make_session(db_session, user.id, subject="Mathematics")
    kept.question_count = 1
    db_session.add(QuestionAttempt(
        session_id=kept.id, question_id=kept_question.id, selected_answer="A", correct=True, response_time=5.0,
    ))
    await db_session.commit()
    # Seeds the (user, "Mathematics") Analytics row, same as a real submit would.
    await update_analytics_after_submission(db=db_session, clerk_id=TEST_CLERK_ID, session_id=kept.id)

    deleted_question = await _make_question(db_session, subject="Science")
    deleted = await _make_session(db_session, user.id, subject="Science")
    deleted.question_count = 1
    db_session.add(QuestionAttempt(
        session_id=deleted.id, question_id=deleted_question.id, selected_answer="B", correct=False, response_time=5.0,
    ))
    await db_session.commit()
    # Seeds the (user, "Science") Analytics row, same as a real submit would.
    await update_analytics_after_submission(db=db_session, clerk_id=TEST_CLERK_ID, session_id=deleted.id)

    before = await client.get("/api/v1/analytics/me")
    assert before.json()["overall_accuracy"] == 50.0

    resp = await client.delete(f"/api/v1/quiz/sessions/{deleted.id}")
    assert resp.status_code == 204

    after = await client.get("/api/v1/analytics/me")
    data = after.json()
    assert data["overall_accuracy"] == 100.0
    assert [s["subject"] for s in data["subjects"]] == ["Mathematics"]
