"""
tests/test_analytics_completion.py
─────────────────────────────────────
Tests for the session-completion analytics added to GET /analytics/me:
completed_sessions, incomplete_sessions, timed_out_sessions,
abandoned_sessions, completion_rate, timeout_rate,
average_session_duration_seconds, average_questions_per_session.

Sessions are built directly against the test DB (QuizSession /
QuizCompletion / QuizProgressSnapshot) rather than via POST /quiz/generate
or /quiz/submit, since those exercise the real Groq API and full grading
pipeline — unrelated to what's being tested here (see test_analytics.py for
the same rationale).
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.quiz_session import QuizSession
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_session(
    db: AsyncSession,
    user_id: int,
    subject: str,
    question_count: int = 10,
    created_at: datetime | None = None,
) -> QuizSession:
    session = QuizSession(
        user_id=user_id,
        subject=subject,
        lesson="Test Lesson",
        difficulty="easy",
        question_count=question_count,
        **({"created_at": created_at} if created_at is not None else {}),
    )
    db.add(session)
    await db.flush()
    return session


async def _complete_session(
    db: AsyncSession,
    session: QuizSession,
    ended_by: str = "submitted",
    total_time: float = 180.0,
) -> QuizCompletion:
    completion = QuizCompletion(
        session_id=session.id,
        ended_by=ended_by,
        total_time=total_time,
        score=5.0,
        accuracy=50.0,
        correct_count=5,
        total_questions=session.question_count,
    )
    db.add(completion)
    await db.commit()
    return completion


async def _save_progress(db: AsyncSession, session: QuizSession, saved_at: datetime) -> None:
    db.add(QuizProgressSnapshot(
        session_id=session.id,
        remaining_time=60.0,
        answered_count=3,
        saved_at=saved_at,
    ))
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Completed session
# ─────────────────────────────────────────────────────────────────────────────

async def test_completed_session_is_classified_correctly(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id, "Mathematics", question_count=10)
    await _complete_session(db_session, session, ended_by="submitted", total_time=200.0)

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_sessions"] == 1
    assert data["completed_sessions"] == 1
    assert data["incomplete_sessions"] == 0
    assert data["timed_out_sessions"] == 0
    assert data["abandoned_sessions"] == 0
    assert data["completion_rate"] == 100.0
    assert data["timeout_rate"] == 0.0
    assert data["average_session_duration_seconds"] == 200.0
    assert data["average_questions_per_session"] == 10.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Active incomplete session (recent activity, not abandoned)
# ─────────────────────────────────────────────────────────────────────────────

async def test_active_incomplete_session_is_not_abandoned(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id, "Science", question_count=8)
    await db_session.commit()
    # Saved progress 1 hour ago — well within the 24h abandonment threshold.
    await _save_progress(db_session, session, datetime.now(timezone.utc) - timedelta(hours=1))

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_sessions"] == 1
    assert data["completed_sessions"] == 0
    assert data["incomplete_sessions"] == 1
    assert data["abandoned_sessions"] == 0
    assert data["completion_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Abandoned session
# ─────────────────────────────────────────────────────────────────────────────

async def test_abandoned_session_is_classified_correctly(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    old_time = datetime.now(timezone.utc) - timedelta(
        hours=settings.ANALYTICS_ABANDONED_AFTER_HOURS + 1
    )
    session = await _make_session(db_session, user.id, "History", question_count=6)
    await db_session.commit()
    # Last saved progress is older than the abandonment threshold.
    await _save_progress(db_session, session, old_time)

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["incomplete_sessions"] == 1
    assert data["abandoned_sessions"] == 1


async def test_incomplete_session_with_no_snapshot_uses_created_at(client, db_session):
    """An incomplete session that was never saved falls back to created_at."""
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    old_time = datetime.now(timezone.utc) - timedelta(
        hours=settings.ANALYTICS_ABANDONED_AFTER_HOURS + 5
    )
    await _make_session(db_session, user.id, "Geography", question_count=5, created_at=old_time)
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    assert data["incomplete_sessions"] == 1
    assert data["abandoned_sessions"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Timeout submission
# ─────────────────────────────────────────────────────────────────────────────

async def test_timeout_submission_counts_as_completed_and_timed_out(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_session(db_session, user.id, "English", question_count=10)
    await _complete_session(db_session, session, ended_by="timeout", total_time=600.0)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    # A timed-out submission still counts as completed (overlap by design)...
    assert data["completed_sessions"] == 1
    assert data["incomplete_sessions"] == 0
    # ...but is ALSO flagged here.
    assert data["timed_out_sessions"] == 1
    assert data["completion_rate"] == 100.0
    assert data["timeout_rate"] == 100.0


async def test_timeout_and_normal_completions_together(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    s1 = await _make_session(db_session, user.id, "Mathematics", question_count=10)
    await _complete_session(db_session, s1, ended_by="submitted")
    s2 = await _make_session(db_session, user.id, "Mathematics", question_count=10)
    await _complete_session(db_session, s2, ended_by="timeout")

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    assert data["total_sessions"] == 2
    assert data["completed_sessions"] == 2   # both count as completed
    assert data["timed_out_sessions"] == 1   # only one was a timeout
    assert data["completion_rate"] == 100.0
    assert data["timeout_rate"] == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. User data isolation
# ─────────────────────────────────────────────────────────────────────────────

async def test_analytics_only_includes_authenticated_users_sessions(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    other_user = await get_or_create_user(db_session, "other-user")

    # The authenticated (test) user has 1 completed session.
    my_session = await _make_session(db_session, user.id, "Mathematics", question_count=10)
    await _complete_session(db_session, my_session, ended_by="submitted")

    # A completely different user has 5 completed sessions — must not leak in.
    for _ in range(5):
        other_session = await _make_session(db_session, other_user.id, "Science", question_count=10)
        await _complete_session(db_session, other_session, ended_by="submitted")

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    assert data["total_sessions"] == 1
    assert data["completed_sessions"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Zero-session user
# ─────────────────────────────────────────────────────────────────────────────

async def test_zero_session_user_returns_all_zeros(client, db_session):
    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_sessions"] == 0
    assert data["completed_sessions"] == 0
    assert data["incomplete_sessions"] == 0
    assert data["timed_out_sessions"] == 0
    assert data["abandoned_sessions"] == 0
    assert data["completion_rate"] == 0.0
    assert data["timeout_rate"] == 0.0
    assert data["average_session_duration_seconds"] == 0.0
    assert data["average_questions_per_session"] == 0.0
