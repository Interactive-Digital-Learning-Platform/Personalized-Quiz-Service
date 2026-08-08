"""
tests/test_analytics_growth.py
──────────────────────────────────
Integration tests for the `growth` object in GET /analytics/me — the
formula itself is unit-tested independently in tests/test_growth_service.py;
these tests verify it's wired up correctly end-to-end (real DB, real HTTP
response, real rolling-window filtering).
"""
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services import growth_service
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

NOW = datetime.now(timezone.utc)


async def _submit_quiz(
    db, user_id: int, subject: str, lesson: str, difficulty: str,
    question_count: int, correct_count: int, created_at: datetime,
    clerk_id: str = TEST_CLERK_ID,
) -> QuizSession:
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty=difficulty,
        question_count=question_count, created_at=created_at,
    )
    db.add(session)
    await db.flush()

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

    accuracy = round(correct_count / question_count * 100.0, 2)
    db.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=question_count * 5.0,
        score=float(correct_count), accuracy=accuracy,
        correct_count=correct_count, total_questions=question_count,
    ))
    await db.commit()
    await update_analytics_after_submission(db, clerk_id, session.id)
    return session


async def _create_empty_session(
    db, user_id: int, subject: str, lesson: str, difficulty: str, created_at: datetime,
) -> QuizSession:
    """A session with no attempts and no completion — the "rapidly created,
    never answered" case this feature must not reward."""
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty=difficulty,
        question_count=10, created_at=created_at,
    )
    db.add(session)
    await db.commit()
    return session


# ─────────────────────────────────────────────────────────────────────────────
# 1. No activity at all -> insufficient_data
# ─────────────────────────────────────────────────────────────────────────────

async def test_growth_insufficient_data_with_no_activity(client, db_session):
    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    growth = resp.json()["growth"]

    assert growth["effort_score"] is None
    assert growth["consistency_score"] is None
    assert growth["improvement_score"] is None
    assert growth["mastery_score"] is None
    assert growth["growth_score"] is None
    assert growth["growth_level"] == "insufficient_data"
    assert growth["components"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sufficient recent activity -> computed, internally consistent
# ─────────────────────────────────────────────────────────────────────────────

async def test_growth_computed_with_sufficient_recent_activity(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_GROWTH_MIN_ATTEMPTS_IN_WINDOW == 3  # default assumed here
    await _submit_quiz(
        db_session, user.id, "Mathematics", "Algebra", "easy", 10, 8,
        created_at=NOW - timedelta(days=2),
    )
    await _submit_quiz(
        db_session, user.id, "Mathematics", "Algebra", "easy", 10, 9,
        created_at=NOW - timedelta(days=5),
    )

    resp = await client.get("/api/v1/analytics/me")
    growth = resp.json()["growth"]

    assert growth["growth_level"] != "insufficient_data"
    assert growth["growth_score"] is not None
    assert growth["components"] is not None
    for scope in ["effort", "consistency", "improvement"]:
        for value in growth["components"][scope].values():
            assert 0.0 <= value <= 100.0

    # The response's own growth_score must be reproducible from its own
    # effort/consistency/improvement/mastery scores.
    expected = growth_service.compute_growth_score(
        growth["effort_score"], growth["consistency_score"],
        growth["improvement_score"], growth["mastery_score"],
        effort_weight=settings.ANALYTICS_GROWTH_EFFORT_WEIGHT,
        consistency_weight=settings.ANALYTICS_GROWTH_CONSISTENCY_WEIGHT,
        improvement_weight=settings.ANALYTICS_GROWTH_IMPROVEMENT_WEIGHT,
        mastery_weight=settings.ANALYTICS_GROWTH_MASTERY_WEIGHT,
    )
    assert growth["growth_score"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3. Empty sessions don't inflate effort/growth (anti-gaming, end to end)
# ─────────────────────────────────────────────────────────────────────────────

async def test_growth_empty_sessions_do_not_inflate_effort(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Just enough real activity to clear the minimum-attempts gate...
    await _submit_quiz(
        db_session, user.id, "Science", "Physics", "easy", 3, 2,
        created_at=NOW - timedelta(days=1),
    )
    # ...alongside a burst of rapidly created, never-answered sessions.
    for i in range(10):
        await _create_empty_session(
            db_session, user.id, "Science", "Physics", "easy",
            created_at=NOW - timedelta(hours=i),
        )

    resp = await client.get("/api/v1/analytics/me")
    growth = resp.json()["growth"]

    assert growth["growth_level"] != "insufficient_data"
    # 1 completed out of 11 total sessions in the window -> a low completion
    # rate, which must show up directly in the effort breakdown.
    assert growth["components"]["effort"]["completion_rate_score"] < 15.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Activity outside the rolling window is ignored
# ─────────────────────────────────────────────────────────────────────────────

async def test_growth_ignores_activity_outside_rolling_window(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_GROWTH_WINDOW_DAYS == 30  # default assumed here
    # All activity is 40 days old — outside the 30-day growth window.
    await _submit_quiz(
        db_session, user.id, "History", "Ancient Rome", "easy", 10, 8,
        created_at=NOW - timedelta(days=40),
    )

    resp = await client.get("/api/v1/analytics/me")
    growth = resp.json()["growth"]

    # No qualifying attempts fall within the window, so growth is
    # insufficient_data despite there being real all-time history.
    assert growth["growth_level"] == "insufficient_data"
    assert growth["growth_score"] is None
