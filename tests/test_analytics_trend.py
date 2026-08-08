"""
tests/test_analytics_trend.py
───────────────────────────────
Tests for the performance-trend analytics added to GET /analytics/me:
`performance_trend` at the overall, subject, and topic levels — comparing
either the latest N completed sessions vs the N before them
("recent_sessions"), or the current vs previous calendar week ("weekly"),
whichever has enough data (see scoring_service.compute_performance_trend()).

Sessions/questions/attempts are built directly against the test DB rather
than via POST /quiz/generate or /quiz/submit — see test_analytics.py for the
same rationale. `completed_at` is set explicitly on each QuizCompletion so
each test controls exactly which comparison period a session lands in.
"""
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from app.services.scoring_service import start_of_iso_week
from tests.conftest import TEST_CLERK_ID

NOW = datetime.now(timezone.utc)
# Safely inside the previous ISO week regardless of what day `NOW` falls on:
# subtracting 3 days from the current week's Monday always lands within
# [current_week_start - 7, current_week_start), i.e. last week, never crossing
# into the week before that or bleeding into the current week.
PREVIOUS_WEEK_DT = start_of_iso_week(NOW) - timedelta(days=3)
CURRENT_WEEK_DT = NOW - timedelta(hours=2)


async def _make_question(db, subject: str, lesson: str, i: int) -> Question:
    q = Question(
        question=f"Q{i}?", options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson=lesson, difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_completed_session(
    db,
    user_id: int,
    subject: str,
    question_count: int,
    correct_count: int,
    completed_at: datetime,
    lesson: str = "General",
    clerk_id: str = TEST_CLERK_ID,
) -> QuizSession:
    """A completed session with `correct_count` correct out of `question_count`."""
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty="easy",
        question_count=question_count,
    )
    db.add(session)
    await db.flush()

    for i in range(question_count):
        is_correct = i < correct_count
        question = await _make_question(db, subject, lesson, i)
        db.add(QuestionAttempt(
            session_id=session.id, question_id=question.id,
            selected_answer="A" if is_correct else "B",
            correct=is_correct, response_time=5.0,
        ))

    db.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=question_count * 5.0,
        score=float(correct_count), accuracy=round(correct_count / question_count * 100.0, 2),
        correct_count=correct_count, total_questions=question_count,
        completed_at=completed_at,
    ))
    await db.commit()

    # Mirrors what POST /quiz/submit does — populates the Analytics row that
    # `subjects` in the response is built from.
    await update_analytics_after_submission(db, clerk_id, session.id)
    return session


def _subject(data: dict, subject: str) -> dict:
    return next(s for s in data["subjects"] if s["subject"] == subject)


def _topic(data: dict, subject: str, topic: str) -> dict:
    return next(t for t in _subject(data, subject)["topics"] if t["topic"] == topic)


async def _make_recent_sessions_window(
    db, user_id: int, subject: str,
    previous_correct_counts: list[int], current_correct_counts: list[int],
    question_count: int = 10,
) -> None:
    """
    Builds 2x ANALYTICS_TREND_SESSION_WINDOW (default 5+5=10) completed
    sessions so the "recent_sessions" method is used: the 5 oldest become the
    previous period, the 5 newest become the current period.
    """
    window = settings.ANALYTICS_TREND_SESSION_WINDOW
    assert len(previous_correct_counts) == window
    assert len(current_correct_counts) == window

    for i, correct in enumerate(previous_correct_counts):
        days_ago = 2 * window - i  # oldest first, strictly decreasing recency
        await _make_completed_session(
            db, user_id, subject, question_count, correct, NOW - timedelta(days=days_ago),
        )
    for i, correct in enumerate(current_correct_counts):
        days_ago = window - i
        await _make_completed_session(
            db, user_id, subject, question_count, correct, NOW - timedelta(days=days_ago),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Improving performance (recent_sessions method)
# ─────────────────────────────────────────────────────────────────────────────

async def test_improving_performance(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _make_recent_sessions_window(
        db_session, user.id, "Mathematics",
        previous_correct_counts=[3, 3, 3, 3, 3],   # 30% each
        current_correct_counts=[8, 8, 8, 8, 8],    # 80% each
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    trend = data["performance_trend"]
    assert trend["method"] == "recent_sessions"
    assert trend["previous_period_accuracy"] == 30.0
    assert trend["current_period_accuracy"] == 80.0
    assert trend["accuracy_change"] == 50.0
    assert trend["current_period_sessions"] == 5
    assert trend["previous_period_sessions"] == 5
    assert trend["trend"] == "improving"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Declining performance (recent_sessions method)
# ─────────────────────────────────────────────────────────────────────────────

async def test_declining_performance(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _make_recent_sessions_window(
        db_session, user.id, "Science",
        previous_correct_counts=[8, 8, 8, 8, 8],    # 80% each
        current_correct_counts=[3, 3, 3, 3, 3],     # 30% each
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    trend = data["performance_trend"]
    assert trend["method"] == "recent_sessions"
    assert trend["previous_period_accuracy"] == 80.0
    assert trend["current_period_accuracy"] == 30.0
    assert trend["accuracy_change"] == -50.0
    assert trend["trend"] == "declining"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stable performance (recent_sessions method)
# ─────────────────────────────────────────────────────────────────────────────

async def test_stable_performance(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # 20 questions/session so a +2-point (well under the 5-point threshold)
    # weighted change is expressible with whole-number correct counts.
    await _make_recent_sessions_window(
        db_session, user.id, "History",
        previous_correct_counts=[10, 10, 10, 10, 10],  # 50/100 = 50%
        current_correct_counts=[11, 10, 10, 10, 11],   # 52/100 = 52%
        question_count=20,
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    trend = data["performance_trend"]
    assert trend["method"] == "recent_sessions"
    assert trend["previous_period_accuracy"] == 50.0
    assert trend["current_period_accuracy"] == 52.0
    assert trend["accuracy_change"] == 2.0
    assert trend["trend"] == "stable"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Only one period has data -> insufficient_data
# ─────────────────────────────────────────────────────────────────────────────

async def test_only_one_period_has_data(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Only 3 sessions total (well under 2x the recent-sessions window), all
    # in the CURRENT week — the previous week has zero completed sessions.
    for _ in range(3):
        await _make_completed_session(
            db_session, user.id, "Geography", 10, 7, CURRENT_WEEK_DT,
        )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    trend = data["performance_trend"]
    assert trend["method"] == "insufficient_data"
    assert trend["trend"] == "insufficient_data"
    assert trend["current_period_accuracy"] == 0.0
    assert trend["previous_period_accuracy"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Different question counts per session -> weighted, not averaged
# ─────────────────────────────────────────────────────────────────────────────

async def test_weighted_accuracy_with_different_question_counts(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    window = settings.ANALYTICS_TREND_SESSION_WINDOW
    assert window == 5  # default assumed by this test

    # Previous period: uniform 50% across 5 sessions (baseline, not the point
    # under test).
    for i in range(window):
        await _make_completed_session(
            db_session, user.id, "English", 10, 5, NOW - timedelta(days=2 * window - i),
        )
    # Current period: four 10-question sessions at 80% each, plus one
    # 2-question session at 0%. Naive average of session percentages would be
    # (80*4 + 0) / 5 = 64%. Weighted (the required method) is
    # (8*4 + 0) / (10*4 + 2) = 32/42 = 76.19%.
    for i in range(4):
        await _make_completed_session(
            db_session, user.id, "English", 10, 8, NOW - timedelta(days=window - i),
        )
    await _make_completed_session(
        db_session, user.id, "English", 2, 0, NOW - timedelta(days=window - 4),
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    trend = data["performance_trend"]
    assert trend["method"] == "recent_sessions"
    expected_weighted_accuracy = round(32 / 42 * 100.0, 2)
    assert trend["current_period_accuracy"] == expected_weighted_accuracy
    assert trend["current_period_accuracy"] != 64.0  # the naive-average trap


# ─────────────────────────────────────────────────────────────────────────────
# 6. Subject-specific trends (weekly method) — independent per subject
# ─────────────────────────────────────────────────────────────────────────────

async def test_subject_specific_trends(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    # Mathematics: improving (30% -> 80%).
    await _make_completed_session(db_session, user.id, "Mathematics", 10, 3, PREVIOUS_WEEK_DT)
    await _make_completed_session(db_session, user.id, "Mathematics", 10, 8, CURRENT_WEEK_DT)
    # Science: declining (80% -> 30%).
    await _make_completed_session(db_session, user.id, "Science", 10, 8, PREVIOUS_WEEK_DT)
    await _make_completed_session(db_session, user.id, "Science", 10, 3, CURRENT_WEEK_DT)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    maths_trend = _subject(data, "Mathematics")["performance_trend"]
    assert maths_trend["method"] == "weekly"
    assert maths_trend["trend"] == "improving"
    assert maths_trend["accuracy_change"] == 50.0

    science_trend = _subject(data, "Science")["performance_trend"]
    assert science_trend["method"] == "weekly"
    assert science_trend["trend"] == "declining"
    assert science_trend["accuracy_change"] == -50.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Topic-specific trends (weekly method) — independent per topic, and
#    distinct from the subject-level trend that blends both topics together.
# ─────────────────────────────────────────────────────────────────────────────

async def test_topic_specific_trends(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    # "Ancient Rome": improving (20% -> 80%).
    await _make_completed_session(
        db_session, user.id, "History", 10, 2, PREVIOUS_WEEK_DT, lesson="Ancient Rome",
    )
    await _make_completed_session(
        db_session, user.id, "History", 10, 8, CURRENT_WEEK_DT, lesson="Ancient Rome",
    )
    # "Medieval Europe": declining (80% -> 20%).
    await _make_completed_session(
        db_session, user.id, "History", 10, 8, PREVIOUS_WEEK_DT, lesson="Medieval Europe",
    )
    await _make_completed_session(
        db_session, user.id, "History", 10, 2, CURRENT_WEEK_DT, lesson="Medieval Europe",
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    rome_trend = _topic(data, "History", "Ancient Rome")["performance_trend"]
    assert rome_trend["method"] == "weekly"
    assert rome_trend["trend"] == "improving"
    assert rome_trend["accuracy_change"] == 60.0

    medieval_trend = _topic(data, "History", "Medieval Europe")["performance_trend"]
    assert medieval_trend["method"] == "weekly"
    assert medieval_trend["trend"] == "declining"
    assert medieval_trend["accuracy_change"] == -60.0

    # The subject-level trend blends both topics (10/20=50% each period) —
    # topic-level drill-down reveals what the subject-level number hides.
    history_trend = _subject(data, "History")["performance_trend"]
    assert history_trend["trend"] == "stable"
    assert history_trend["accuracy_change"] == 0.0
