"""
tests/test_analytics_daily.py
───────────────────────────────
Tests for GET /analytics/daily — per-day activity (quizzes completed,
questions answered, accuracy, study time, average time per question)
bucketed by the student's LOCAL calendar day, over a 7- or 30-day window
that always includes zero-activity days.

Sessions are built directly against the test DB with an explicit
`completed_at` (same approach as test_analytics_trend.py) so each test
controls exactly which day a quiz lands on.
"""
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.services.analytics.daily_service import build_daily_analytics, progress_score
from app.services.analytics.types import DailySessionRow
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID
from tests.test_analytics_trend import _make_completed_session

COLOMBO = ZoneInfo("Asia/Colombo")  # UTC+5:30, no DST


def _local_today(tz: ZoneInfo):
    return datetime.now(UTC).astimezone(tz).date()


def _at_local(day, hour: int, minute: int = 0, tz: ZoneInfo = COLOMBO) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=tz).astimezone(UTC)


def _day(data: dict, iso: str) -> dict:
    return next(d for d in data["daily"] if d["date"] == iso)


async def test_window_has_every_day_oldest_first_with_empty_days(client, db_session):
    await get_or_create_user(db_session, TEST_CLERK_ID)

    resp = await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "Asia/Colombo"})
    assert resp.status_code == 200
    data = resp.json()

    today = _local_today(COLOMBO)
    assert data["timezone"] == "Asia/Colombo"
    assert data["end_date"] == today.isoformat()
    assert [d["date"] for d in data["daily"]] == [(today - timedelta(days=6 - i)).isoformat() for i in range(7)]
    assert all(d["quizzes_completed"] == 0 and d["accuracy"] is None for d in data["daily"])
    assert data["active_days"] == 0
    assert data["accuracy"] is None


async def test_days_are_bucketed_by_local_date_not_utc(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    # 01:00 in Colombo today is 19:30 UTC on the previous UTC day.
    await _make_completed_session(db_session, user.id, "Science", 4, 3, _at_local(today, 1))

    local = (await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "Asia/Colombo"})).json()
    assert _day(local, today.isoformat())["quizzes_completed"] == 1

    utc = (await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "UTC"})).json()
    utc_day = _at_local(today, 1).date().isoformat()
    assert _day(utc, utc_day)["quizzes_completed"] == 1
    assert utc_day != today.isoformat()


async def test_same_day_quizzes_aggregate_with_weighted_accuracy(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    day = _local_today(COLOMBO) - timedelta(days=1)
    await _make_completed_session(db_session, user.id, "Mathematics", 10, 9, _at_local(day, 9))
    await _make_completed_session(db_session, user.id, "English", 2, 0, _at_local(day, 18))

    data = (await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "Asia/Colombo"})).json()
    point = _day(data, day.isoformat())

    assert point["quizzes_completed"] == 2
    assert point["questions_answered"] == 12
    assert point["correct_answers"] == 9
    assert point["accuracy"] == 75.0  # 9/12 weighted -- not the 45% mean of the two quiz scores
    assert point["study_time_seconds"] == 60.0  # 12 questions x 5s
    assert point["avg_response_time"] == 5.0
    assert data["active_days"] == 1
    assert data["total_questions_answered"] == 12


async def test_excludes_retakes_deleted_and_out_of_window_quizzes(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    await _make_completed_session(db_session, user.id, "History", 4, 2, _at_local(today, 10))
    retake = await _make_completed_session(db_session, user.id, "History", 4, 4, _at_local(today, 11))
    deleted = await _make_completed_session(db_session, user.id, "History", 4, 4, _at_local(today, 12))
    await _make_completed_session(db_session, user.id, "History", 4, 4, _at_local(today - timedelta(days=8), 12))
    retake.is_retake = True
    deleted.deleted_at = datetime.now(UTC)
    await db_session.commit()

    data = (await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "Asia/Colombo"})).json()

    # Only the first quiz counts: the retake, the deleted one and the 8-day-old one are all excluded.
    assert data["total_quizzes_completed"] == 1
    assert data["total_questions_answered"] == 4
    assert _day(data, today.isoformat())["accuracy"] == 50.0


async def test_thirty_day_window_includes_older_activity(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    old_day = _local_today(COLOMBO) - timedelta(days=20)
    await _make_completed_session(db_session, user.id, "ICT", 5, 5, _at_local(old_day, 15))

    week = (await client.get("/api/v1/analytics/daily", params={"days": 7, "tz": "Asia/Colombo"})).json()
    month = (await client.get("/api/v1/analytics/daily", params={"days": 30, "tz": "Asia/Colombo"})).json()

    assert week["total_quizzes_completed"] == 0
    assert len(month["daily"]) == 30
    assert _day(month, old_day.isoformat())["accuracy"] == 100.0


async def test_rejects_unsupported_window_and_unknown_timezone(client, db_session):
    await get_or_create_user(db_session, TEST_CLERK_ID)

    assert (await client.get("/api/v1/analytics/daily", params={"days": 10})).status_code == 422
    assert (await client.get("/api/v1/analytics/daily", params={"tz": "Mars/Olympus"})).status_code == 422


def test_build_daily_analytics_keeps_no_activity_distinct_from_zero_accuracy():
    now = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)
    rows = [DailySessionRow(
        session_id=1, completed_at=datetime(2026, 10, 12, 4, 0, tzinfo=UTC),
        study_time=20.0, graded_count=4, correct_count=0, response_time_sum=20.0, response_time_count=4,
    )]

    data = build_daily_analytics(rows, days=7, tz=COLOMBO, now=now)

    assert data["daily"][-1]["accuracy"] == 0.0     # answered, all wrong
    assert data["daily"][-2]["accuracy"] is None    # no activity at all


# ─────────────────────────────────────────────────────────────────────────────
# Daily Progress Score: 0.5 x accuracy + 0.3 x improvement + 0.2 x effort
#   improvement = 50 + (day accuracy - accuracy over everything before that day), clamped 0-100 (50 with no history)
#   effort      = questions answered / 10, capped at 100
# ─────────────────────────────────────────────────────────────────────────────

async def test_progress_score_without_prior_history_uses_neutral_improvement(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    await _make_completed_session(db_session, user.id, "Mathematics", 10, 8, _at_local(today, 10))

    point = _day((await client.get("/api/v1/analytics/daily", params={"tz": "Asia/Colombo"})).json(), today.isoformat())

    assert point["baseline_accuracy"] is None
    assert (point["score_accuracy"], point["score_improvement"], point["score_effort"]) == (80.0, 50.0, 100.0)
    assert point["progress_score"] == 75.0  # 0.5*80 + 0.3*50 + 0.2*100


async def test_progress_score_compares_against_everything_before_that_day(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    # 20 days ago -- outside the 7-day window, but still part of the baseline.
    await _make_completed_session(db_session, user.id, "Science", 10, 5, _at_local(today - timedelta(days=20), 10))
    await _make_completed_session(db_session, user.id, "Science", 10, 8, _at_local(today, 10))

    data = (await client.get("/api/v1/analytics/daily", params={"tz": "Asia/Colombo"})).json()
    point = _day(data, today.isoformat())

    assert point["baseline_accuracy"] == 50.0
    assert point["score_improvement"] == 80.0  # 50 + (80 - 50)
    assert point["progress_score"] == 84.0     # 0.5*80 + 0.3*80 + 0.2*100
    assert data["total_quizzes_completed"] == 1  # the old quiz only feeds the baseline, not the window totals


async def test_baseline_includes_earlier_days_inside_the_window(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    await _make_completed_session(db_session, user.id, "English", 4, 4, _at_local(today - timedelta(days=2), 10))
    await _make_completed_session(db_session, user.id, "English", 4, 1, _at_local(today, 10))

    point = _day((await client.get("/api/v1/analytics/daily", params={"tz": "Asia/Colombo"})).json(), today.isoformat())

    assert point["baseline_accuracy"] == 100.0
    assert point["score_improvement"] == 0.0    # 50 + (25 - 100) = -25, clamped to 0
    assert point["score_effort"] == 40.0        # 4 / 10 questions
    assert point["progress_score"] == 20.5      # 0.5*25 + 0.3*0 + 0.2*40


def test_improvement_is_clamped_and_effort_capped():
    low = progress_score(0.0, 100.0, 2)
    assert low["score_improvement"] == 0.0      # 50 - 100 clamped up to 0
    high = progress_score(100.0, 0.0, 50)
    assert high["score_improvement"] == 100.0   # 50 + 100 clamped down to 100
    assert high["score_effort"] == 100.0        # 50 questions capped at the target
    assert high["progress_score"] == 100.0


async def test_weekly_score_averages_active_days_and_skips_rest_days(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    await _make_completed_session(db_session, user.id, "ICT", 10, 8, _at_local(today - timedelta(days=3), 10))  # 75.0
    await _make_completed_session(db_session, user.id, "ICT", 10, 9, _at_local(today, 10))  # baseline 80 -> 83.0

    data = (await client.get("/api/v1/analytics/daily", params={"tz": "Asia/Colombo"})).json()

    scores = {d["date"]: d["progress_score"] for d in data["daily"]}
    assert scores[(today - timedelta(days=3)).isoformat()] == 75.0
    assert scores[today.isoformat()] == 83.0     # 0.5*90 + 0.3*(50 + 90 - 80) + 0.2*100
    assert scores[(today - timedelta(days=1)).isoformat()] is None
    assert data["weekly_progress_score"] == 79.0  # mean of the two active days only


async def test_weekly_score_is_null_without_activity(client, db_session):
    await get_or_create_user(db_session, TEST_CLERK_ID)
    data = (await client.get("/api/v1/analytics/daily", params={"tz": "Asia/Colombo"})).json()
    assert data["weekly_progress_score"] is None
    assert all(d["progress_score"] is None for d in data["daily"])


async def test_fourteen_day_window_covers_this_week_and_last(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    today = _local_today(COLOMBO)
    await _make_completed_session(db_session, user.id, "Science", 5, 4, _at_local(today - timedelta(days=10), 10))
    await _make_completed_session(db_session, user.id, "Science", 5, 5, _at_local(today, 10))

    data = (await client.get("/api/v1/analytics/daily", params={"days": 14, "tz": "Asia/Colombo"})).json()

    assert len(data["daily"]) == 14
    assert data["start_date"] == (today - timedelta(days=13)).isoformat()
    previous_week, this_week = data["daily"][:7], data["daily"][7:]
    assert sum(d["quizzes_completed"] for d in previous_week) == 1
    assert sum(d["quizzes_completed"] for d in this_week) == 1
