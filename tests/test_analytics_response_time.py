"""
tests/test_analytics_response_time.py
──────────────────────────────────────
Tests for the response-time statistics added to GET /analytics/me at the
overall, subject, and topic levels: median_response_time,
fastest_response_time, slowest_response_time,
correct_answer_avg_response_time, incorrect_answer_avg_response_time,
response_time_standard_deviation, and answering_behavior.

Sessions/questions/attempts are built directly against the test DB (bypassing
the QuestionAttempt schema's `ge=0.0` validation on purpose, in the invalid-
response-time test) rather than via POST /quiz/generate or /quiz/submit —
see test_analytics.py for the same rationale.
"""
import statistics

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_question(db, subject: str, lesson: str, i: int) -> Question:
    q = Question(
        question=f"Q{i}?",
        options=["A", "B", "C", "D"],
        correct_answer="A",
        subject=subject,
        lesson=lesson,
        difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_session_with_attempts(
    db,
    user_id: int,
    subject: str,
    results: list[tuple[bool, float | None]],
    lesson: str = "General",
    clerk_id: str = TEST_CLERK_ID,
) -> QuizSession:
    """
    `results`: one (is_correct, response_time) pair per question/attempt.
    `response_time` may be None/negative/huge to simulate invalid data —
    QuestionAttempt.response_time is nullable and unconstrained at the ORM
    level (the `ge=0.0` constraint only lives on the API request schema).
    """
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty="easy",
        question_count=len(results),
    )
    db.add(session)
    await db.flush()

    correct_count = 0
    for i, (is_correct, rt) in enumerate(results):
        question = await _make_question(db, subject, lesson, i)
        db.add(QuestionAttempt(
            session_id=session.id, question_id=question.id,
            selected_answer="A" if is_correct else "B",
            correct=is_correct, response_time=rt,
        ))
        if is_correct:
            correct_count += 1

    valid_times = [rt for _, rt in results if rt is not None and rt >= 0]
    db.add(QuizCompletion(
        session_id=session.id, ended_by="submitted",
        total_time=sum(valid_times) if valid_times else 0.0,
        score=float(correct_count),
        accuracy=round(correct_count / len(results) * 100.0, 2),
        correct_count=correct_count, total_questions=len(results),
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. All-correct attempts
# ─────────────────────────────────────────────────────────────────────────────

async def test_all_correct_attempts_response_time_stats(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    times = [2.0, 4.0, 6.0, 8.0, 10.0]
    await _make_session_with_attempts(
        db_session, user.id, "Mathematics",
        results=[(True, t) for t in times],
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    expected_stddev = round(statistics.stdev(times), 3)
    assert data["median_response_time"] == 6.0
    assert data["fastest_response_time"] == 2.0
    assert data["slowest_response_time"] == 10.0
    assert data["correct_answer_avg_response_time"] == 6.0
    assert data["incorrect_answer_avg_response_time"] == 0.0
    assert data["response_time_standard_deviation"] == expected_stddev

    subj = _subject(data, "Mathematics")
    assert subj["correct_answer_avg_response_time"] == 6.0
    assert subj["incorrect_answer_avg_response_time"] == 0.0
    assert subj["median_response_time"] == 6.0
    # avg == median exactly here, so timing is neither fast nor slow.
    assert subj["answering_behavior"] == "balanced"


# ─────────────────────────────────────────────────────────────────────────────
# 2. All-incorrect attempts
# ─────────────────────────────────────────────────────────────────────────────

async def test_all_incorrect_attempts_response_time_stats(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    await _make_session_with_attempts(
        db_session, user.id, "Science",
        results=[(False, t) for t in times],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    subj = _subject(data, "Science")
    assert subj["correct_answer_avg_response_time"] == 0.0
    assert subj["incorrect_answer_avg_response_time"] == 3.0
    assert subj["median_response_time"] == 3.0
    assert subj["fastest_response_time"] == 1.0
    assert subj["slowest_response_time"] == 5.0
    assert subj["response_time_standard_deviation"] == round(statistics.stdev(times), 3)
    assert subj["accuracy"] == 0.0
    # avg (3.0) == median (3.0) exactly -> "balanced" regardless of accuracy.
    assert subj["answering_behavior"] == "balanced"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mixed attempts
# ─────────────────────────────────────────────────────────────────────────────

async def test_mixed_attempts_response_time_stats(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    results = [(True, 1.0), (True, 2.0), (True, 3.0), (False, 20.0), (False, 25.0)]
    await _make_session_with_attempts(db_session, user.id, "History", results=results)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    all_times = [t for _, t in results]
    subj = _subject(data, "History")
    assert subj["correct_answer_avg_response_time"] == 2.0
    assert subj["incorrect_answer_avg_response_time"] == 22.5
    assert subj["median_response_time"] == 3.0
    assert subj["fastest_response_time"] == 1.0
    assert subj["slowest_response_time"] == 25.0
    assert subj["response_time_standard_deviation"] == round(statistics.stdev(all_times), 3)
    assert subj["accuracy"] == 60.0
    # avg (10.2) is far above median (3.0) -> slow; accuracy 60% < 70% -> inaccurate.
    assert subj["answering_behavior"] == "slow_and_inaccurate"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Invalid response times (negative, huge, null) are excluded from stats
#    but not from attempt/accuracy counts.
# ─────────────────────────────────────────────────────────────────────────────

async def test_invalid_response_times_excluded_from_stats(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_MAX_VALID_RESPONSE_TIME_SECONDS == 600.0  # default assumed here
    results = [
        (True, 5.0),
        (True, -3.0),    # invalid: negative
        (True, 700.0),   # invalid: above max
        (True, None),    # invalid: missing
        (True, 7.0),
    ]
    await _make_session_with_attempts(
        db_session, user.id, "English", results=results, lesson="Grammar",
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    subj = _subject(data, "English")
    # Attempt/accuracy counts are response-time-agnostic: all 5 still count.
    assert subj["accuracy"] == 100.0
    topic = _topic(data, "English", "Grammar")
    assert topic["total_attempted"] == 5
    assert topic["total_correct"] == 5

    # But only the two valid times (5.0, 7.0) feed the response-time stats.
    assert subj["median_response_time"] == 6.0
    assert subj["fastest_response_time"] == 5.0
    assert subj["slowest_response_time"] == 7.0
    assert subj["correct_answer_avg_response_time"] == 6.0
    assert topic["median_response_time"] == 6.0
    assert topic["fastest_response_time"] == 5.0
    assert topic["slowest_response_time"] == 7.0
    # Only 2 valid attempts < ANALYTICS_BEHAVIOR_MIN_ATTEMPTS (5).
    assert subj["answering_behavior"] == "insufficient_data"
    assert topic["answering_behavior"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Median with even and odd record counts
# ─────────────────────────────────────────────────────────────────────────────

async def test_median_with_odd_and_even_record_counts(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Odd count (5 valid times) -> median is the middle value.
    await _make_session_with_attempts(
        db_session, user.id, "Physics",
        results=[(True, t) for t in [10.0, 20.0, 30.0, 40.0, 50.0]],
    )
    # Even count (4 valid times) -> median is the average of the two middle values.
    await _make_session_with_attempts(
        db_session, user.id, "Chemistry",
        results=[(True, t) for t in [10.0, 20.0, 30.0, 40.0]],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    assert _subject(data, "Physics")["median_response_time"] == 30.0
    assert _subject(data, "Chemistry")["median_response_time"] == 25.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Insufficient data for answering_behavior (but stats still computed)
# ─────────────────────────────────────────────────────────────────────────────

async def test_insufficient_data_for_answering_behavior(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_BEHAVIOR_MIN_ATTEMPTS == 5  # default assumed here
    await _make_session_with_attempts(
        db_session, user.id, "Geography",
        results=[(True, 5.0), (True, 10.0), (False, 15.0)],  # only 3 valid attempts
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    subj = _subject(data, "Geography")
    assert subj["answering_behavior"] == "insufficient_data"
    # Numeric stats are still real values, not zeroed out just because the
    # behavior classification is withheld.
    assert subj["median_response_time"] == 10.0
    assert subj["fastest_response_time"] == 5.0
    assert subj["slowest_response_time"] == 15.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Zero response time is valid (not treated like a missing/invalid value)
# ─────────────────────────────────────────────────────────────────────────────

async def test_zero_response_time_is_valid(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _make_session_with_attempts(
        db_session, user.id, "Art",
        results=[(True, 0.0), (True, 0.0), (True, 0.0), (True, 0.0), (True, 0.0)],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    subj = _subject(data, "Art")
    assert subj["median_response_time"] == 0.0
    assert subj["fastest_response_time"] == 0.0
    assert subj["slowest_response_time"] == 0.0
    assert subj["correct_answer_avg_response_time"] == 0.0
    assert subj["response_time_standard_deviation"] == 0.0
    # avg (0.0) <= median (0.0) -> fast; accuracy 100% -> accurate.
    assert subj["answering_behavior"] == "fast_and_accurate"
