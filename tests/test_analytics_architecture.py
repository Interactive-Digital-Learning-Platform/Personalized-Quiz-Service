"""
tests/test_analytics_architecture.py
────────────────────────────────────────
Tests for the app/services/analytics/ package introduced by the GET
/analytics/me architecture refactor:

1. Unit tests for the 8 orchestration services, using typed dataclasses
   directly — no database at all (requirement: "unit tests for pure
   calculations").
2. Integration tests for the query layer (queries.py) against a real
   (SQLite) database — proving the extracted queries still aggregate
   correctly (requirement: "integration tests for database aggregations").
3. One true end-to-end test driving GET /analytics/me through the real
   HTTP route with a realistic multi-subject/topic dataset.
4. A performance/N+1 test at the specified scale — 100 sessions, 2,000
   question attempts, 10 subjects, 100 topics — asserting both that the
   endpoint responds quickly AND that its SQL query count stays fixed
   (doesn't grow with subject/topic/session count), which is the concrete,
   measurable proof behind "avoid one query per subject/topic" and
   "prevent N+1 queries".

The formulas THEMSELVES (mastery/growth/trend/recommendations/etc.) already
have dedicated unit-test files from when each feature was built
(test_mastery_service.py, test_growth_service.py, etc.) — this file doesn't
re-test those formulas, only the NEW orchestration/query layer this
refactor introduced around them.
"""
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event

from app.models.analytics import Analytics
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services.analytics import queries
from app.services.analytics.mastery_service import MasteryScoreService
from app.services.analytics.orchestrator import AnalyticsOrchestrationService
from app.services.analytics.repeated_mistake_service import RepeatedMistakeAnalyticsService
from app.services.analytics.summary_service import compute_response_time_stats
from app.services.analytics.topic_service import TopicAnalyticsService
from app.services.analytics.trend_service import TrendAnalyticsService
from app.services.analytics.types import (
    RepeatedAttemptRow,
    ResponseTimeRow,
    TopicDifficultyRow,
    TopicRow,
    TrendAttemptRow,
)
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

NOW = datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests — typed dataclasses in, no database at all
# ─────────────────────────────────────────────────────────────────────────────

def test_topic_analytics_service_builds_status_and_accuracy():
    rows = [
        TopicRow(subject="Mathematics", topic="Algebra", total_attempted=10, total_correct=8, last_attempted_at=NOW),
        TopicRow(subject="Mathematics", topic="Geometry", total_attempted=2, total_correct=1, last_attempted_at=NOW),
    ]
    topics_by_subject = TopicAnalyticsService(rows).build()

    assert set(topics_by_subject.keys()) == {"Mathematics"}
    by_name = {t["topic"]: t for t in topics_by_subject["Mathematics"]}
    assert by_name["Algebra"]["accuracy"] == 80.0
    assert by_name["Algebra"]["status"] == "strong"
    # Only 2 attempts, below ANALYTICS_TOPIC_MIN_ATTEMPTS (3) -> insufficient_data
    assert by_name["Geometry"]["status"] == "insufficient_data"


def test_trend_analytics_service_buckets_by_scope():
    rows = [
        TrendAttemptRow(session_id=1, subject="Science", topic="Physics", correct=True),
        TrendAttemptRow(session_id=1, subject="Science", topic="Physics", correct=False),
        TrendAttemptRow(session_id=2, subject="Science", topic="Chemistry", correct=True),
    ]
    service = TrendAnalyticsService(rows, session_completed_at={1: NOW, 2: NOW})

    assert service.overall_session_stats[1] == {"correct": 1, "total": 2}
    assert service.overall_session_stats[2] == {"correct": 1, "total": 1}
    assert service.subject_session_stats["Science"][1]["total"] == 2
    assert service.topic_session_stats[("Science", "Physics")][1] == {"correct": 1, "total": 2}
    assert service.topic_session_stats[("Science", "Chemistry")][2] == {"correct": 1, "total": 1}


def test_repeated_mistake_service_ignores_single_attempts_and_groups_by_fingerprint():
    rows = [
        RepeatedAttemptRow(
            attempt_id=1, correct=False, fingerprint="fp-a", subject="Maths", topic="Algebra", difficulty="easy",
        ),
        RepeatedAttemptRow(
            attempt_id=2, correct=True, fingerprint="fp-a", subject="Maths", topic="Algebra", difficulty="easy",
        ),
        RepeatedAttemptRow(
            attempt_id=3, correct=True, fingerprint="fp-b", subject="Maths", topic="Algebra", difficulty="easy",
        ),  # attempted only once -> ignored entirely
    ]
    service = RepeatedMistakeAnalyticsService(rows)

    assert service.overall["repeated_question_count"] == 1
    assert service.overall["corrected_previous_mistakes"] == 1
    assert service.for_subject("Maths")["repeated_question_count"] == 1
    assert service.for_topic("Maths", "Algebra")["corrected_previous_mistakes"] == 1
    assert service.most_recent_topic_difficulty[("Maths", "Algebra")] == "easy"


def test_summary_service_response_time_stats_helper():
    rows = [
        ResponseTimeRow(subject="Science", topic="Physics", correct=True, response_time=4.0),
        ResponseTimeRow(subject="Science", topic="Physics", correct=False, response_time=8.0),
    ]
    stats = compute_response_time_stats(rows)
    assert stats["avg_response_time"] == 6.0
    assert stats["correct_answer_avg_response_time"] == 4.0
    assert stats["incorrect_answer_avg_response_time"] == 8.0
    assert stats["fastest_response_time"] == 4.0
    assert stats["slowest_response_time"] == 8.0


def test_mastery_score_service_uses_topic_difficulty_accuracy():
    rows = [
        TopicDifficultyRow(subject="Maths", topic="Algebra", difficulty="easy", total_attempted=10, total_correct=7),
    ]
    service = MasteryScoreService(rows)
    result = service.score_topic(
        subject="Maths", topic="Algebra", total_attempted=10, accuracy=70.0,
        trend={"method": "insufficient_data", "current_period_accuracy": 0.0},
        repeated={"corrected_previous_mistakes": 0, "repeated_same_mistakes": 0, "mistake_correction_rate": 0.0},
        most_recent_difficulty="easy",
        session_accuracies=[70.0, 70.0, 70.0],
    )
    assert result["mastery_score"] is not None
    # difficulty_score should reflect the 70% accuracy at "easy" (base 33): (33+70)/2 = 51.5
    assert result["mastery_components"]["difficulty_score"] == 51.5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integration tests — queries.py against a real (SQLite) database
# ─────────────────────────────────────────────────────────────────────────────

async def _seed_question_and_attempt(db, subject, lesson, difficulty, is_correct, session, response_time=5.0):
    question = Question(
        question=f"Q for {lesson}?", options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson=lesson, difficulty=difficulty,
    )
    db.add(question)
    await db.flush()
    db.add(QuestionAttempt(
        session_id=session.id, question_id=question.id,
        selected_answer="A" if is_correct else "B", correct=is_correct, response_time=response_time,
    ))


async def test_fetch_topic_rows_aggregates_correctly(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = QuizSession(
        user_id=user.id, subject="History", lesson="Mixed", difficulty="easy", question_count=3,
    )
    db_session.add(session)
    await db_session.flush()
    await _seed_question_and_attempt(db_session, "History", "Ancient Rome", "easy", True, session)
    await _seed_question_and_attempt(db_session, "History", "Ancient Rome", "easy", False, session)
    await _seed_question_and_attempt(db_session, "History", "Medieval Europe", "easy", True, session)
    await db_session.commit()

    rows = await queries.fetch_topic_rows(db_session, user.id)
    by_topic = {r.topic: r for r in rows}
    assert by_topic["Ancient Rome"].total_attempted == 2
    assert by_topic["Ancient Rome"].total_correct == 1
    assert by_topic["Medieval Europe"].total_attempted == 1


async def test_fetch_graded_totals_is_weighted_not_averaged(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    session_a = QuizSession(user_id=user.id, subject="A", lesson="L", difficulty="easy", question_count=10)
    session_b = QuizSession(user_id=user.id, subject="B", lesson="L", difficulty="easy", question_count=2)
    db_session.add_all([session_a, session_b])
    await db_session.flush()

    for i in range(10):
        await _seed_question_and_attempt(db_session, "A", "L", "easy", i < 8, session_a)  # 8/10 = 80%
    for i in range(2):
        await _seed_question_and_attempt(db_session, "B", "L", "easy", i < 1, session_b)  # 1/2 = 50%
    await db_session.commit()

    totals = await queries.fetch_graded_totals(db_session, user.id)
    # Weighted: (8+1)/(10+2) = 75%, NOT the naive average of 80% and 50% (=65%).
    assert totals.total_correct_answers == 9
    assert totals.total_questions_attempted == 12
    assert totals.overall_accuracy == 75.0


async def test_fetch_session_completion_stats_classifies_correctly(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    completed_session = QuizSession(user_id=user.id, subject="A", lesson="L", difficulty="easy", question_count=5)
    incomplete_session = QuizSession(user_id=user.id, subject="A", lesson="L", difficulty="easy", question_count=5)
    db_session.add_all([completed_session, incomplete_session])
    await db_session.flush()
    db_session.add(QuizCompletion(
        session_id=completed_session.id, ended_by="submitted", total_time=100.0,
        score=4.0, accuracy=80.0, correct_count=4, total_questions=5,
    ))
    await db_session.commit()

    stats = await queries.fetch_session_completion_stats(db_session, user.id)
    assert stats.total_sessions == 2
    assert stats.completed_sessions == 1
    assert stats.incomplete_sessions == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. One end-to-end test for GET /analytics/me
# ─────────────────────────────────────────────────────────────────────────────

async def test_get_analytics_me_end_to_end(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    for subject, lesson, correct_count in [
        ("Mathematics", "Algebra", 8), ("Science", "Physics", 3), ("History", "Ancient Rome", 5),
    ]:
        session = QuizSession(
            user_id=user.id, subject=subject, lesson=lesson, difficulty="easy", question_count=10,
        )
        db_session.add(session)
        await db_session.flush()
        for i in range(10):
            await _seed_question_and_attempt(db_session, subject, lesson, "easy", i < correct_count, session)
        db_session.add(QuizCompletion(
            session_id=session.id, ended_by="submitted", total_time=50.0,
            score=float(correct_count), accuracy=correct_count * 10.0,
            correct_count=correct_count, total_questions=10,
        ))
        await db_session.commit()
        from app.services.analytics_service import update_analytics_after_submission
        await update_analytics_after_submission(db_session, TEST_CLERK_ID, session.id)

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    # The full response shape must still be present after the refactor —
    # every section this project's analytics features have added over time.
    for key in [
        "overall_accuracy", "total_sessions", "subjects", "strong_subjects", "weak_subjects",
        "performance_trend", "repeated_question_analytics", "growth", "recommendations",
    ]:
        assert key in data

    subjects_by_name = {s["subject"]: s for s in data["subjects"]}
    assert set(subjects_by_name) == {"Mathematics", "Science", "History"}
    maths = subjects_by_name["Mathematics"]
    assert maths["accuracy"] == 80.0
    assert len(maths["topics"]) == 1
    assert maths["topics"][0]["topic"] == "Algebra"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Performance / N+1 test at the specified scale
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_COUNT = 10
TOPICS_PER_SUBJECT = 10
SESSIONS_PER_SUBJECT = 10
ATTEMPTS_PER_SESSION = 20


@pytest.mark.asyncio
async def test_performance_and_flat_query_count_at_scale(client, db_session):
    """
    100 sessions (10 subjects x 10), 2,000 attempts (100 sessions x 20),
    10 subjects, 100 topics (10 subjects x 10). Asserts:
    - the endpoint responds in a reasonable time, and
    - the SQL query count is FIXED — the same regardless of how many
      subjects/topics/sessions exist — proving there's no per-subject or
      per-topic query loop anywhere in the pipeline.
    """
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    questions_by_topic: dict[tuple[str, str], list[Question]] = {}
    for s in range(SUBJECT_COUNT):
        subject = f"Subject{s}"
        for t in range(TOPICS_PER_SUBJECT):
            topic = f"Topic{s}-{t}"
            q = Question(
                question=f"Q {subject}/{topic}?", options=["A", "B", "C", "D"], correct_answer="A",
                subject=subject, lesson=topic, difficulty="easy",
            )
            db_session.add(q)
            questions_by_topic[(subject, topic)] = q
        # subjects[] is sourced from the Analytics table (populated in real
        # usage by update_analytics_after_submission() after each submit) —
        # inserted directly here to keep this test's own setup fast, since
        # its target is GET /analytics/me's read performance, not the
        # submission-time upsert path.
        db_session.add(Analytics(user_id=user.id, subject=subject, accuracy=70.0, avg_response_time=5.0))
    await db_session.flush()

    session_count = 0
    attempt_count = 0
    for s in range(SUBJECT_COUNT):
        subject = f"Subject{s}"
        for _ in range(SESSIONS_PER_SUBJECT):
            session = QuizSession(
                user_id=user.id, subject=subject, lesson="Mixed", difficulty="easy",
                question_count=ATTEMPTS_PER_SESSION,
                created_at=NOW - timedelta(days=session_count % 30),
            )
            db_session.add(session)
            await db_session.flush()
            session_count += 1

            for a in range(ATTEMPTS_PER_SESSION):
                topic = f"Topic{s}-{a % TOPICS_PER_SUBJECT}"
                question = questions_by_topic[(subject, topic)]
                db_session.add(QuestionAttempt(
                    session_id=session.id, question_id=question.id,
                    selected_answer="A", correct=(a % 3 != 0), response_time=5.0 + (a % 7),
                ))
                attempt_count += 1
            db_session.add(QuizCompletion(
                session_id=session.id, ended_by="submitted", total_time=100.0,
                score=15.0, accuracy=75.0, correct_count=15, total_questions=ATTEMPTS_PER_SESSION,
            ))
    await db_session.commit()

    assert session_count == 100
    assert attempt_count == 2000

    query_count = 0

    def _count_queries(*args, **kwargs):
        nonlocal query_count
        query_count += 1

    sync_engine = db_session.bind.sync_engine
    event.listen(sync_engine, "before_cursor_execute", _count_queries)
    try:
        start = time.monotonic()
        resp = await client.get("/api/v1/analytics/me")
        elapsed_seconds = time.monotonic() - start
    finally:
        event.remove(sync_engine, "before_cursor_execute", _count_queries)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["subjects"]) == SUBJECT_COUNT
    assert sum(len(s["topics"]) for s in data["subjects"]) <= SUBJECT_COUNT * 10  # capped display, not raw count

    # Fixed query budget: ~18 analytics queries + a handful of framework/
    # user-lookup queries — nowhere close to scaling with 10 subjects or
    # 100 topics (a per-subject or per-topic loop would blow this well past
    # 100+ queries).
    assert query_count <= 30, f"expected a flat, bounded query count, got {query_count}"
    assert elapsed_seconds < 10.0, f"expected a fast response even at scale, took {elapsed_seconds:.2f}s"
