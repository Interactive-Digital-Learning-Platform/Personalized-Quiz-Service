"""
tests/test_analytics_recommendations.py
──────────────────────────────────────────
Integration tests for the `recommendations` array in GET /analytics/me —
the priority/dedup logic itself is unit-tested independently in
tests/test_recommendation_service.py; these tests verify it's wired up
correctly end-to-end (real DB, real HTTP response) using genuinely weak
performance data to trigger real recommendations.
"""
from datetime import datetime, timedelta, timezone

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
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


async def test_recommendations_empty_with_no_history(client, db_session):
    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    assert resp.json()["recommendations"] == []


async def test_recommendations_include_weak_topic(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # 3 attempts correct out of 10 = 30% -> "weak" topic status.
    await _submit_quiz(
        db_session, user.id, "English", "Writing Techniques", "easy", 10, 3,
        created_at=NOW - timedelta(days=1),
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    assert len(recs) >= 1
    weak = next((r for r in recs if r["type"] == "weak_topic"), None)
    assert weak is not None
    assert weak["subject"] == "English"
    assert weak["topic"] == "Writing Techniques"
    assert weak["recommended_difficulty"] == "easy"
    assert weak["supporting_metrics"]["accuracy"] == 30.0
    assert weak["supporting_metrics"]["attempts"] == 10

    # priorities are sequential and consistent with the returned order
    assert [r["priority"] for r in recs] == list(range(1, len(recs) + 1))


async def test_recommendations_capped_at_five_across_many_weak_topics(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    for i in range(8):
        await _submit_quiz(
            db_session, user.id, "Mathematics", f"WeakTopic{i}", "easy", 10, 2,
            created_at=NOW - timedelta(days=i + 1),
        )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    assert len(data["recommendations"]) == 5


async def test_recommendations_field_present_alongside_other_analytics(client, db_session):
    """Sanity check that adding `recommendations` didn't disturb the rest of
    the response shape (mastery/growth/etc. from earlier features)."""
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit_quiz(
        db_session, user.id, "Science", "Physics", "easy", 10, 8,
        created_at=NOW - timedelta(days=1),
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    assert "recommendations" in data
    assert "growth" in data
    assert data["subjects"][0]["mastery_score"] is not None
