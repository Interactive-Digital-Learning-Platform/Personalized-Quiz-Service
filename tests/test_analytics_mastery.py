"""
tests/test_analytics_mastery.py
──────────────────────────────────
Integration tests for mastery_score/mastery_level/mastery_components on
subjects[] and subjects[].topics[] in GET /analytics/me — the formula itself
is unit-tested independently in tests/test_mastery_service.py; these tests
verify it's wired up correctly end-to-end (real DB, real HTTP response).
"""
from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services import mastery_service
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _submit_quiz(
    db, user_id: int, subject: str, lesson: str, difficulty: str,
    question_count: int, correct_count: int, clerk_id: str = TEST_CLERK_ID,
) -> None:
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty=difficulty,
        question_count=question_count,
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


def _subject(data: dict, subject: str) -> dict:
    return next(s for s in data["subjects"] if s["subject"] == subject)


def _topic(data: dict, subject: str, topic: str) -> dict:
    return next(t for t in _subject(data, subject)["topics"] if t["topic"] == topic)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-level mastery
# ─────────────────────────────────────────────────────────────────────────────

async def test_subject_mastery_computed_with_sufficient_data(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_MASTERY_MIN_ATTEMPTS == 5  # default assumed here
    await _submit_quiz(db_session, user.id, "Mathematics", "Algebra", "easy", 10, 7)

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()
    subj = _subject(data, "Mathematics")

    assert subj["mastery_score"] is not None
    assert subj["mastery_level"] != "insufficient_data"
    assert subj["mastery_components"] is not None
    for key in [
        "accuracy_score", "recent_performance_score", "difficulty_score",
        "retention_score", "consistency_score",
    ]:
        component = subj["mastery_components"][key]
        assert 0.0 <= component <= 100.0

    # The response's own mastery_score must match what the formula produces
    # from the response's own component breakdown — i.e. the breakdown is a
    # faithful, reproducible explanation of the score, not decorative.
    components = subj["mastery_components"]
    expected = mastery_service.compute_mastery_score(
        components["accuracy_score"],
        components["recent_performance_score"],
        components["difficulty_score"],
        components["retention_score"],
        components["consistency_score"],
        accuracy_weight=settings.ANALYTICS_MASTERY_ACCURACY_WEIGHT,
        recent_performance_weight=settings.ANALYTICS_MASTERY_RECENT_PERFORMANCE_WEIGHT,
        difficulty_weight=settings.ANALYTICS_MASTERY_DIFFICULTY_WEIGHT,
        retention_weight=settings.ANALYTICS_MASTERY_RETENTION_WEIGHT,
        consistency_weight=settings.ANALYTICS_MASTERY_CONSISTENCY_WEIGHT,
    )
    assert subj["mastery_score"] == expected


async def test_subject_mastery_insufficient_data(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Only 3 graded attempts — below ANALYTICS_MASTERY_MIN_ATTEMPTS (5).
    await _submit_quiz(db_session, user.id, "Science", "Physics", "easy", 3, 2)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Science")

    assert subj["mastery_score"] is None
    assert subj["mastery_level"] == "insufficient_data"
    assert subj["mastery_components"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Topic-level mastery
# ─────────────────────────────────────────────────────────────────────────────

async def test_topic_mastery_computed_with_sufficient_data(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit_quiz(db_session, user.id, "History", "Ancient Rome", "medium", 8, 6)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    topic = _topic(data, "History", "Ancient Rome")

    assert topic["mastery_score"] is not None
    assert topic["mastery_level"] != "insufficient_data"
    assert topic["mastery_components"] is not None
    for key in [
        "accuracy_score", "recent_performance_score", "difficulty_score",
        "retention_score", "consistency_score",
    ]:
        assert 0.0 <= topic["mastery_components"][key] <= 100.0


async def test_topic_mastery_insufficient_data_independent_of_subject(client, db_session):
    """
    A subject can have plenty of OVERALL data while one specific topic within
    it still has too few attempts of its own — the topic's mastery must be
    judged on its own attempt count, not inherit sufficiency from the subject.
    """
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Plenty of data in "Geometry" (this subject's other topic)...
    await _submit_quiz(db_session, user.id, "Mathematics", "Geometry", "easy", 10, 8)
    # ...but "Trigonometry" only has 2 attempts, below the mastery minimum.
    await _submit_quiz(db_session, user.id, "Mathematics", "Trigonometry", "easy", 2, 1)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    geometry = _topic(data, "Mathematics", "Geometry")
    assert geometry["mastery_score"] is not None
    assert geometry["mastery_level"] != "insufficient_data"

    trigonometry = _topic(data, "Mathematics", "Trigonometry")
    assert trigonometry["mastery_score"] is None
    assert trigonometry["mastery_level"] == "insufficient_data"
    assert trigonometry["mastery_components"] is None

    # The subject itself has enough TOTAL attempts (12 combined) to be scored.
    subj = _subject(data, "Mathematics")
    assert subj["mastery_score"] is not None
