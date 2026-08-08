"""
tests/test_analytics_topics.py
─────────────────────────────────
Tests for the per-topic breakdown added to GET /analytics/me
(subjects[].topics), derived from each Question's own `lesson` field rather
than QuizSession.lesson — a single quiz can span multiple lessons.

Sessions/questions/attempts are built directly against the test DB rather
than via POST /quiz/generate or /quiz/submit (real Groq calls, unrelated to
what's tested here — see test_analytics.py for the same rationale).
"""
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


async def _make_completed_session_with_topics(
    db,
    user_id: int,
    subject: str,
    topic_results: list[tuple[str, bool]],
    response_time: float = 5.0,
    clerk_id: str = TEST_CLERK_ID,
) -> QuizSession:
    """
    `topic_results`: one (lesson, is_correct) pair per question/attempt —
    lets a single session span multiple lessons, matching how quizzes are
    actually generated (each question independently assigned a lesson).
    """
    session = QuizSession(
        user_id=user_id,
        subject=subject,
        lesson="Mixed",
        difficulty="easy",
        question_count=len(topic_results),
    )
    db.add(session)
    await db.flush()

    correct_count = 0
    for i, (lesson, is_correct) in enumerate(topic_results):
        question = await _make_question(db, subject, lesson, i)
        db.add(QuestionAttempt(
            session_id=session.id,
            question_id=question.id,
            selected_answer="A" if is_correct else "B",
            correct=is_correct,
            response_time=response_time,
        ))
        if is_correct:
            correct_count += 1

    db.add(QuizCompletion(
        session_id=session.id,
        ended_by="submitted",
        total_time=response_time * len(topic_results),
        score=float(correct_count),
        accuracy=round(correct_count / len(topic_results) * 100.0, 2),
        correct_count=correct_count,
        total_questions=len(topic_results),
    ))
    await db.commit()

    # Mirrors what the real POST /quiz/submit route does after submit_quiz():
    # populates the Analytics row that `subjects` in the response is built
    # from. Without this, the session/attempts/completion would exist but
    # the subject would never appear in the response at all.
    await update_analytics_after_submission(db, clerk_id, session.id)

    return session


def _topics_by_name(data: dict, subject: str) -> dict:
    subj = next(s for s in data["subjects"] if s["subject"] == subject)
    return {t["topic"]: t for t in subj["topics"]}


# ─────────────────────────────────────────────────────────────────────────────
# 1. One subject with several topics
# ─────────────────────────────────────────────────────────────────────────────

async def test_one_subject_with_several_topics(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _make_completed_session_with_topics(
        db_session, user.id, "Mathematics",
        topic_results=[
            ("Linear Equations", True), ("Linear Equations", True), ("Linear Equations", False),
            ("Quadratic Equations", True), ("Quadratic Equations", True), ("Quadratic Equations", True),
            ("Geometry", False), ("Geometry", False), ("Geometry", False), ("Geometry", True),
        ],
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    topics = _topics_by_name(data, "Mathematics")
    assert set(topics.keys()) == {"Linear Equations", "Quadratic Equations", "Geometry"}

    linear = topics["Linear Equations"]
    assert linear["total_attempted"] == 3
    assert linear["total_correct"] == 2
    assert linear["total_incorrect"] == 1
    assert linear["accuracy"] == 66.67
    assert linear["status"] == "developing"

    quadratic = topics["Quadratic Equations"]
    assert quadratic["total_attempted"] == 3
    assert quadratic["accuracy"] == 100.0
    assert quadratic["status"] == "strong"

    geometry = topics["Geometry"]
    assert geometry["total_attempted"] == 4
    assert geometry["total_correct"] == 1
    assert geometry["accuracy"] == 25.0
    assert geometry["status"] == "weak"

    # weak_topic is derived from the lowest-accuracy ELIGIBLE topic.
    subj = next(s for s in data["subjects"] if s["subject"] == "Mathematics")
    assert subj["weak_topic"] == "Geometry"

    # Returned topics list is sorted ascending by accuracy (weakest first).
    subj_topics = subj["topics"]
    accuracies = [t["accuracy"] for t in subj_topics]
    assert accuracies == sorted(accuracies)
    assert subj_topics[0]["topic"] == "Geometry"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mixed lessons within one quiz
# ─────────────────────────────────────────────────────────────────────────────

async def test_mixed_lessons_within_one_quiz_session(client, db_session):
    """
    A single QuizSession (session.lesson == "Mixed") whose individual
    questions belong to different lessons must still be broken out per
    question-lesson, not lumped under the session-level label.
    """
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = await _make_completed_session_with_topics(
        db_session, user.id, "Science",
        topic_results=[("Photosynthesis", True), ("Atomic Structure", False), ("Photosynthesis", True)],
    )
    assert session.lesson == "Mixed"  # the session-level label, unused for topic stats

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    topics = _topics_by_name(data, "Science")
    assert "Mixed" not in topics
    assert topics["Photosynthesis"]["total_attempted"] == 2
    assert topics["Photosynthesis"]["total_correct"] == 2
    assert topics["Atomic Structure"]["total_attempted"] == 1
    assert topics["Atomic Structure"]["total_correct"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing topic value
# ─────────────────────────────────────────────────────────────────────────────

async def test_missing_topic_value_normalizes_to_unknown_topic(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Question.lesson is NOT NULL in the schema, so "missing" in practice
    # means blank/whitespace-only, not a true NULL.
    await _make_completed_session_with_topics(
        db_session, user.id, "History",
        topic_results=[("", True), ("   ", False), ("British Colonial Era", True)],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    topics = _topics_by_name(data, "History")
    assert "Unknown Topic" in topics
    assert "" not in topics
    assert topics["Unknown Topic"]["total_attempted"] == 2
    assert topics["Unknown Topic"]["total_correct"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fewer than 3 topic attempts
# ─────────────────────────────────────────────────────────────────────────────

async def test_topic_with_fewer_than_minimum_attempts_is_insufficient_data(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert settings.ANALYTICS_TOPIC_MIN_ATTEMPTS == 3  # default assumed by this test
    await _make_completed_session_with_topics(
        db_session, user.id, "English",
        # 100% accuracy, but only 2 attempts — must NOT be "strong".
        topic_results=[("Essay Writing", True), ("Essay Writing", True)],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    topic = _topics_by_name(data, "English")["Essay Writing"]
    assert topic["total_attempted"] == 2
    assert topic["accuracy"] == 100.0
    assert topic["status"] == "insufficient_data"

    # An insufficient-data topic must not be picked as weak_topic.
    subj = next(s for s in data["subjects"] if s["subject"] == "English")
    assert subj["weak_topic"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Equal accuracy topics
# ─────────────────────────────────────────────────────────────────────────────

async def test_equal_accuracy_topics_have_deterministic_order(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Both topics: 2/4 correct = 50% accuracy exactly.
    await _make_completed_session_with_topics(
        db_session, user.id, "Geography",
        topic_results=[
            ("Rivers", True), ("Rivers", True), ("Rivers", False), ("Rivers", False),
            ("Mountains", True), ("Mountains", True), ("Mountains", False), ("Mountains", False),
        ],
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    subj = next(s for s in data["subjects"] if s["subject"] == "Geography")
    tied = [t for t in subj["topics"] if t["accuracy"] == 50.0]
    assert len(tied) == 2
    # Tie-broken alphabetically by topic name for a stable, predictable order.
    assert [t["topic"] for t in tied] == ["Mountains", "Rivers"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multiple users (isolation)
# ─────────────────────────────────────────────────────────────────────────────

async def test_topics_are_isolated_per_user(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    other_user = await get_or_create_user(db_session, "other-user")

    await _make_completed_session_with_topics(
        db_session, user.id, "Mathematics",
        topic_results=[("Algebra", True), ("Algebra", True), ("Algebra", True)],
    )
    # A different user's data, including a topic name that also exists for
    # the authenticated user — must not blend into their aggregation.
    await _make_completed_session_with_topics(
        db_session, other_user.id, "Mathematics",
        topic_results=[("Algebra", False), ("Algebra", False), ("Algebra", False)],
        clerk_id="other-user",
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    topic = _topics_by_name(data, "Mathematics")["Algebra"]
    assert topic["total_attempted"] == 3
    assert topic["total_correct"] == 3
    assert topic["accuracy"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. No attempts
# ─────────────────────────────────────────────────────────────────────────────

async def test_no_attempts_returns_no_subjects_or_topics(client, db_session):
    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["subjects"] == []
