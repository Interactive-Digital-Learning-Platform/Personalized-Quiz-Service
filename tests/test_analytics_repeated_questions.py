"""
tests/test_analytics_repeated_questions.py
─────────────────────────────────────────────
Tests for the repeated-question / repeated-mistake analytics added to
GET /analytics/me: repeated_question_analytics at the overall, subject, and
topic levels.

Repeated questions are identified by Question.question_fingerprint (a
deterministic hash of normalized subject/lesson/question text — see
app/services/question_fingerprint.py), since this app generates fresh
Question rows on every quiz rather than reusing one row's ID. Each test
below builds attempts directly against the test DB — see test_analytics.py
for the same rationale (avoids the real Groq/full-submission pipeline,
which is unrelated to what's tested here).

Chronological ordering in the real implementation is
(QuizSession.created_at, QuestionAttempt.id) — QuestionAttempt has no
per-attempt timestamp of its own — so each attempt below lives in its own
session with an explicit, strictly increasing `created_at`.
"""
from datetime import datetime, timedelta, timezone

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

NOW = datetime.now(timezone.utc)


async def _make_question(db, subject: str, lesson: str, text: str) -> Question:
    q = Question(
        question=text, options=["A", "B", "C", "D"], correct_answer="A",
        subject=subject, lesson=lesson, difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_attempt(
    db, user_id: int, question: Question, is_correct: bool, when: datetime,
    clerk_id: str = TEST_CLERK_ID,
) -> None:
    """One session containing exactly one attempt on `question`, at a
    precisely controlled `created_at` so chronological order is unambiguous."""
    session = QuizSession(
        user_id=user_id, subject=question.subject, lesson=question.lesson,
        difficulty="easy", question_count=1, created_at=when,
    )
    db.add(session)
    await db.flush()
    db.add(QuestionAttempt(
        session_id=session.id, question_id=question.id,
        selected_answer="A" if is_correct else "B", correct=is_correct, response_time=5.0,
    ))
    db.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=5.0,
        score=1.0 if is_correct else 0.0, accuracy=100.0 if is_correct else 0.0,
        correct_count=1 if is_correct else 0, total_questions=1,
    ))
    await db.commit()
    await update_analytics_after_submission(db, clerk_id, session.id)


def _subject(data: dict, subject: str) -> dict:
    return next(s for s in data["subjects"] if s["subject"] == subject)


def _topic(data: dict, subject: str, topic: str) -> dict:
    return next(t for t in _subject(data, subject)["topics"] if t["topic"] == topic)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Incorrect followed by correct -> corrected_previous_mistakes
# ─────────────────────────────────────────────────────────────────────────────

async def test_incorrect_followed_by_correct(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    q1 = await _make_question(db_session, "Mathematics", "Algebra", "What is 2 + 2?")
    q2 = await _make_question(db_session, "Mathematics", "Algebra", "What is 2 + 2?")  # separate row, same content
    await _make_attempt(db_session, user.id, q1, False, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, q2, True, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    for stats in [
        data["repeated_question_analytics"],
        _subject(data, "Mathematics")["repeated_question_analytics"],
        _topic(data, "Mathematics", "Algebra")["repeated_question_analytics"],
    ]:
        assert stats["repeated_question_count"] == 1
        assert stats["repeated_correct_count"] == 1
        assert stats["repeated_incorrect_count"] == 0
        assert stats["corrected_previous_mistakes"] == 1
        assert stats["repeated_same_mistakes"] == 0
        assert stats["mistake_correction_rate"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Incorrect followed by incorrect -> repeated_same_mistakes
# ─────────────────────────────────────────────────────────────────────────────

async def test_incorrect_followed_by_incorrect(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session, "Science", "Physics", "What is gravity?")
    await _make_attempt(db_session, user.id, question, False, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, question, False, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    stats = data["repeated_question_analytics"]
    assert stats["repeated_question_count"] == 1
    assert stats["repeated_correct_count"] == 0
    assert stats["repeated_incorrect_count"] == 1
    assert stats["corrected_previous_mistakes"] == 0
    assert stats["repeated_same_mistakes"] == 1
    assert stats["mistake_correction_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correct followed by correct -> not a mistake event either way
# ─────────────────────────────────────────────────────────────────────────────

async def test_correct_followed_by_correct(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session, "History", "Ancient Rome", "Who founded Rome?")
    await _make_attempt(db_session, user.id, question, True, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, question, True, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    stats = data["repeated_question_analytics"]
    assert stats["repeated_question_count"] == 1
    assert stats["repeated_correct_count"] == 1
    assert stats["repeated_incorrect_count"] == 0
    assert stats["corrected_previous_mistakes"] == 0
    assert stats["repeated_same_mistakes"] == 0
    # 0/0 -> defined as 0.0, not an error.
    assert stats["mistake_correction_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Three or more attempts -> each adjacent pair evaluated independently
# ─────────────────────────────────────────────────────────────────────────────

async def test_three_or_more_attempts(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    question = await _make_question(db_session, "English", "Grammar", "Identify the verb.")
    # incorrect -> incorrect -> correct
    await _make_attempt(db_session, user.id, question, False, NOW - timedelta(days=3))
    await _make_attempt(db_session, user.id, question, False, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, question, True, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    stats = data["repeated_question_analytics"]
    # 2 attempts after the first (attempts #2 and #3).
    assert stats["repeated_question_count"] == 2
    assert stats["repeated_correct_count"] == 1     # attempt #3
    assert stats["repeated_incorrect_count"] == 1   # attempt #2
    # pair(1,2)=incorrect->incorrect => repeated_same_mistakes
    # pair(2,3)=incorrect->correct   => corrected_previous_mistakes
    assert stats["corrected_previous_mistakes"] == 1
    assert stats["repeated_same_mistakes"] == 1
    assert stats["mistake_correction_rate"] == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Identical question text in different subjects -> NOT grouped together
# ─────────────────────────────────────────────────────────────────────────────

async def test_identical_text_in_different_subjects_not_grouped(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    same_text = "What is the capital?"
    q_maths = await _make_question(db_session, "Mathematics", "Geometry", same_text)
    q_science = await _make_question(db_session, "Science", "Geography", same_text)
    # Each attempted only once — even if they WERE (incorrectly) grouped
    # together, that group would still have exactly 2 attempts and should
    # NOT be ignored; the fact it nets to 0 here proves they were correctly
    # treated as two separate, single-attempt (thus ignored) questions.
    await _make_attempt(db_session, user.id, q_maths, False, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, q_science, True, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    stats = data["repeated_question_analytics"]
    assert stats["repeated_question_count"] == 0
    assert stats["corrected_previous_mistakes"] == 0
    assert stats["repeated_same_mistakes"] == 0
    assert stats["mistake_correction_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Different users receiving the same question -> fully isolated
# ─────────────────────────────────────────────────────────────────────────────

async def test_different_users_same_question_isolated(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    other_user = await get_or_create_user(db_session, "other-user")

    question = await _make_question(db_session, "Geography", "Rivers", "What is the longest river?")
    # The authenticated user repeats it: incorrect -> correct.
    await _make_attempt(db_session, user.id, question, False, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, question, True, NOW - timedelta(days=1))
    # A different user attempts the SAME question, but only once.
    await _make_attempt(
        db_session, other_user.id, question, False, NOW - timedelta(days=1, hours=12),
        clerk_id="other-user",
    )

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    # The authenticated user's own repeat is counted...
    stats = data["repeated_question_analytics"]
    assert stats["repeated_question_count"] == 1
    assert stats["corrected_previous_mistakes"] == 1
    assert stats["repeated_same_mistakes"] == 0
    # ...and is NOT inflated by the other user's single attempt on the same
    # underlying question (which, on its own, wouldn't even qualify as a
    # repeat for that other user).


# ─────────────────────────────────────────────────────────────────────────────
# 7. No repeated questions -> everything zero
# ─────────────────────────────────────────────────────────────────────────────

async def test_no_repeated_questions(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    q1 = await _make_question(db_session, "Art", "Painting", "Who painted the Mona Lisa?")
    q2 = await _make_question(db_session, "Art", "Sculpture", "Who sculpted David?")
    await _make_attempt(db_session, user.id, q1, True, NOW - timedelta(days=2))
    await _make_attempt(db_session, user.id, q2, False, NOW - timedelta(days=1))

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    stats = data["repeated_question_analytics"]
    assert stats["repeated_question_count"] == 0
    assert stats["repeated_correct_count"] == 0
    assert stats["repeated_incorrect_count"] == 0
    assert stats["corrected_previous_mistakes"] == 0
    assert stats["repeated_same_mistakes"] == 0
    assert stats["mistake_correction_rate"] == 0.0
