"""
tests/test_analytics.py
─────────────────────────
Tests for GET /analytics/me, focused on the weighted overall-accuracy fix
and the new summary fields (total_questions_attempted, total_correct_answers,
total_incorrect_answers, total_unanswered_questions).

Each test builds QuizSession / Question / QuestionAttempt rows directly
against the test DB (see conftest.py) rather than going through
POST /quiz/generate — quiz generation calls the real Groq API, which is
unrelated to what's being tested here and would make these tests slow,
flaky, and dependent on network/API keys.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _make_question(db: AsyncSession, subject: str, i: int) -> Question:
    q = Question(
        question=f"Q{i}?",
        options=["A", "B", "C", "D"],
        correct_answer="A",
        subject=subject,
        lesson="Test Lesson",
        difficulty="easy",
    )
    db.add(q)
    await db.flush()
    return q


async def _make_session_with_attempts(
    db: AsyncSession,
    user_id: int,
    subject: str,
    question_count: int,
    results: list[bool],
    response_time: float = 5.0,
) -> QuizSession:
    """
    Creates a QuizSession with `question_count` questions, but only records
    a QuestionAttempt for the first len(results) of them (True/False = correct
    or incorrect). Any remaining questions (question_count - len(results))
    get NO attempt row at all, simulating unanswered questions per the
    session-vs-attempts comparison the service uses.
    """
    session = QuizSession(
        user_id=user_id,
        subject=subject,
        lesson="Test Lesson",
        difficulty="easy",
        question_count=question_count,
    )
    db.add(session)
    await db.flush()

    for i, is_correct in enumerate(results):
        question = await _make_question(db, subject, i)
        db.add(QuestionAttempt(
            session_id=session.id,
            question_id=question.id,
            selected_answer="A" if is_correct else "B",
            correct=is_correct,
            response_time=response_time,
        ))

    await db.commit()
    return session


# ─────────────────────────────────────────────────────────────────────────────
# 1. No quiz data
# ─────────────────────────────────────────────────────────────────────────────

async def test_analytics_no_quiz_data(client, db_session):
    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["overall_accuracy"] == 0
    assert data["overall_avg_response_time"] == 0
    assert data["total_sessions"] == 0
    assert data["total_questions_attempted"] == 0
    assert data["total_correct_answers"] == 0
    assert data["total_incorrect_answers"] == 0
    assert data["total_unanswered_questions"] == 0
    assert data["subjects"] == []
    assert data["strong_subjects"] == []
    assert data["weak_subjects"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. One completed quiz
# ─────────────────────────────────────────────────────────────────────────────

async def test_analytics_one_completed_quiz(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # 5 questions, 3 correct + 2 incorrect, nothing unanswered
    await _make_session_with_attempts(
        db_session, user.id, "Mathematics", question_count=5,
        results=[True, True, True, False, False],
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_sessions"] == 1
    assert data["total_questions_attempted"] == 5
    assert data["total_correct_answers"] == 3
    assert data["total_incorrect_answers"] == 2
    assert data["total_unanswered_questions"] == 0
    assert data["overall_accuracy"] == 60.0  # 3/5 * 100
    assert data["overall_avg_response_time"] == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multiple subjects with different question counts — proves the weighted fix
# ─────────────────────────────────────────────────────────────────────────────

async def test_analytics_weighted_across_subjects_with_different_counts(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    # Subject A: 2 attempts, both correct -> 100% subject accuracy
    await _make_session_with_attempts(
        db_session, user.id, "Subject A", question_count=2,
        results=[True, True],
    )
    # Subject B: 8 attempts, 2 correct -> 25% subject accuracy
    await _make_session_with_attempts(
        db_session, user.id, "Subject B", question_count=8,
        results=[True, True, False, False, False, False, False, False],
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    # Naive (buggy) average of subject accuracies would be (100 + 25) / 2 = 62.5.
    # Correct weighted figure: total_correct / total_attempted = 4 / 10 = 40.0.
    assert data["total_sessions"] == 2
    assert data["total_questions_attempted"] == 10
    assert data["total_correct_answers"] == 4
    assert data["total_incorrect_answers"] == 6
    assert data["total_unanswered_questions"] == 0
    assert data["overall_accuracy"] == 40.0
    assert data["overall_accuracy"] != 62.5


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sessions containing unanswered questions
# ─────────────────────────────────────────────────────────────────────────────

async def test_analytics_with_unanswered_questions(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # 5 questions in the session, but only 3 got an attempt row at all
    # (2 correct, 1 incorrect) -> 2 unanswered.
    await _make_session_with_attempts(
        db_session, user.id, "Science", question_count=5,
        results=[True, True, False],
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_sessions"] == 1
    assert data["total_unanswered_questions"] == 2
    assert data["total_correct_answers"] == 2
    assert data["total_incorrect_answers"] == 1
    # Denominator includes the unanswered questions (5), not just graded ones (3).
    assert data["total_questions_attempted"] == 5
    assert data["overall_accuracy"] == 40.0  # 2/5 * 100, NOT 2/3 * 100


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shuffle Mode submission attributes analytics to each REAL subject, not
#    the session's own "Mixed" literal.
# ─────────────────────────────────────────────────────────────────────────────

async def test_shuffle_submission_splits_analytics_per_real_subject(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    # A Shuffle Mode session: QuizSession.subject is the "Mixed" literal
    # (set at generation time whenever a shuffle quiz spans >1 subject), but
    # each individual Question keeps its own real subject.
    session = QuizSession(user_id=user.id, subject="Mixed", lesson="Mixed", difficulty="easy", question_count=4)
    db_session.add(session)
    await db_session.flush()

    math_q1 = await _make_question(db_session, "Mathematics", 1)
    math_q2 = await _make_question(db_session, "Mathematics", 2)
    sci_q1 = await _make_question(db_session, "Science", 3)
    sci_q2 = await _make_question(db_session, "Science", 4)
    await db_session.commit()

    resp = await client.post(
        "/api/v1/quiz/submit",
        json={
            "session_id": session.id,
            "ended_by": "submitted",
            "answers": [
                {"question_id": math_q1.id, "selected_answer": "A", "response_time": 4.0},
                {"question_id": math_q2.id, "selected_answer": "A", "response_time": 4.0},
                {"question_id": sci_q1.id, "selected_answer": "B", "response_time": 4.0},
                {"question_id": sci_q2.id, "selected_answer": "B", "response_time": 4.0},
            ],
        },
    )
    assert resp.status_code == 200

    analytics_resp = await client.get("/api/v1/analytics/me")
    assert analytics_resp.status_code == 200
    subjects_by_name = {s["subject"]: s for s in analytics_resp.json()["subjects"]}

    # The bug this test guards against: a naive implementation attributes
    # every subject's answers to the session's own (literal "Mixed")
    # subject instead of each question's real subject.
    assert "Mixed" not in subjects_by_name
    assert subjects_by_name["Mathematics"]["accuracy"] == 100.0
    assert subjects_by_name["Science"]["accuracy"] == 0.0
