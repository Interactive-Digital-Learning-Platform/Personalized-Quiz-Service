"""
tests/test_analytics_difficulty.py
─────────────────────────────────────
Tests for the difficulty-level analytics added to GET /analytics/me:
subjects[].difficulty_performance plus the adaptive-difficulty description
fields (current_difficulty, consecutive_strong_quizzes,
consecutive_weak_quizzes, promotion_threshold, demotion_threshold,
quizzes_required_for_promotion, promotion_progress_percentage,
next_difficulty, difficulty_status_message).

These fields must be a read-only PROJECTION of the real adaptive engine
(difficulty_service.SubjectMastery + update_subject_mastery_after_submission)
— never a second implementation of the promote/demote rules — so these tests
drive state through the actual difficulty_service functions rather than
hand-constructing SubjectMastery rows wherever avoidable, mirroring exactly
what POST /quiz/submit does (see quiz_service.submit_quiz).
"""
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.models.subject_mastery import SubjectMastery
from app.services import difficulty_service
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _submit_quiz(
    db,
    user_id: int,
    subject: str,
    difficulty: str,
    question_count: int,
    correct_count: int,
    lesson: str = "General",
    clerk_id: str = TEST_CLERK_ID,
    update_mastery: bool = True,
) -> float:
    """
    Builds one full quiz submission (session/questions/attempts/completion),
    updates Analytics (so the subject appears in `subjects`), and — unless
    `update_mastery=False` — feeds the result into
    difficulty_service.update_subject_mastery_after_submission(), exactly as
    POST /quiz/submit does. Returns the accuracy used.
    """
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
    if update_mastery:
        await difficulty_service.update_subject_mastery_after_submission(db, user_id, subject, accuracy)

    return accuracy


def _subject(data: dict, subject: str) -> dict:
    return next(s for s in data["subjects"] if s["subject"] == subject)


def _difficulty_bucket(subject_data: dict, difficulty: str) -> dict:
    return next(d for d in subject_data["difficulty_performance"] if d["difficulty"] == difficulty)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Easy subject with no SubjectMastery history -> safe defaults
# ─────────────────────────────────────────────────────────────────────────────

async def test_easy_subject_with_no_mastery_history(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Attempt/Analytics data exists (so the subject appears at all), but no
    # SubjectMastery row was ever created — exercises requirement 9's "safe
    # values when SubjectMastery does not exist".
    await _submit_quiz(
        db_session, user.id, "Mathematics", "easy", 10, 7, update_mastery=False,
    )

    resp = await client.get("/api/v1/analytics/me")
    assert resp.status_code == 200
    data = resp.json()

    subj = _subject(data, "Mathematics")
    assert subj["current_difficulty"] == "easy"
    assert subj["consecutive_strong_quizzes"] == 0
    assert subj["consecutive_weak_quizzes"] == 0
    assert subj["promotion_threshold"] == 80.0
    assert subj["demotion_threshold"] == 40.0
    assert subj["quizzes_required_for_promotion"] == 2
    assert subj["promotion_progress_percentage"] == 0.0
    assert subj["next_difficulty"] == "medium"
    assert subj["difficulty_status_message"] == "At easy difficulty. Keep practicing to progress."

    bucket = _difficulty_bucket(subj, "easy")
    assert bucket["total_attempted"] == 10
    assert bucket["total_correct"] == 7
    assert bucket["accuracy"] == 70.0
    assert bucket["completed_sessions"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. One strong quiz -> streak progress, not yet promoted
# ─────────────────────────────────────────────────────────────────────────────

async def test_one_strong_quiz(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert difficulty_service.PROMOTE_ACCURACY_THRESHOLD == 80.0  # default assumed here
    await _submit_quiz(db_session, user.id, "Mathematics", "easy", 10, 8)  # 80%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    assert subj["current_difficulty"] == "easy"  # PROMOTE_STREAK_REQUIRED=2, only 1 so far
    assert subj["consecutive_strong_quizzes"] == 1
    assert subj["consecutive_weak_quizzes"] == 0
    assert subj["next_difficulty"] == "medium"
    assert subj["promotion_progress_percentage"] == 50.0
    assert subj["difficulty_status_message"] == "1 more strong quiz needed to reach medium difficulty."


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two consecutive strong quizzes -> promotion actually happens
# ─────────────────────────────────────────────────────────────────────────────

async def test_two_consecutive_strong_quizzes_promotes(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit_quiz(db_session, user.id, "Mathematics", "easy", 10, 8)  # 80%
    await _submit_quiz(db_session, user.id, "Mathematics", "easy", 10, 9)  # 90%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    # Promoted after the 2nd consecutive strong quiz; streak resets.
    assert subj["current_difficulty"] == "medium"
    assert subj["consecutive_strong_quizzes"] == 0
    assert subj["next_difficulty"] == "hard"
    assert subj["promotion_progress_percentage"] == 0.0
    assert subj["difficulty_status_message"] == "At medium difficulty. Keep practicing to progress."

    # Both actual quizzes were taken at "easy" — difficulty_performance
    # reflects what was REALLY attempted, independent of current_difficulty
    # (which describes the NEXT quiz).
    assert len(subj["difficulty_performance"]) == 1
    easy_bucket = _difficulty_bucket(subj, "easy")
    assert easy_bucket["total_attempted"] == 20
    assert easy_bucket["total_correct"] == 17
    assert easy_bucket["completed_sessions"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Weak quiz demotion
# ─────────────────────────────────────────────────────────────────────────────

async def test_weak_quiz_demotes(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    assert difficulty_service.DEMOTE_ACCURACY_THRESHOLD == 40.0  # default assumed here
    # Seed the subject as already having been promoted to "medium" previously.
    db_session.add(SubjectMastery(
        user_id=user.id, subject="Mathematics", difficulty="medium",
        last_accuracy=85.0, consecutive_strong=0, consecutive_weak=0,
    ))
    await db_session.commit()

    await _submit_quiz(db_session, user.id, "Mathematics", "medium", 10, 3)  # 30%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    assert subj["current_difficulty"] == "easy"  # demoted from medium
    assert subj["consecutive_weak_quizzes"] == 0  # reset immediately after demoting
    assert subj["consecutive_strong_quizzes"] == 0
    assert subj["next_difficulty"] == "medium"
    assert subj["difficulty_status_message"] == "At easy difficulty. Keep practicing to progress."

    # The actual quiz was taken at "medium" difficulty.
    medium_bucket = _difficulty_bucket(subj, "medium")
    assert medium_bucket["total_attempted"] == 10
    assert medium_bucket["total_correct"] == 3
    assert medium_bucket["accuracy"] == 30.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Subject already at hard -> no further promotion possible
# ─────────────────────────────────────────────────────────────────────────────

async def test_subject_already_at_hard(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    db_session.add(SubjectMastery(
        user_id=user.id, subject="Mathematics", difficulty="hard",
        last_accuracy=90.0, consecutive_strong=1, consecutive_weak=0,
    ))
    await db_session.commit()

    await _submit_quiz(db_session, user.id, "Mathematics", "hard", 10, 9, update_mastery=False)

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    assert subj["current_difficulty"] == "hard"
    assert subj["next_difficulty"] == "hard"  # clamped — no level above hard
    assert subj["promotion_progress_percentage"] == 0.0
    assert subj["difficulty_status_message"] == "Already at the highest difficulty (hard)."


# ─────────────────────────────────────────────────────────────────────────────
# 6. Separate mastery states for Mathematics and Science
# ─────────────────────────────────────────────────────────────────────────────

async def test_separate_mastery_states_per_subject(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Mathematics: one strong quiz — still climbing toward promotion.
    await _submit_quiz(db_session, user.id, "Mathematics", "easy", 10, 8)  # 80%
    # Science: two consecutive strong quizzes — already promoted.
    await _submit_quiz(db_session, user.id, "Science", "easy", 10, 8)   # 80%
    await _submit_quiz(db_session, user.id, "Science", "easy", 10, 9)   # 90%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()

    maths = _subject(data, "Mathematics")
    assert maths["current_difficulty"] == "easy"
    assert maths["consecutive_strong_quizzes"] == 1

    science = _subject(data, "Science")
    assert science["current_difficulty"] == "medium"
    assert science["consecutive_strong_quizzes"] == 0

    # No cross-contamination between the two subjects' difficulty_performance.
    assert _difficulty_bucket(maths, "easy")["total_attempted"] == 10
    assert _difficulty_bucket(science, "easy")["total_attempted"] == 20
