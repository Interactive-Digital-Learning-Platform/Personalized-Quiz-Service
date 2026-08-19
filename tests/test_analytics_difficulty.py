"""
tests/test_analytics_difficulty.py
─────────────────────────────────────
Tests for the difficulty-level analytics added to GET /analytics/me:
subjects[].difficulty_performance plus the adaptive-difficulty description
fields (current_difficulty, consecutive_strong_quizzes,
consecutive_weak_quizzes, promotion_threshold, demotion_threshold,
quizzes_required_for_promotion, promotion_progress_percentage,
next_difficulty, difficulty_status_message, promotion_readiness).

These fields must be a read-only PROJECTION of the real adaptive engine
(difficulty_service.SubjectMastery + update_subject_mastery_after_submission,
which now runs on top of the Continuous Evidence-Weighted Mastery System in
difficulty_mastery_engine.py) — never a second implementation of the
promote/demote rules — so these tests drive state through the actual
difficulty_service functions rather than hand-constructing SubjectMastery
rows wherever avoidable, mirroring exactly what POST /quiz/submit does (see
quiz_service.submit_quiz). Expected mastery_score/evidence_count/confidence
values below were derived by running the real engine functions for these
exact inputs (see difficulty_mastery_engine.calculate_quiz_evidence /
update_mastery_score / calculate_confidence / determine_difficulty_transition),
not hand-guessed.
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

    graded_answers: list[difficulty_service.GradedAnswer] = []
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
        graded_answers.append(difficulty_service.GradedAnswer(
            subject=subject, lesson=lesson, difficulty=difficulty, correct=is_correct,
            response_time=5.0, fingerprint=question.question_fingerprint,
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
        await difficulty_service.update_subject_mastery_after_submission(
            db, user_id, subject, accuracy,
            session=session, ended_by="submitted", graded_answers=graded_answers,
        )

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
    # New engine: promotion_threshold is now the easy->medium mastery_score
    # threshold (65.0), not the old flat 80% accuracy streak threshold.
    assert subj["promotion_threshold"] == 65.0
    assert subj["demotion_threshold"] == 40.0
    assert subj["quizzes_required_for_promotion"] == 2
    assert subj["promotion_progress_percentage"] == 0.0
    assert subj["promotion_readiness"] == 0.0
    assert subj["next_difficulty"] == "medium"
    assert subj["difficulty_status_message"] == (
        "Keep practicing easy questions to build enough evidence for medium difficulty."
    )

    bucket = _difficulty_bucket(subj, "easy")
    assert bucket["total_attempted"] == 10
    assert bucket["total_correct"] == 7
    assert bucket["accuracy"] == 70.0
    assert bucket["completed_sessions"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. One strong quiz -> evidence/mastery progress, not yet promoted
# ─────────────────────────────────────────────────────────────────────────────

async def test_one_strong_quiz(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit_quiz(db_session, user.id, "Mathematics", "easy", 10, 8)  # 80%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    # 1 qualifying quiz isn't enough evidence (needs evidence_count>=15 and
    # >=2 qualifying completions), so no promotion yet even though this
    # quiz's accuracy alone would qualify.
    assert subj["current_difficulty"] == "easy"
    assert subj["consecutive_strong_quizzes"] == 1
    assert subj["consecutive_weak_quizzes"] == 0
    assert subj["next_difficulty"] == "medium"
    # mastery_score after 1 quiz (blended 50/50 from a neutral 50 start at
    # low evidence): 61.15 -> 61.15/65*100 = 94.08% mastery progress, but
    # evidence_count=10/15=66.67% evidence progress -- readiness is the min.
    assert subj["promotion_progress_percentage"] == 66.67
    assert subj["promotion_readiness"] == 66.67
    # evidence_count (10) is still under the easy->medium minimum (15), so
    # the message calls out the evidence gap rather than a bare percentage.
    assert subj["difficulty_status_message"] == (
        "Keep practicing easy questions to build enough evidence for medium difficulty."
    )


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

    # mastery reaches 65.72 (>=65), evidence_count=20 (>=15), and both
    # qualifying quizzes were at the easy tier (>=2 required) -> promotes.
    assert subj["current_difficulty"] == "medium"
    # consecutive_strong_quizzes is now purely secondary/informational
    # evidence -- it no longer resets when a promotion happens, since
    # promotion itself is decided by determine_difficulty_transition(),
    # not by this streak.
    assert subj["consecutive_strong_quizzes"] == 2
    assert subj["next_difficulty"] == "hard"
    assert subj["promotion_progress_percentage"] == 80.0
    assert subj["promotion_readiness"] == 80.0
    assert subj["difficulty_status_message"] == (
        "Keep practicing medium questions to build enough evidence for hard difficulty."
    )

    # Both actual quizzes were taken at "easy" — difficulty_performance
    # reflects what was REALLY attempted, independent of current_difficulty
    # (which describes the NEXT quiz).
    assert len(subj["difficulty_performance"]) == 1
    easy_bucket = _difficulty_bucket(subj, "easy")
    assert easy_bucket["total_attempted"] == 20
    assert easy_bucket["total_correct"] == 17
    assert easy_bucket["completed_sessions"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Weak quizzes demotion (needs >=2 weak results AND mastery crossing the
#    demotion threshold — a single bad quiz must not swing the difficulty)
# ─────────────────────────────────────────────────────────────────────────────

async def test_single_weak_quiz_does_not_demote(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    db_session.add(SubjectMastery(
        user_id=user.id, subject="Mathematics", difficulty="medium",
        last_accuracy=85.0, consecutive_strong=0, consecutive_weak=0,
        mastery_score=50.0, fluency_score=50.0, confidence_score=0.0, evidence_count=0,
    ))
    await db_session.commit()

    await _submit_quiz(db_session, user.id, "Mathematics", "medium", 10, 3)  # 30%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    # One weak quiz nudges mastery down (blended 50/50 from neutral 50) but
    # not below the medium->easy threshold (40.0), and there's only 1 weak
    # result so far (needs >=2) -- stays at medium.
    assert subj["current_difficulty"] == "medium"
    assert subj["consecutive_weak_quizzes"] == 1
    assert subj["consecutive_strong_quizzes"] == 0


async def test_two_weak_quizzes_demote(client, db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Seeded close to the medium->easy boundary with some prior evidence, so
    # two real weak quizzes are enough to cross it within this test.
    db_session.add(SubjectMastery(
        user_id=user.id, subject="Mathematics", difficulty="medium",
        last_accuracy=45.0, consecutive_strong=0, consecutive_weak=0,
        mastery_score=41.0, fluency_score=50.0, confidence_score=70.0, evidence_count=20,
    ))
    await db_session.commit()

    await _submit_quiz(db_session, user.id, "Mathematics", "medium", 10, 2)  # 20%
    await _submit_quiz(db_session, user.id, "Mathematics", "medium", 10, 2)  # 20%

    resp = await client.get("/api/v1/analytics/me")
    data = resp.json()
    subj = _subject(data, "Mathematics")

    # mastery crosses below 40.0 and there are now 2 weak results -> demotes.
    assert subj["current_difficulty"] == "easy"
    assert subj["consecutive_weak_quizzes"] == 2
    assert subj["consecutive_strong_quizzes"] == 0
    assert subj["next_difficulty"] == "medium"

    # The actual quizzes were taken at "medium" difficulty.
    medium_bucket = _difficulty_bucket(subj, "medium")
    assert medium_bucket["total_attempted"] == 20
    assert medium_bucket["total_correct"] == 4
    assert medium_bucket["accuracy"] == 20.0


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
    assert subj["promotion_readiness"] == 0.0
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
    assert science["consecutive_strong_quizzes"] == 2

    # No cross-contamination between the two subjects' difficulty_performance.
    assert _difficulty_bucket(maths, "easy")["total_attempted"] == 10
    assert _difficulty_bucket(science, "easy")["total_attempted"] == 20
