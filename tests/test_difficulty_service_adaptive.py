"""
tests/test_difficulty_service_adaptive.py
────────────────────────────────────────────
Integration-level tests for the Continuous Evidence-Weighted Mastery System's
DB orchestration in app/services/difficulty_service.py — new-user defaults,
soft-deleted session handling, lesson->subject roll-up, and idempotency.
Pure-math scenarios live in tests/test_difficulty_mastery_engine.py.

Sessions/questions/attempts/completions are built directly against the test
DB (same rationale as test_analytics_difficulty.py / test_quiz_session_delete.py)
so these exercise the real difficulty_service functions without depending on
the Groq pipeline.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.models.lesson_mastery import LessonMastery
from app.models.subject_mastery import SubjectMastery
from app.services import difficulty_service
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID


async def _submit(
    db,
    user_id: int,
    subject: str,
    lesson: str,
    difficulty: str,
    question_count: int,
    correct_count: int,
    ended_by: str = "submitted",
    answered_count: int | None = None,
) -> None:
    """Builds one full quiz submission and runs it through BOTH
    update_mastery_after_submission (lesson-level) and
    update_subject_mastery_after_submission (subject-level), in the same
    order quiz_service.submit_quiz() uses.
    """
    answered_count = answered_count if answered_count is not None else question_count

    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty=difficulty,
        question_count=question_count,
    )
    db.add(session)
    await db.flush()

    graded_answers: list[difficulty_service.GradedAnswer] = []
    for i in range(answered_count):
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
            lesson=lesson, difficulty=difficulty, correct=is_correct,
            response_time=5.0, fingerprint=question.question_fingerprint,
        ))

    accuracy = round(correct_count / answered_count * 100.0, 2) if answered_count else 0.0
    db.add(QuizCompletion(
        session_id=session.id, ended_by=ended_by, total_time=answered_count * 5.0,
        score=float(correct_count), accuracy=accuracy,
        correct_count=correct_count, total_questions=answered_count,
        lesson_accuracy_breakdown={lesson: {"correct": correct_count, "total": answered_count, "accuracy": accuracy}},
    ))
    await db.commit()

    await difficulty_service.update_mastery_after_submission(
        db=db, user_id=user_id, subject=subject,
        lesson_accuracy_breakdown={lesson: {"correct": correct_count, "total": answered_count, "accuracy": accuracy}},
        session=session, ended_by=ended_by, graded_answers=graded_answers,
    )
    await difficulty_service.update_subject_mastery_after_submission(
        db=db, user_id=user_id, subject=subject, accuracy=accuracy,
        session=session, ended_by=ended_by, graded_answers=graded_answers,
    )


async def _get_subject_row(db, user_id: int, subject: str) -> SubjectMastery | None:
    stmt = select(SubjectMastery).where(SubjectMastery.user_id == user_id, SubjectMastery.subject == subject)
    return (await db.execute(stmt)).scalar_one_or_none()


async def _get_lesson_row(db, user_id: int, subject: str, lesson: str) -> LessonMastery | None:
    stmt = select(LessonMastery).where(
        LessonMastery.user_id == user_id, LessonMastery.subject == subject, LessonMastery.lesson == lesson,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


# ─────────────────────────────────────────────────────────────────────────────
# New-user defaults
# ─────────────────────────────────────────────────────────────────────────────

async def test_new_user_gets_safe_defaults_before_any_submission(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    mastery = await _get_subject_row(db_session, user.id, "Mathematics")
    assert mastery is None
    # describe_subject_mastery() must handle a missing row gracefully.
    described = difficulty_service.describe_subject_mastery(mastery)
    assert described["current_difficulty"] == "easy"
    assert described["promotion_readiness"] == 0.0


async def test_first_submission_creates_row_with_correct_starting_evidence(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit(db_session, user.id, "Mathematics", "Algebra", "easy", question_count=10, correct_count=8)

    mastery = await _get_subject_row(db_session, user.id, "Mathematics")
    assert mastery is not None
    assert mastery.evidence_count == 10  # question-count-based, not quiz-count-based
    assert mastery.confidence_score > 0.0
    assert mastery.mastery_score != 50.0  # moved away from the neutral default


# ─────────────────────────────────────────────────────────────────────────────
# Small quizzes must not out-weigh larger ones (question-count-based evidence)
# ─────────────────────────────────────────────────────────────────────────────

async def test_small_perfect_quiz_does_not_outweigh_larger_imperfect_quiz(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # 100% on 2 questions should NOT move evidence_count as much as 90% on 20.
    await _submit(db_session, user.id, "Mathematics", "Algebra", "easy", question_count=2, correct_count=2)
    small_quiz_evidence = (await _get_subject_row(db_session, user.id, "Mathematics")).evidence_count
    assert small_quiz_evidence == 2

    await _submit(db_session, user.id, "Mathematics", "Algebra", "easy", question_count=20, correct_count=18)
    total_evidence = (await _get_subject_row(db_session, user.id, "Mathematics")).evidence_count
    assert total_evidence == 22


# ─────────────────────────────────────────────────────────────────────────────
# Soft-deleted sessions: mastery already applied must not be undone
# ─────────────────────────────────────────────────────────────────────────────

async def test_mastery_update_persists_after_session_is_later_soft_deleted(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit(db_session, user.id, "Mathematics", "Algebra", "easy", question_count=10, correct_count=9)

    mastery_before = await _get_subject_row(db_session, user.id, "Mathematics")
    evidence_before = mastery_before.evidence_count
    mastery_score_before = mastery_before.mastery_score

    # Soft-delete every session for this user/subject (mirrors DELETE
    # /quiz/sessions/{id}) -- mastery is incrementally updated, not derived
    # by re-scanning session history, so this must NOT change what's
    # already stored.
    sessions = (await db_session.execute(
        select(QuizSession).where(QuizSession.user_id == user.id, QuizSession.subject == "Mathematics")
    )).scalars().all()
    for s in sessions:
        s.deleted_at = datetime.now(timezone.utc)
    await db_session.commit()

    mastery_after = await _get_subject_row(db_session, user.id, "Mathematics")
    assert mastery_after.evidence_count == evidence_before
    assert mastery_after.mastery_score == mastery_score_before


async def test_history_queries_read_through_soft_deleted_sessions(db_session):
    # The adaptive-difficulty engine deliberately reads through soft-deletes
    # (generation-critical side of the split) -- a soft-deleted session's
    # completion must still count toward evidence-based transition checks
    # in a LATER submission.
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await _submit(db_session, user.id, "Science", "Physics", "medium", question_count=10, correct_count=9)

    sessions = (await db_session.execute(
        select(QuizSession).where(QuizSession.user_id == user.id, QuizSession.subject == "Science")
    )).scalars().all()
    for s in sessions:
        s.deleted_at = datetime.now(timezone.utc)
    await db_session.commit()

    rows = await difficulty_service._fetch_recent_completions(db_session, user.id, "Science", limit=10)
    assert len(rows) == 1  # still visible despite the soft-delete


# ─────────────────────────────────────────────────────────────────────────────
# Lesson -> subject roll-up
# ─────────────────────────────────────────────────────────────────────────────

async def test_subject_mastery_rolls_up_from_lesson_mastery(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # A strong lesson and a weak lesson -- the subject score should land
    # somewhere between them, not just track the raw subject-wide accuracy.
    for _ in range(3):
        await _submit(db_session, user.id, "Mathematics", "Algebra", "medium", question_count=10, correct_count=9)
    for _ in range(3):
        await _submit(db_session, user.id, "Mathematics", "Geometry", "medium", question_count=10, correct_count=2)

    algebra = await _get_lesson_row(db_session, user.id, "Mathematics", "Algebra")
    geometry = await _get_lesson_row(db_session, user.id, "Mathematics", "Geometry")
    subject = await _get_subject_row(db_session, user.id, "Mathematics")

    assert algebra.mastery_score > geometry.mastery_score
    assert geometry.mastery_score < subject.mastery_score < algebra.mastery_score


async def test_subject_mastery_falls_back_to_direct_evidence_without_lesson_data(db_session):
    # "unknown" lessons never create a LessonMastery row (see
    # update_mastery_after_submission's skip), so the subject-level update
    # must gracefully fall back to its own direct quiz-evidence value.
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session = QuizSession(user_id=user.id, subject="Mathematics", lesson="unknown", difficulty="easy", question_count=5)
    db_session.add(session)
    await db_session.flush()
    graded_answers = [
        difficulty_service.GradedAnswer(lesson="unknown", difficulty="easy", correct=True, response_time=5.0, fingerprint=f"fp{i}")
        for i in range(5)
    ]
    db_session.add(QuizCompletion(
        session_id=session.id, ended_by="submitted", total_time=25.0, score=5.0, accuracy=100.0,
        correct_count=5, total_questions=5, lesson_accuracy_breakdown={"unknown": {"correct": 5, "total": 5, "accuracy": 100.0}},
    ))
    await db_session.commit()

    await difficulty_service.update_mastery_after_submission(
        db=db_session, user_id=user.id, subject="Mathematics",
        lesson_accuracy_breakdown={"unknown": {"correct": 5, "total": 5, "accuracy": 100.0}},
        session=session, ended_by="submitted", graded_answers=graded_answers,
    )
    await difficulty_service.update_subject_mastery_after_submission(
        db=db_session, user_id=user.id, subject="Mathematics", accuracy=100.0,
        session=session, ended_by="submitted", graded_answers=graded_answers,
    )

    lesson_rows = await difficulty_service._get_all_lesson_mastery_rows(db_session, user.id, "Mathematics")
    assert lesson_rows == []  # "unknown" never created a row
    subject = await _get_subject_row(db_session, user.id, "Mathematics")
    assert subject is not None and subject.mastery_score > 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Timeout completion evidence
# ─────────────────────────────────────────────────────────────────────────────

async def test_timeout_completion_uses_answered_ratio_not_zero(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Only 6 of 10 intended questions were answered before timing out.
    await _submit(
        db_session, user.id, "Mathematics", "Algebra", "easy",
        question_count=10, correct_count=5, ended_by="timeout", answered_count=6,
    )
    mastery = await _get_subject_row(db_session, user.id, "Mathematics")
    assert mastery is not None
    assert mastery.evidence_count == 6  # question-count-based on what was actually answered


# ─────────────────────────────────────────────────────────────────────────────
# Explicit-override bypass (generation-time reads, not mastery updates)
# ─────────────────────────────────────────────────────────────────────────────

async def test_get_current_difficulty_reads_lesson_level_row_independent_of_subject(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    db_session.add(LessonMastery(user_id=user.id, subject="Mathematics", lesson="Algebra", difficulty="hard"))
    db_session.add(SubjectMastery(user_id=user.id, subject="Mathematics", difficulty="easy"))
    await db_session.commit()

    lesson_level = await difficulty_service.get_current_difficulty(db_session, user.id, "Mathematics", "Algebra")
    subject_level = await difficulty_service.get_subject_difficulty(db_session, user.id, "Mathematics")
    assert lesson_level == "hard"
    assert subject_level == "easy"
