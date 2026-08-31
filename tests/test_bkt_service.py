"""
tests/test_bkt_service.py
──────────────────────────
Unit tests for app/services/bkt_service.py: the pure BKT math
(update_p_know/mastery_label) plus the DB-level get_or_create_state/
apply_observation/update_bkt_after_submission helpers that the live
submission path (quiz_service.submit_quiz) and the historical backfill
script (app/scripts/backfill_bkt_mastery.py) both build on.
"""
from sqlalchemy import select

from app.models.skill_bkt_state import SkillBKTState
from app.services import bkt_service
from app.services.difficulty_service import GradedAnswer
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

DEFAULT_KWARGS = dict(p_transit=0.20, p_slip=0.10, p_guess=0.20)


# ─────────────────────────────────────────────────────────────────────────────
# update_p_know (pure math)
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_answer_increases_p_know():
    result = bkt_service.update_p_know(0.30, True, **DEFAULT_KWARGS)
    assert result > 0.30


def test_incorrect_answer_decreases_p_know_but_not_to_zero():
    # Slip (p_slip=0.10) means "knows it but slipped" is always possible, so
    # one wrong answer must never crash the estimate to 0 — and p_transit
    # alone puts a floor under how low it can go.
    result = bkt_service.update_p_know(0.80, False, **DEFAULT_KWARGS)
    assert 0.0 < result < 0.80


def test_converges_toward_one_with_many_correct_observations():
    p_know = 0.05
    for _ in range(30):
        p_know = bkt_service.update_p_know(p_know, True, **DEFAULT_KWARGS)
    assert p_know > 0.95


def test_result_always_clamped_to_valid_probability_range():
    for correct in (True, False):
        for p_know in (0.0, 0.5, 1.0):
            result = bkt_service.update_p_know(p_know, correct, **DEFAULT_KWARGS)
            assert 0.0 <= result <= 1.0


def test_p_know_never_decreases_below_p_transit_floor():
    # Even starting from p_know=0 with a wrong answer, the learning
    # transition still applies — this is what stops the estimate flatlining
    # at 0 forever after one unlucky start.
    result = bkt_service.update_p_know(0.0, False, **DEFAULT_KWARGS)
    assert result == DEFAULT_KWARGS["p_transit"]


# ─────────────────────────────────────────────────────────────────────────────
# mastery_label
# ─────────────────────────────────────────────────────────────────────────────

def test_mastery_label_boundaries():
    kwargs = dict(mastered_threshold=0.80, learning_threshold=0.30)
    assert bkt_service.mastery_label(0.0, **kwargs) == "not_started"
    assert bkt_service.mastery_label(0.29, **kwargs) == "not_started"
    assert bkt_service.mastery_label(0.30, **kwargs) == "learning"
    assert bkt_service.mastery_label(0.79, **kwargs) == "learning"
    assert bkt_service.mastery_label(0.80, **kwargs) == "mastered"
    assert bkt_service.mastery_label(1.0, **kwargs) == "mastered"


# ─────────────────────────────────────────────────────────────────────────────
# get_or_create_state / apply_observation / update_bkt_after_submission
# ─────────────────────────────────────────────────────────────────────────────

async def _get_state(db, user_id: int, subject: str, lesson: str, grade: int | None) -> SkillBKTState | None:
    grade_filter = SkillBKTState.grade == grade if grade is not None else SkillBKTState.grade.is_(None)
    stmt = select(SkillBKTState).where(
        SkillBKTState.user_id == user_id, SkillBKTState.subject == subject,
        SkillBKTState.lesson == lesson, grade_filter,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def test_update_bkt_after_submission_creates_state_scoped_by_subject_lesson_grade(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    graded_answers = [
        GradedAnswer(subject="Mathematics", lesson="Percentages", difficulty="easy", correct=True, response_time=5.0, fingerprint=""),
        GradedAnswer(subject="Mathematics", lesson="Percentages", difficulty="easy", correct=False, response_time=5.0, fingerprint=""),
    ]
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=10, graded_answers=graded_answers,
    )
    await db_session.commit()

    state = await _get_state(db_session, user.id, "Mathematics", "Percentages", 10)
    assert state is not None
    assert state.opportunities == 2
    assert state.last_correct is False  # last answer folded in was wrong

    # Same-named lesson under a different grade must not collide.
    other_grade = await _get_state(db_session, user.id, "Mathematics", "Percentages", 11)
    assert other_grade is None


async def test_update_bkt_after_submission_same_skill_repeated_in_one_batch_does_not_duplicate(db_session):
    # Regression test: AsyncSessionLocal runs with autoflush=False in
    # production (app/core/database.py), so without update_bkt_after_
    # submission's internal StateCache, a second occurrence of the same
    # skill within one batch would create and try to insert a second row
    # for the same (user, subject, lesson, grade) before the first was ever
    # flushed, violating the unique constraint.
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    graded_answers = [
        GradedAnswer(subject="Science", lesson="Kinematics", difficulty="easy", correct=True, response_time=5.0, fingerprint="")
        for _ in range(5)
    ]
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=None, graded_answers=graded_answers,
    )
    await db_session.commit()  # would raise IntegrityError if a duplicate row was staged

    stmt = select(SkillBKTState).where(
        SkillBKTState.user_id == user.id, SkillBKTState.subject == "Science", SkillBKTState.lesson == "Kinematics",
    )
    rows = (await db_session.execute(stmt)).scalars().all()
    assert len(rows) == 1
    assert rows[0].opportunities == 5


async def test_update_bkt_after_submission_shuffle_mode_updates_each_skill_independently(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    graded_answers = [
        GradedAnswer(subject="Mathematics", lesson="Algebra", difficulty="easy", correct=True, response_time=5.0, fingerprint=""),
        GradedAnswer(subject="Science", lesson="Kinematics", difficulty="easy", correct=False, response_time=5.0, fingerprint=""),
        GradedAnswer(subject="Mathematics", lesson="Algebra", difficulty="easy", correct=True, response_time=5.0, fingerprint=""),
    ]
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=10, graded_answers=graded_answers,
    )
    await db_session.commit()

    math_state = await _get_state(db_session, user.id, "Mathematics", "Algebra", 10)
    science_state = await _get_state(db_session, user.id, "Science", "Kinematics", 10)
    assert math_state.opportunities == 2
    assert science_state.opportunities == 1
    assert science_state.last_correct is False


async def test_update_bkt_after_submission_skips_ungraded_answers(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    graded_answers = [
        GradedAnswer(subject="History", lesson="Ancient Rome", difficulty="easy", correct=None, response_time=5.0, fingerprint=""),
    ]
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=None, graded_answers=graded_answers,
    )
    await db_session.commit()

    state = await _get_state(db_session, user.id, "History", "Ancient Rome", None)
    assert state is None


# ─────────────────────────────────────────────────────────────────────────────
# get_subject_lesson_p_know_scores
# ─────────────────────────────────────────────────────────────────────────────

async def test_get_subject_lesson_p_know_scores_returns_mapping(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=10, graded_answers=[
            GradedAnswer(subject="Mathematics", lesson="Algebra", difficulty="easy", correct=True, response_time=5.0, fingerprint=""),
            GradedAnswer(subject="Mathematics", lesson="Geometry", difficulty="easy", correct=False, response_time=5.0, fingerprint=""),
        ],
    )
    await db_session.commit()

    scores = await bkt_service.get_subject_lesson_p_know_scores(db_session, user.id, "Mathematics")
    assert set(scores.keys()) == {"Algebra", "Geometry"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


async def test_get_subject_lesson_p_know_scores_empty_when_no_state(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    scores = await bkt_service.get_subject_lesson_p_know_scores(db_session, user.id, "Mathematics")
    assert scores == {}


async def test_get_subject_lesson_p_know_scores_ignores_other_subjects(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    await bkt_service.update_bkt_after_submission(
        db=db_session, user_id=user.id, grade=10, graded_answers=[
            GradedAnswer(subject="Science", lesson="Kinematics", difficulty="easy", correct=True, response_time=5.0, fingerprint=""),
        ],
    )
    await db_session.commit()

    scores = await bkt_service.get_subject_lesson_p_know_scores(db_session, user.id, "Mathematics")
    assert scores == {}
