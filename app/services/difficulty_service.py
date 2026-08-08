"""
services/difficulty_service.py
────────────────────────────────
Adaptive difficulty: the user no longer picks "easy/medium/hard" — the system
tracks accuracy history and decides each quiz's difficulty automatically,
independently per subject (and per lesson, for the narrower override case).

Two tracks, same promote/demote rules, different granularity:
- `SubjectMastery` (user, subject) — drives the DEFAULT random-lesson quiz
  flow, since each question there gets an independently random lesson and
  there's no single (subject, lesson) pair to look up ahead of generation.
  Updated from the quiz's OVERALL accuracy. This is what makes e.g. a
  student's Maths quizzes gradually get harder while their Science quizzes
  stay easy — tracked completely independently per subject.
- `LessonMastery` (user, subject, lesson) — drives quizzes where the caller
  explicitly pins one specific lesson (manual override / focused practice).

Entry points:
- `get_current_difficulty()` / `get_subject_difficulty()` — read-only lookups
  called from `generate_quiz` before calling the AI. Default to "easy".
- `update_mastery_after_submission()` / `update_subject_mastery_after_submission()`
  — called from `submit_quiz` to adjust the stored difficulty after grading.

Promotion requires two consecutive strong sessions (>= 80% accuracy) so a
single lucky quiz doesn't bump difficulty prematurely. Demotion only requires
one weak session (<= 40% accuracy) so a struggling user isn't left stuck on
a difficulty that's discouraging them.
"""
import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lesson_mastery import LessonMastery
from app.models.subject_mastery import SubjectMastery

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DEFAULT_DIFFICULTY = "easy"

# Base 0-100 score per difficulty tier, used by mastery_service.py's
# difficulty_score component. Kept here (not duplicated in mastery_service.py
# or config) since it's fundamentally a difficulty-domain constant, same as
# DIFFICULTY_LEVELS above.
DIFFICULTY_BASE_SCORES = {"easy": 33.0, "medium": 66.0, "hard": 100.0}

PROMOTE_ACCURACY_THRESHOLD = 80.0
DEMOTE_ACCURACY_THRESHOLD = 40.0
PROMOTE_STREAK_REQUIRED = 2
DEMOTE_STREAK_REQUIRED = 1


class _MasteryRow(Protocol):
    """Structural type shared by LessonMastery and SubjectMastery rows."""
    difficulty: str
    last_accuracy: float
    consecutive_strong: int
    consecutive_weak: int


def _step_difficulty(current: str, direction: int) -> str:
    """direction: +1 to promote, -1 to demote. Clamped to the level bounds."""
    idx = DIFFICULTY_LEVELS.index(current)
    new_idx = max(0, min(len(DIFFICULTY_LEVELS) - 1, idx + direction))
    return DIFFICULTY_LEVELS[new_idx]


def _apply_promote_demote(mastery: _MasteryRow, accuracy: float, *, log_label: str) -> None:
    """
    Shared promote/demote streak logic, mutating `mastery` in place. Works for
    both LessonMastery and SubjectMastery rows since they share the same
    difficulty/streak-counter field names.
    """
    if accuracy >= PROMOTE_ACCURACY_THRESHOLD:
        mastery.consecutive_strong += 1
        mastery.consecutive_weak = 0
        if mastery.consecutive_strong >= PROMOTE_STREAK_REQUIRED:
            new_difficulty = _step_difficulty(mastery.difficulty, +1)
            if new_difficulty != mastery.difficulty:
                logger.info("Promoting difficulty for %s: %s -> %s", log_label, mastery.difficulty, new_difficulty)
            mastery.difficulty = new_difficulty
            mastery.consecutive_strong = 0
    elif accuracy <= DEMOTE_ACCURACY_THRESHOLD:
        mastery.consecutive_weak += 1
        mastery.consecutive_strong = 0
        if mastery.consecutive_weak >= DEMOTE_STREAK_REQUIRED:
            new_difficulty = _step_difficulty(mastery.difficulty, -1)
            if new_difficulty != mastery.difficulty:
                logger.info("Demoting difficulty for %s: %s -> %s", log_label, mastery.difficulty, new_difficulty)
            mastery.difficulty = new_difficulty
            mastery.consecutive_weak = 0
    else:
        # Stable performance in the middle band — no streak progress either way.
        mastery.consecutive_strong = 0
        mastery.consecutive_weak = 0

    mastery.last_accuracy = round(accuracy, 2)


# ─────────────────────────────────────────────────────────────────────────────
# LESSON-LEVEL MASTERY (explicit lesson override / focused practice)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_lesson_mastery_row(
    db: AsyncSession, user_id: int, subject: str, lesson: str
) -> LessonMastery | None:
    stmt = select(LessonMastery).where(
        LessonMastery.user_id == user_id,
        LessonMastery.subject == subject,
        LessonMastery.lesson == lesson,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_current_difficulty(
    db: AsyncSession, user_id: int, subject: str, lesson: str
) -> str:
    """
    Read-only: returns the difficulty the next quiz for this exact
    (subject, lesson) should use. Does NOT create a row — new lessons simply
    default to "easy" until the first submission establishes a mastery record.
    """
    mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def update_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    lesson_accuracy_breakdown: dict[str, dict],
) -> None:
    """
    Given the per-lesson accuracy breakdown from a just-graded quiz submission,
    update (or create) the LessonMastery row for each lesson and adjust its
    stored difficulty according to the promote/demote streak rules.

    `lesson_accuracy_breakdown` shape (from quiz_service.submit_quiz):
        {"Algebra": {"correct": 4, "total": 5, "accuracy": 80.0}, ...}
    """
    for lesson, stats in lesson_accuracy_breakdown.items():
        if lesson == "unknown":
            continue

        accuracy = float(stats.get("accuracy", 0.0))
        mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson)
        if mastery is None:
            # Set counters explicitly rather than relying on the column
            # `default=` — that only applies at INSERT time, but we need to
            # mutate these counters immediately below, before the row is flushed.
            mastery = LessonMastery(
                user_id=user_id,
                subject=subject,
                lesson=lesson,
                difficulty=DEFAULT_DIFFICULTY,
                last_accuracy=0.0,
                consecutive_strong=0,
                consecutive_weak=0,
            )
            db.add(mastery)

        _apply_promote_demote(mastery, accuracy, log_label=f"user={user_id}, subject={subject}, lesson={lesson}")

    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT-LEVEL MASTERY (default random-lesson quiz flow)
# ─────────────────────────────────────────────────────────────────────────────

async def _get_subject_mastery_row(
    db: AsyncSession, user_id: int, subject: str
) -> SubjectMastery | None:
    stmt = select(SubjectMastery).where(
        SubjectMastery.user_id == user_id,
        SubjectMastery.subject == subject,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_subject_difficulty(
    db: AsyncSession, user_id: int, subject: str
) -> str:
    """
    Read-only: returns the difficulty the next random-lesson quiz for this
    subject should use, based on the student's overall accuracy history in
    that subject. Does NOT create a row — a subject with no history yet
    defaults to "easy".
    """
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def update_subject_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    accuracy: float,
) -> None:
    """
    Update (or create) the SubjectMastery row for this (user, subject) using
    the OVERALL accuracy of the just-graded quiz (spanning however many
    random lessons it touched), and adjust the stored difficulty according to
    the same promote/demote streak rules used for lessons.
    """
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    if mastery is None:
        mastery = SubjectMastery(
            user_id=user_id,
            subject=subject,
            difficulty=DEFAULT_DIFFICULTY,
            last_accuracy=0.0,
            consecutive_strong=0,
            consecutive_weak=0,
        )
        db.add(mastery)

    _apply_promote_demote(mastery, accuracy, log_label=f"user={user_id}, subject={subject}")

    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# READ-ONLY PROJECTION FOR ANALYTICS (GET /analytics/me)
# ─────────────────────────────────────────────────────────────────────────────

def describe_subject_mastery(mastery: SubjectMastery | None) -> dict:
    """
    Read-only, pure projection of a SubjectMastery row (or safe defaults if
    the subject has no mastery row yet) into the fields
    GET /analytics/me exposes alongside subjects[].difficulty_performance.

    This is the ONLY place that turns raw mastery state into a human-facing
    description of "how close to promotion/demotion" a subject is — it reuses
    the exact same constants and _step_difficulty() helper that
    update_subject_mastery_after_submission() uses to actually move the
    difficulty, so analytics can never drift from the real adaptive engine or
    duplicate its thresholds. This function NEVER mutates `mastery` or
    touches the DB — promotion/demotion only ever happens via
    update_subject_mastery_after_submission(), called from quiz submission.

    consecutive_strong/consecutive_weak are read as-is from the stored row:
    by construction, a streak count reaching its required threshold triggers
    a promotion/demotion AND a reset to 0 in the same update, so in practice
    consecutive_strong is only ever observed as 0..(PROMOTE_STREAK_REQUIRED-1)
    and consecutive_weak as 0..(DEMOTE_STREAK_REQUIRED-1) — this function
    doesn't assume that, it just reads whatever is actually stored, so it
    stays correct even if those thresholds are ever reconfigured.
    """
    current_difficulty = mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY
    consecutive_strong = mastery.consecutive_strong if mastery is not None else 0
    consecutive_weak = mastery.consecutive_weak if mastery is not None else 0

    next_difficulty = _step_difficulty(current_difficulty, +1)
    at_max_difficulty = next_difficulty == current_difficulty

    promotion_progress_percentage = (
        0.0 if at_max_difficulty
        else round(min(consecutive_strong, PROMOTE_STREAK_REQUIRED) / PROMOTE_STREAK_REQUIRED * 100.0, 2)
    )

    if at_max_difficulty:
        message = f"Already at the highest difficulty ({current_difficulty})."
    elif consecutive_strong > 0:
        remaining = max(PROMOTE_STREAK_REQUIRED - consecutive_strong, 0)
        if remaining == 0:
            message = f"Ready to advance to {next_difficulty} difficulty."
        else:
            quiz_word = "quiz" if remaining == 1 else "quizzes"
            message = f"{remaining} more strong {quiz_word} needed to reach {next_difficulty} difficulty."
    elif consecutive_weak > 0:
        remaining = max(DEMOTE_STREAK_REQUIRED - consecutive_weak, 0)
        quiz_word = "quiz" if remaining == 1 else "quizzes"
        message = f"{remaining} more weak {quiz_word} until {current_difficulty} difficulty drops."
    else:
        message = f"At {current_difficulty} difficulty. Keep practicing to progress."

    return {
        "current_difficulty": current_difficulty,
        "consecutive_strong_quizzes": consecutive_strong,
        "consecutive_weak_quizzes": consecutive_weak,
        "promotion_threshold": PROMOTE_ACCURACY_THRESHOLD,
        "demotion_threshold": DEMOTE_ACCURACY_THRESHOLD,
        "quizzes_required_for_promotion": PROMOTE_STREAK_REQUIRED,
        "promotion_progress_percentage": promotion_progress_percentage,
        "next_difficulty": next_difficulty,
        "difficulty_status_message": message,
    }
