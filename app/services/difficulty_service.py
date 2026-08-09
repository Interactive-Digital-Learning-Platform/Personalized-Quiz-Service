import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lesson_mastery import LessonMastery
from app.models.subject_mastery import SubjectMastery

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DEFAULT_DIFFICULTY = "easy"
DIFFICULTY_BASE_SCORES = {"easy": 33.0, "medium": 66.0, "hard": 100.0}

PROMOTE_ACCURACY_THRESHOLD = 80.0
DEMOTE_ACCURACY_THRESHOLD = 40.0
PROMOTE_STREAK_REQUIRED = 2
DEMOTE_STREAK_REQUIRED = 1


class _MasteryRow(Protocol):
    difficulty: str
    last_accuracy: float
    consecutive_strong: int
    consecutive_weak: int


def _step_difficulty(current: str, direction: int) -> str:
    idx = DIFFICULTY_LEVELS.index(current)
    new_idx = max(0, min(len(DIFFICULTY_LEVELS) - 1, idx + direction))
    return DIFFICULTY_LEVELS[new_idx]


def _apply_promote_demote(mastery: _MasteryRow, accuracy: float, *, log_label: str) -> None:
    # Promotion needs 2 strong quizzes in a row so one lucky result doesn't
    # bump the difficulty; demotion only needs 1 weak quiz so a struggling
    # student isn't left stuck somewhere discouraging. Shared between
    # LessonMastery and SubjectMastery since both have the same fields.
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
        mastery.consecutive_strong = 0
        mastery.consecutive_weak = 0

    mastery.last_accuracy = round(accuracy, 2)


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
    mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def update_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    lesson_accuracy_breakdown: dict[str, dict],
) -> None:
    for lesson, stats in lesson_accuracy_breakdown.items():
        if lesson == "unknown":
            continue

        accuracy = float(stats.get("accuracy", 0.0))
        mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson)
        if mastery is None:
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
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def update_subject_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    accuracy: float,
) -> None:
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


def describe_subject_mastery(mastery: SubjectMastery | None) -> dict:
    # Read-only projection of a SubjectMastery row into the "how close to
    # promotion/demotion" fields GET /analytics/me shows — reuses the exact
    # same constants and _step_difficulty() that actually drive promotion,
    # so this can never drift from the real engine. Never writes to the DB;
    # that only happens in update_subject_mastery_after_submission().
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
