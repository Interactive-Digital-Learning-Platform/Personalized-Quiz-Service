from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.skill_bkt_state import SkillBKTState
from app.services.difficulty_service import GradedAnswer


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def update_p_know(
    p_know: float,
    correct: bool,
    *,
    p_transit: float,
    p_slip: float,
    p_guess: float,
) -> float:
    # Standard 2-parameter-observation BKT update: Bayesian posterior from
    # the observation (using slip/guess), then apply the learning
    # transition. p_transit models that even a "doesn't know it yet"
    # student may have learned the skill during this very attempt.
    if correct:
        numerator = p_know * (1 - p_slip)
        denominator = numerator + (1 - p_know) * p_guess
    else:
        numerator = p_know * p_slip
        denominator = numerator + (1 - p_know) * (1 - p_guess)

    p_posterior = numerator / denominator if denominator > 0 else p_know
    p_know_new = p_posterior + (1 - p_posterior) * p_transit
    return _clamp01(p_know_new)


def mastery_label(
    p_know: float,
    *,
    mastered_threshold: float,
    learning_threshold: float,
) -> str:
    if p_know >= mastered_threshold:
        return "mastered"
    if p_know >= learning_threshold:
        return "learning"
    return "not_started"


# Keyed by (subject, lesson, grade). AsyncSessionLocal is created with
# autoflush=False (app/core/database.py), so a pending SkillBKTState insert
# is NOT visible to a subsequent SELECT within the same batch — without this
# cache, folding two answers for the same skill in one batch (e.g. every
# question in a single-lesson quiz, or replaying backfill history) would
# create and try to insert two rows for the same natural key, violating
# uq_skill_bkt_state_user_subject_lesson_grade. Callers that process more
# than one answer/attempt per batch must create one cache and reuse it for
# every get_or_create_state call in that batch.
StateCache = dict[tuple[str, str, int | None], SkillBKTState]


async def get_or_create_state(
    db: AsyncSession, *, user_id: int, subject: str, lesson: str, grade: int | None,
    cache: StateCache | None = None,
) -> SkillBKTState:
    cache_key = (subject, lesson, grade)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    grade_filter = SkillBKTState.grade == grade if grade is not None else SkillBKTState.grade.is_(None)
    stmt = select(SkillBKTState).where(
        SkillBKTState.user_id == user_id,
        SkillBKTState.subject == subject,
        SkillBKTState.lesson == lesson,
        grade_filter,
    )
    state = (await db.execute(stmt)).scalar_one_or_none()
    if state is None:
        # Column defaults (opportunities=0, last_correct=None) only apply at
        # flush time — set them explicitly since apply_observation() mutates
        # this object before it's ever flushed.
        state = SkillBKTState(
            user_id=user_id, subject=subject, lesson=lesson, grade=grade,
            p_know=settings.BKT_P_INIT, opportunities=0, last_correct=None,
        )
        db.add(state)
    if cache is not None:
        cache[cache_key] = state
    return state


def apply_observation(state: SkillBKTState, correct: bool) -> None:
    # Shared by the live submission path and the historical backfill script
    # so both fold an observation into a SkillBKTState row identically.
    state.p_know = update_p_know(
        state.p_know, correct,
        p_transit=settings.BKT_P_TRANSIT,
        p_slip=settings.BKT_P_SLIP,
        p_guess=settings.BKT_P_GUESS,
    )
    state.opportunities += 1
    state.last_correct = correct
    state.updated_at = datetime.now(UTC)


async def get_subject_lesson_p_know_scores(db: AsyncSession, user_id: int, subject: str) -> dict[str, float]:
    # Mirrors difficulty_service.get_subject_lesson_mastery_scores()'s shape
    # and (deliberately) its lack of grade-scoping, so the two can be
    # blended directly — see difficulty_mastery_engine.blend_lesson_weakness_scores.
    # A lesson can have more than one row if the user has attempted it under
    # different grades; same precedent as analytics/topic_service.py's
    # _latest_bkt_by_topic — keep the most recently updated row rather than
    # picking arbitrarily or trying to merge them.
    stmt = select(SkillBKTState).where(SkillBKTState.user_id == user_id, SkillBKTState.subject == subject)
    rows = (await db.execute(stmt)).scalars().all()

    scores: dict[str, float] = {}
    latest_seen: dict[str, datetime] = {}
    for row in rows:
        if row.lesson not in latest_seen or row.updated_at > latest_seen[row.lesson]:
            latest_seen[row.lesson] = row.updated_at
            scores[row.lesson] = row.p_know
    return scores


async def update_bkt_after_submission(
    db: AsyncSession,
    user_id: int,
    grade: int | None,
    graded_answers: list[GradedAnswer],
) -> None:
    # Answers must be processed in submission order — a shuffle-mode session
    # interleaves multiple skills across one answer list, and each skill's
    # own state needs to fold in only its own answers, in the order they
    # were actually attempted.
    cache: StateCache = {}
    for answer in graded_answers:
        if answer.correct is None:
            continue

        state = await get_or_create_state(
            db, user_id=user_id, subject=answer.subject, lesson=answer.lesson, grade=grade, cache=cache,
        )
        apply_observation(state, answer.correct)
