"""Optional, explicit backfill for Bayesian Knowledge Tracing state
(SkillBKTState) — replays existing per-question QuestionAttempt history
(chronologically, oldest first) through bkt_service.update_p_know() so
users with quiz history from before this feature shipped get their
p_know populated from real history immediately, instead of starting back
at the neutral prior and rebuilding evidence from scratch going forward.

Deliberately a separate script from backfill_adaptive_mastery.py rather
than an extension of it: that script synthesizes graded_answers with
correct=True for every question (fine for CEWM, which derives its actual
accuracy signal from lesson_accuracy_breakdown instead) — reusing that
approach here would replay backfilled history into BKT as if every past
question was answered correctly, silently inflating every backfilled
skill's p_know. BKT needs the real per-question QuestionAttempt.correct
values, so this script reads those directly instead.

NOT run automatically anywhere (no import of this module from app
startup) — must be invoked explicitly:

    uv run python -m app.scripts.backfill_bkt_mastery [--dry-run] [--force]

Idempotent by default: a (user, subject, lesson, grade) skill is only
replayed if its SkillBKTState row doesn't exist yet or still has
opportunities == 0 (i.e. never touched by a real submission or a
previous backfill run) — running it twice does not double-count history.
Pass --force to replay and OVERWRITE skills that already have evidence,
e.g. after deliberately tuning the BKT_* parameters and wanting existing
users' history rescored against the new values.
"""
import argparse
import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.skill_bkt_state import SkillBKTState
from app.models.user import User
from app.services import bkt_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _skill_sort_key(skill: tuple[str, str, int | None]) -> tuple[str, str, int]:
    subject, lesson, grade = skill
    return (subject, lesson, -1 if grade is None else grade)


async def _fetch_attempts_chronological(
    db: AsyncSession, user_id: int,
) -> list[tuple[QuestionAttempt, Question, int | None]]:
    # Reads through soft-deletes on purpose, same as backfill_adaptive_mastery.py
    # and the rest of the mastery-tracking pipeline. Excludes retakes: the
    # live submission path never updates BKT for a retake (see quiz_service.
    # submit_quiz), since the user has already seen the correct answers.
    # Ordered by session creation, then attempt id (== submission order
    # within a session, per QuestionAttempt's insertion order) so
    # shuffle-mode sessions replay each skill's own answers in the order
    # they were actually attempted.
    stmt = (
        select(QuestionAttempt, Question, QuizSession.grade)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(QuizSession.user_id == user_id, QuizSession.is_retake.is_(False))
        .order_by(QuizSession.created_at.asc(), QuestionAttempt.id.asc())
    )
    rows = (await db.execute(stmt)).all()
    return [(r[0], r[1], r[2]) for r in rows]


async def _skills_needing_backfill(db: AsyncSession, user_id: int, force: bool) -> set[tuple[str, str, int | None]]:
    stmt = (
        select(Question.subject, Question.lesson, QuizSession.grade)
        .join(QuestionAttempt, QuestionAttempt.question_id == Question.id)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(QuizSession.user_id == user_id, QuizSession.is_retake.is_(False))
        .distinct()
    )
    all_skills = {(subj, lesson, grade) for subj, lesson, grade in (await db.execute(stmt)).all()}

    if force:
        return all_skills

    state_stmt = select(SkillBKTState).where(SkillBKTState.user_id == user_id)
    existing = {(s.subject, s.lesson, s.grade): s for s in (await db.execute(state_stmt)).scalars().all()}

    return {s for s in all_skills if s not in existing or existing[s].opportunities == 0}


async def _reset_existing_rows(db: AsyncSession, user_id: int, skills: set[tuple[str, str, int | None]]) -> None:
    # --force replays from a clean slate for the affected skills rather than
    # blending backfilled evidence into whatever's already there — same
    # "deterministic, not additive" rationale as backfill_adaptive_mastery.py.
    stmt = select(SkillBKTState).where(SkillBKTState.user_id == user_id)
    for row in (await db.execute(stmt)).scalars().all():
        if (row.subject, row.lesson, row.grade) in skills:
            await db.delete(row)
    await db.commit()


async def backfill_user(db: AsyncSession, user: User, *, force: bool, dry_run: bool) -> int:
    skills_to_backfill = await _skills_needing_backfill(db, user.id, force)
    if not skills_to_backfill:
        return 0

    if dry_run:
        logger.info("[dry-run] user=%s would backfill skills=%s", user.clerk_id, sorted(skills_to_backfill, key=_skill_sort_key))
        return len(skills_to_backfill)

    if force:
        await _reset_existing_rows(db, user.id, skills_to_backfill)

    attempts = await _fetch_attempts_chronological(db, user.id)
    replayed = 0
    # One cache for the whole replay (not per-session) — the same skill
    # recurs across many historical sessions, and without caching the
    # unflushed row created for its first occurrence, a later occurrence's
    # lookup wouldn't see it yet (AsyncSessionLocal uses autoflush=False)
    # and would try to insert a second row for the same natural key.
    state_cache: bkt_service.StateCache = {}
    for attempt, question, grade in attempts:
        skill = (question.subject, question.lesson, grade)
        if skill not in skills_to_backfill or attempt.correct is None:
            continue

        state = await bkt_service.get_or_create_state(
            db, user_id=user.id, subject=question.subject, lesson=question.lesson, grade=grade,
            cache=state_cache,
        )
        bkt_service.apply_observation(state, attempt.correct)
        replayed += 1

    logger.info("user=%s backfilled skills=%s from %d historical attempts", user.clerk_id, sorted(skills_to_backfill, key=_skill_sort_key), replayed)
    return len(skills_to_backfill)


async def run(*, force: bool, dry_run: bool) -> None:
    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        total_skills = 0
        for user in users:
            total_skills += await backfill_user(db, user, force=force, dry_run=dry_run)
        if not dry_run:
            await db.commit()
        logger.info("Backfill complete: %d user-skill pairs processed across %d users.", total_skills, len(users))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log what would be backfilled without writing anything.")
    parser.add_argument(
        "--force", action="store_true",
        help="Also replay skills that already have evidence, resetting them first (e.g. after tuning BKT_* parameters).",
    )
    args = parser.parse_args()
    asyncio.run(run(force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
