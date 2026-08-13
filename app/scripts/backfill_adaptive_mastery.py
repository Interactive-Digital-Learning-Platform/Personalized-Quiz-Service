"""Optional, explicit backfill for the Continuous Evidence-Weighted Mastery
System — replays existing quiz history through the real
difficulty_service.update_mastery_after_submission() /
update_subject_mastery_after_submission() functions (chronologically,
oldest first) so users with quiz history from before this feature shipped
get their evidence_count/mastery_score/confidence/trend populated from that
history immediately, instead of starting back at the neutral defaults and
rebuilding evidence from scratch going forward.

NOT run automatically anywhere (no import of this module from app startup) —
must be invoked explicitly:

    uv run python -m app.scripts.backfill_adaptive_mastery [--dry-run] [--force]

Idempotent by default: a (user, subject) or (user, subject, lesson) pair is
only replayed if its mastery row doesn't exist yet or still has
evidence_count == 0 (i.e. never touched by a real submission or a previous
backfill run) — running it twice does not double-count history. Pass
--force to replay and OVERWRITE rows that already have evidence, e.g. after
deliberately tuning the ADAPTIVE_MASTERY_* weights and wanting existing
users' history rescored against the new weights.
"""
import argparse
import asyncio
import logging
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.lesson_mastery import LessonMastery
from app.models.quiz_session import QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.models.subject_mastery import SubjectMastery
from app.models.user import User
from app.services import difficulty_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _fetch_all_completions_chronological(db: AsyncSession, user_id: int) -> list[tuple[QuizSession, QuizCompletion]]:
    # Reads through soft-deletes on purpose, same as the rest of the
    # adaptive-difficulty engine (this IS the generation-critical side of
    # that split) -- a session being hidden from the user's history list
    # doesn't mean it stopped being real evidence of what they knew.
    stmt = (
        select(QuizSession, QuizCompletion)
        .join(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(QuizSession.user_id == user_id)
        .order_by(QuizSession.created_at.asc())
    )
    rows = (await db.execute(stmt)).all()
    return [(r[0], r[1]) for r in rows]


async def _subjects_needing_backfill(db: AsyncSession, user_id: int, force: bool) -> set[str]:
    if force:
        stmt = select(QuizSession.subject).where(QuizSession.user_id == user_id).distinct()
        return set((await db.execute(stmt)).scalars().all())

    stmt = select(QuizSession.subject).where(QuizSession.user_id == user_id).distinct()
    all_subjects = set((await db.execute(stmt)).scalars().all())

    mastery_stmt = select(SubjectMastery).where(SubjectMastery.user_id == user_id)
    existing = {m.subject: m for m in (await db.execute(mastery_stmt)).scalars().all()}

    return {s for s in all_subjects if s not in existing or existing[s].evidence_count == 0}


async def _reset_existing_rows(db: AsyncSession, user_id: int, subjects: set[str]) -> None:
    # --force replays from a clean slate for the affected subjects/lessons
    # rather than blending backfilled evidence into whatever's already
    # there, so the result is exactly "what the current weights would have
    # produced from this user's full history" -- deterministic, not additive.
    subj_stmt = select(SubjectMastery).where(SubjectMastery.user_id == user_id, SubjectMastery.subject.in_(subjects))
    for row in (await db.execute(subj_stmt)).scalars().all():
        await db.delete(row)
    lesson_stmt = select(LessonMastery).where(LessonMastery.user_id == user_id, LessonMastery.subject.in_(subjects))
    for row in (await db.execute(lesson_stmt)).scalars().all():
        await db.delete(row)
    await db.commit()


async def backfill_user(db: AsyncSession, user: User, *, force: bool, dry_run: bool) -> int:
    subjects_to_backfill = await _subjects_needing_backfill(db, user.id, force)
    if not subjects_to_backfill:
        return 0

    if dry_run:
        logger.info("[dry-run] user=%s would backfill subjects=%s", user.clerk_id, sorted(subjects_to_backfill))
        return len(subjects_to_backfill)

    if force:
        await _reset_existing_rows(db, user.id, subjects_to_backfill)

    completions = await _fetch_all_completions_chronological(db, user.id)
    replayed = 0
    for session, completion in completions:
        if session.subject not in subjects_to_backfill:
            continue

        graded_answers = [
            difficulty_service.GradedAnswer(
                lesson=session.lesson, difficulty=session.difficulty, correct=True,
                response_time=completion.total_time / max(completion.total_questions, 1),
                fingerprint="",
            )
            for _ in range(completion.total_questions)
        ]
        lesson_accuracy_breakdown = completion.lesson_accuracy_breakdown or {
            session.lesson: {
                "correct": completion.correct_count, "total": completion.total_questions,
                "accuracy": completion.accuracy,
            }
        }

        await difficulty_service.update_mastery_after_submission(
            db=db, user_id=user.id, subject=session.subject,
            lesson_accuracy_breakdown=lesson_accuracy_breakdown,
            session=session, ended_by=completion.ended_by, graded_answers=graded_answers,
        )
        await difficulty_service.update_subject_mastery_after_submission(
            db=db, user_id=user.id, subject=session.subject, accuracy=completion.accuracy,
            session=session, ended_by=completion.ended_by, graded_answers=graded_answers,
        )
        replayed += 1

    logger.info("user=%s backfilled subjects=%s from %d historical completions", user.clerk_id, sorted(subjects_to_backfill), replayed)
    return len(subjects_to_backfill)


async def run(*, force: bool, dry_run: bool) -> None:
    async with AsyncSessionLocal() as db:
        users = (await db.execute(select(User))).scalars().all()
        total_subjects = 0
        for user in users:
            total_subjects += await backfill_user(db, user, force=force, dry_run=dry_run)
        logger.info("Backfill complete: %d user-subject pairs processed across %d users.", total_subjects, len(users))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log what would be backfilled without writing anything.")
    parser.add_argument(
        "--force", action="store_true",
        help="Also replay subjects/lessons that already have evidence, resetting them first (e.g. after tuning weights).",
    )
    args = parser.parse_args()
    asyncio.run(run(force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
