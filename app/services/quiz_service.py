import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.user import User
from app.schemas.quiz import (
    AnswerItem,
    GenerateQuizRequest,
    SaveProgressRequest,
    SubmitQuizRequest,
)
from app.services import (
    bkt_service,
    curriculum_service,
    difficulty_service,
    provenance_columns,
    rag_service,
    telemetry_service,
)
from app.services import difficulty_mastery_engine as mastery_engine
from app.services.groq_service import (
    generate_questions,
    generate_shuffle_quiz_questions,
    references_missing_visual,
)

logger = logging.getLogger(__name__)


async def _get_owned_session(db: AsyncSession, user_id: int, session_id: int) -> QuizSession:
    session_stmt = select(QuizSession).where(
        QuizSession.id == session_id,
        QuizSession.user_id == user_id,
        QuizSession.deleted_at.is_(None),
    )
    result = await db.execute(session_stmt)
    session: QuizSession | None = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session not found or does not belong to this user.",
        )

    return session


async def create_retake_session(db: AsyncSession, clerk_id: str, session_id: int) -> QuizSession:
    # "Restart Quiz" used to resubmit into the same session_id, which submit_quiz() rejects with 409 once completed -- see QuizSession.is_retake's docstring for why a fresh row is what analytics needs.
    user = await get_or_create_user(db, clerk_id)
    source = await _get_owned_session(db=db, user_id=user.id, session_id=session_id)

    # Chain to the original attempt, not the immediate parent, so retaking a retake doesn't build an ever-deepening chain.
    root_session_id = source.retake_of_session_id or source.id

    retake = QuizSession(
        user_id=user.id,
        subject=source.subject,
        lesson=source.lesson,
        grade=source.grade,
        difficulty=source.difficulty,
        question_count=source.question_count,
        questions_snapshot=source.questions_snapshot,
        is_retake=True,
        retake_of_session_id=root_session_id,
    )
    db.add(retake)
    await db.commit()
    await db.refresh(retake)

    logger.info(
        "Created retake QuizSession id=%d (of session=%d) for user=%d",
        retake.id, root_session_id, user.id,
    )
    return retake


def _serialize_question(question: Question) -> dict:
    return {
        "id": question.id,
        "question": question.question,
        "options": question.options,
        "subject": question.subject,
        "lesson": question.lesson,
        "difficulty": question.difficulty,
        "correct_answer": question.correct_answer,
    }


async def _get_recent_lessons(
    db: AsyncSession, user_id: int, subject: str, limit: int = 6
) -> list[str]:
    # Lessons this user was recently quizzed on, most recent first — fed to generate_questions() as "please avoid these" for topic variety.
    stmt = (
        select(QuizSession.lesson)
        .where(QuizSession.user_id == user_id, QuizSession.subject == subject)
        .order_by(QuizSession.created_at.desc())
        .limit(limit * 3)
    )
    rows = (await db.execute(stmt)).scalars().all()

    seen: list[str] = []
    for lesson in rows:
        if lesson and lesson != "Mixed" and lesson not in seen:
            seen.append(lesson)
        if len(seen) >= limit:
            break
    return seen


def _snippet_count(snippets: list[str] | None, by_lesson: dict[str, list[str]] | None) -> int:
    return len(snippets or []) + sum(len(v) for v in (by_lesson or {}).values())


async def _get_existing_subject_questions(db: AsyncSession, subject: str, limit: int = 40) -> list[str]:
    # A random sample of this subject's already-cached question bank, fed to the shuffle AI call as an "avoid these" list -- random rather than newest-first so repeated shuffle requests see a rotating slice.
    stmt = (
        select(Question.question)
        .where(Question.subject == subject)
        .order_by(func.random())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return list(rows)


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
    "or", "which", "what", "how", "why", "when", "where", "that", "this",
    "it", "its", "be", "been", "by", "for", "with", "as", "at", "from",
    "on", "not", "does", "do", "did", "will", "would", "can", "could",
    "following", "given", "most", "one", "two", "three",
})


def _significant_words(text: str) -> frozenset[str]:
    return frozenset(
        w for w in text.lower().split() if w.isalpha() and w not in _STOP_WORDS and len(w) > 2
    )


def _jaccard_similarity(a: str, b: str) -> float:
    wa, wb = _significant_words(a), _significant_words(b)
    union = wa | wb
    if not union:
        return 0.0
    return len(wa & wb) / len(union)


def _is_near_duplicate(candidate: str, existing: list[str], threshold: float = 0.55) -> bool:
    # Two questions count as "the same" if they share 55%+ of their meaningful words -- catches Groq rephrasing a question it was already asked not to repeat.
    for ex in existing:
        if _jaccard_similarity(candidate, ex) >= threshold:
            return True
    return False


def _allocate_shuffle_subject_counts(subjects: list[str], total: int) -> dict[str, int]:
    # Spreads `total` questions as evenly as possible across `subjects` for Shuffle Mode; unlike mastery_engine.allocate_question_counts()'s deterministic largest-remainder method, this randomizes which subjects get leftovers/get picked at all, so a shuffle quiz doesn't always favor the same subjects.
    if total <= 0 or not subjects:
        return {}

    if len(subjects) > total:
        # More subjects than questions requested — only `total` subjects get to participate, one question each, chosen at random.
        chosen = random.sample(subjects, total)
        return {subject: 1 for subject in chosen}

    base = total // len(subjects)
    remainder = total % len(subjects)
    counts = {subject: base for subject in subjects}
    if remainder > 0:
        bonus_subjects = random.sample(subjects, remainder)
        for subject in bonus_subjects:
            counts[subject] += 1
    return {subject: count for subject, count in counts.items() if count > 0}


# Buckets currently being topped up in the background, keyed by (subject, difficulty, grade) -- an in-process debounce so concurrent requests don't each spawn their own top-up task; no cross-process lock needed since this service is single-worker.
_replenishing_buckets: set[tuple[str, str, int]] = set()


async def _fetch_pool_questions(
    db: AsyncSession,
    *,
    user_id: int,
    subject: str,
    difficulty: str,
    grade: int,
    limit: int,
    exclude_ids: set[int],
    exclude_texts: list[str],
) -> list[Question]:
    # Primary read path for quiz generation: pre-generated questions for this (subject, difficulty, grade) pool that `user_id` hasn't already been asked, so a request only calls Groq for the shortfall; over-fetches 3x `limit` for near-duplicate filtering, scoped to `grade`'s curriculum lesson list where available, and triggers a debounced background top-up when the bucket runs low.
    if limit <= 0:
        return []

    already_seen_by_user = (
        select(QuestionAttempt.question_id)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(QuizSession.user_id == user_id)
    )
    filters = [
        Question.subject == subject,
        Question.difficulty == difficulty,
        Question.grade == grade,
        Question.id.notin_(already_seen_by_user),
    ]
    lesson_choices = (
        curriculum_service.lessons_for(grade, subject)
        if curriculum_service.has_curriculum(grade)
        else None
    )
    if lesson_choices:
        filters.append(Question.lesson.in_(lesson_choices))
    if exclude_ids:
        filters.append(Question.id.notin_(exclude_ids))

    stmt = select(Question).where(*filters).order_by(func.random()).limit(limit * 3)
    candidates = list((await db.execute(stmt)).scalars().all())

    accepted: list[Question] = []
    for candidate in candidates:
        if len(accepted) >= limit:
            break
        # Rows generated before the text-only rule may point at a graph/table that can't be shown.
        if references_missing_visual(candidate.question):
            continue
        if _is_near_duplicate(candidate.question, exclude_texts):
            continue
        accepted.append(candidate)
        exclude_texts.append(candidate.question)

    bucket_size_stmt = (
        select(func.count())
        .select_from(Question)
        .where(Question.subject == subject, Question.difficulty == difficulty, Question.grade == grade)
    )
    bucket_size = (await db.execute(bucket_size_stmt)).scalar() or 0
    if bucket_size < settings.POOL_MIN_SIZE:
        _kickoff_pool_replenish(subject, difficulty, grade)

    return accepted


def _kickoff_pool_replenish(subject: str, difficulty: str, grade: int) -> None:
    # A thin, easily-mockable wrapper around asyncio.create_task -- tests patch this directly so they never spawn a real background Groq call.
    key = (subject, difficulty, grade)
    if key in _replenishing_buckets:
        return
    _replenishing_buckets.add(key)
    asyncio.create_task(_replenish_pool_task(subject, difficulty, grade))


async def _replenish_pool_task(subject: str, difficulty: str, grade: int) -> None:
    # Runs after the triggering request has already returned its response -- opens its own DB session since the request-scoped one is closed by then; fully decoupled from any user-facing request/response.
    key = (subject, difficulty, grade)
    try:
        async with AsyncSessionLocal() as session:
            count_stmt = (
                select(func.count())
                .select_from(Question)
                .where(Question.subject == subject, Question.difficulty == difficulty, Question.grade == grade)
            )
            current = (await session.execute(count_stmt)).scalar() or 0
            needed = settings.POOL_TARGET_SIZE - current
            if needed <= 0:
                return

            batch_size = min(needed, settings.POOL_TOPUP_MAX_BATCH)
            # Curriculum-backed grades (10/11) must only ever top up the pool with questions pinned to a real syllabus lesson, to stop off-syllabus lessons leaking into "Auto (AI picks)" quizzes.
            lesson_choices = (
                curriculum_service.lessons_for(grade, subject)
                if curriculum_service.has_curriculum(grade)
                else None
            )
            try:
                batch = await generate_questions(
                    grade=grade,
                    subject=subject,
                    difficulty=difficulty,
                    question_count=batch_size,
                    lesson_choices=lesson_choices,
                )
            except HTTPException as exc:
                logger.warning(
                    "Pool replenish failed subject=%s difficulty=%s: %s", subject, difficulty, exc.detail,
                )
                return

            for q_data in batch:
                session.add(Question(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    subject=subject,
                    lesson=q_data["lesson"],
                    difficulty=difficulty,
                    grade=grade,
                ))
            await session.commit()
            logger.info(
                "Pool replenished subject=%s difficulty=%s added=%d new_total=%d",
                subject, difficulty, len(batch), current + len(batch),
            )
    except Exception:
        logger.exception("Unexpected error replenishing pool subject=%s difficulty=%s", subject, difficulty)
    finally:
        _replenishing_buckets.discard(key)


async def _fetch_shuffle_fallback_questions(
    db: AsyncSession,
    *,
    subjects: list[str],
    grade: int,
    difficulties: list[str] | None,
    limit: int,
    exclude_ids: set[int],
    exclude_texts: list[str],
) -> list[Question]:
    # Blind DB-cache fallback -- the final safety net once both the pool read and a live Groq call have already been tried and it's still short; scoped to `grade`'s curriculum lesson list like _fetch_pool_questions, but deliberately doesn't filter by per-user attempt history since as a last resort a repeat beats returning fewer than requested.
    if limit <= 0 or not subjects:
        return []

    filters = [Question.subject.in_(subjects), Question.grade == grade]
    if difficulties:
        filters.append(Question.difficulty.in_(difficulties))

    if curriculum_service.has_curriculum(grade):
        lesson_filters = []
        for subject in subjects:
            lesson_choices = curriculum_service.lessons_for(grade, subject)
            if lesson_choices:
                lesson_filters.append(
                    (Question.subject == subject) & Question.lesson.in_(lesson_choices)
                )
        if lesson_filters:
            filters.append(or_(*lesson_filters))

    if exclude_ids:
        filters.append(Question.id.notin_(exclude_ids))

    stmt = select(Question).where(*filters).order_by(func.random()).limit(limit * 3)
    candidates = list((await db.execute(stmt)).scalars().all())

    accepted: list[Question] = []
    for candidate in candidates:
        if len(accepted) >= limit:
            break
        # Rows generated before the text-only rule may point at a graph/table that can't be shown.
        if references_missing_visual(candidate.question):
            continue
        if _is_near_duplicate(candidate.question, exclude_texts):
            continue
        accepted.append(candidate)
        exclude_texts.append(candidate.question)

    return accepted


@dataclass
class _ShuffleSubjectOutcome:
    # Purely for structured logging/debugging of one shuffle quiz's AI phase -- never persisted.
    subject: str
    requested: int
    ai_generated: int
    status: str
    reason: str | None = None


async def get_or_create_user(db: AsyncSession, clerk_id: str) -> User:
    stmt = select(User).where(User.clerk_id == clerk_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        logger.info("First login — creating user record for clerk_id=%s", clerk_id)
        # The app fires several requests in parallel right after sign-in (user sync, analytics, sessions), so two of
        # them can both miss the SELECT above and race to INSERT. The savepoint lets the loser's unique-clerk_id
        # violation roll back on its own and re-read the row the winner just created, instead of failing with a 500.
        user = User(clerk_id=clerk_id)
        try:
            async with db.begin_nested():
                db.add(user)
            await db.commit()
        except IntegrityError:
            user = (await db.execute(stmt)).scalar_one()
            return user
        await db.refresh(user)

    return user


@dataclass
class _CallBudget:
    # Shared, mutable across every tier of one generate_quiz() request -- without a shared cap, a single "generate quiz" tap could fire off far more Groq calls than it should (this used to trip Groq's 429 rate limit).
    remaining: int

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


async def _generate_or_cache_tier(
    db: AsyncSession,
    *,
    payload: GenerateQuizRequest,
    user_id: int,
    subject: str,
    lesson: str | None,
    difficulty: str,
    question_count: int,
    force_cache: bool,
    recent_lessons: list[str],
    seen_texts: list[str],
    preferred_lessons: list[str] | None,
    adaptive_context: str | None,
    telemetry_counts: dict,
    call_budget: _CallBudget,
    reference_snippets: list[str] | None = None,
    reference_by_lesson: dict[str, list[str]] | None = None,
) -> tuple[list[Question], bool, bool, bool, int, int, str | None]:
    # Generates (or falls back to cached) `question_count` questions at one subject+difficulty tier — AI grounded in the textbook excerpts first, then AI without excerpts if that fails, then the question pool (unseen-by-this-user first, any cached question as the last resort) for whatever's still short. Factored out so a challenge-zone quiz can call it once per tier; `seen_texts` is shared across tiers for cross-tier dedup, `reference_snippets`/`reference_by_lesson` are RAG excerpts computed once per subject by the caller, and an explicit lesson pin skips the pool entirely. Returns (questions, cache_hit, ai_failed, ai_succeeded, generation_calls_made, duplicate_count, error_category).
    exclude_ids: list[int] = payload.excluded_question_ids or []
    picked_ids: set[int] = set(exclude_ids)

    cache_hit = force_cache
    ai_failed = False
    questions: list[Question] = []
    generation_calls_made = 0
    duplicate_count = 0
    error_category: str | None = None

    # Only meaningful when lesson isn't already pinned — a pinned lesson is already validated/normalized against the curriculum in the request schema. None for grades without curriculum data.
    lesson_choices = (
        curriculum_service.lessons_for(payload.grade, subject) or None
        if lesson is None and curriculum_service.has_curriculum(payload.grade)
        else None
    )

    # Scoped to grade (and, where curriculum data exists, that grade's lesson list) so the blind final fallback below can never hand back a different grade's or off-syllabus question -- same reasoning as _fetch_pool_questions.
    base_filter = [Question.subject == subject, Question.difficulty == difficulty, Question.grade == payload.grade]
    if lesson is not None:
        base_filter.append(Question.lesson == lesson)
    elif lesson_choices:
        base_filter.append(Question.lesson.in_(lesson_choices))

    remaining = question_count - len(questions)

    if remaining > 0 and not force_cache:
        try:
            existing_texts_filter = [Question.subject == subject, Question.difficulty == difficulty]
            if lesson is not None:
                existing_texts_filter.append(Question.lesson == lesson)
            existing_texts_stmt = select(Question.question).where(*existing_texts_filter)
            existing_texts: list[str] = list((await db.execute(existing_texts_stmt)).scalars().all())

            async def _ask_ai(
                snippets: list[str] | None,
                by_lesson: dict[str, list[str]] | None,
                count: int | None = None,
                also_seen: list[str] | None = None,
            ) -> list[dict]:
                nonlocal generation_calls_made, duplicate_count
                # Ask Groq for questions, drop near-duplicates, and if short, ask again for the remainder — up to 3 rounds, keeping whatever unique set results even if a narrow topic can't fill the full count.
                needed = count if count is not None else remaining
                deduped_ai: list[dict] = []
                local_seen = seen_texts + existing_texts + (also_seen or [])
                max_attempts = 3

                for attempt in range(max_attempts):
                    still_needed = needed - len(deduped_ai)
                    if still_needed <= 0:
                        break
                    if not call_budget.consume():
                        logger.info(
                            "Generation call budget exhausted — stopping dedup retries [difficulty=%s]", difficulty,
                        )
                        break

                    batch = await generate_questions(
                        grade=payload.grade,
                        subject=subject,
                        lesson=lesson,
                        lesson_choices=lesson_choices,
                        difficulty=difficulty,
                        question_count=still_needed,
                        existing_questions=local_seen,
                        avoid_lessons=recent_lessons,
                        preferred_lessons=preferred_lessons,
                        adaptive_context=adaptive_context,
                        telemetry=telemetry_counts,
                        reference_snippets=snippets,
                        reference_by_lesson=by_lesson,
                    )
                    generation_calls_made += 1

                    added = 0
                    for q in batch:
                        q_text = q["question"]
                        if not _is_near_duplicate(q_text, local_seen):
                            deduped_ai.append(q)
                            local_seen.append(q_text)
                            added += 1
                        else:
                            duplicate_count += 1
                            logger.info("Dedup: discarded near-duplicate question (attempt %d)", attempt + 1)

                    logger.info(
                        "Dedup attempt %d/%d [difficulty=%s]: +%d accepted, %d total, %d still needed",
                        attempt + 1, max_attempts, difficulty, added, len(deduped_ai), needed - len(deduped_ai),
                    )

                if deduped_ai:
                    return deduped_ai
                if call_budget.consume():
                    fallback_batch = await generate_questions(
                        grade=payload.grade,
                        subject=subject,
                        lesson=lesson,
                        lesson_choices=lesson_choices,
                        difficulty=difficulty,
                        question_count=needed,
                        avoid_lessons=recent_lessons,
                        preferred_lessons=preferred_lessons,
                        adaptive_context=adaptive_context,
                        telemetry=telemetry_counts,
                        reference_snippets=snippets,
                        reference_by_lesson=by_lesson,
                    )
                    generation_calls_made += 1
                    return fallback_batch
                # Budget exhausted before this tier got anything from AI at all — treat it the same as an AI failure so this tier falls back to the pool.
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Generation call budget exhausted for this request.",
                )

            grounded = bool(reference_snippets) or bool(reference_by_lesson)
            try:
                ai_data = await _ask_ai(reference_snippets, reference_by_lesson)
            except HTTPException as grounded_exc:
                if not grounded:
                    raise
                # Grounded generation failed -- try plain AI before falling back to the pool.
                logger.warning(
                    "AI generation with textbook excerpts failed for subject=%s, lesson=%s, difficulty=%s — "
                    "retrying without excerpts: %s",
                    subject, lesson or "<random per question>", difficulty, grounded_exc.detail,
                )
                ai_data = await _ask_ai(None, None)
            else:
                # Grounded generation worked but came back short (a narrow chapter, dropped near-duplicates):
                # plain AI fills the rest before the pool is touched -- knowledge base + AI, then AI only, then pool.
                shortfall = remaining - len(ai_data)
                if grounded and shortfall > 0:
                    try:
                        extra = await _ask_ai(
                            None, None, count=shortfall, also_seen=[q["question"] for q in ai_data],
                        )
                        ai_data = [*ai_data, *extra[:shortfall]]
                    except HTTPException as plain_exc:
                        logger.warning(
                            "AI top-up without excerpts failed for subject=%s, lesson=%s, difficulty=%s — the pool "
                            "covers the remaining %d: %s",
                            subject, lesson or "<random per question>", difficulty, shortfall, plain_exc.detail,
                        )

            for q_data in ai_data:
                question = Question(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    subject=subject,
                    lesson=lesson if lesson is not None else q_data["lesson"],
                    difficulty=difficulty,
                    grade=payload.grade,
                )
                question.pending_source_pages = q_data.get("source_pages")  # plain attribute, see provenance_columns
                db.add(question)
                questions.append(question)
                seen_texts.append(q_data["question"])

            await db.flush()
            picked_ids.update(q.id for q in questions if q.id is not None)

        except HTTPException as exc:
            logger.warning(
                "AI generation failed for subject=%s, lesson=%s, difficulty=%s — falling back to the question pool: %s",
                subject, lesson or "<random per question>", difficulty, exc.detail,
            )
            ai_failed = True
            error_category = telemetry_service.categorize_generation_error(exc)

    # ── Pool fallback #1: whatever AI couldn't cover, from questions this user hasn't been asked yet (an explicit lesson pin skips this -- the pool reader isn't lesson-scoped -- and goes straight to the lesson-filtered fallback below).
    remaining = question_count - len(questions)
    if remaining > 0 and lesson is None:
        pool_questions = await _fetch_pool_questions(
            db,
            user_id=user_id,
            subject=subject,
            difficulty=difficulty,
            grade=payload.grade,
            limit=remaining,
            exclude_ids=picked_ids,
            exclude_texts=seen_texts,
        )
        if pool_questions:
            questions.extend(pool_questions)
            picked_ids.update(q.id for q in pool_questions)
            cache_hit = True

    # ── Pool fallback #2 (blind): whatever's still short, ignoring per-user attempt history (a repeat beats returning fewer questions than requested).
    remaining = question_count - len(questions)
    if remaining > 0:
        # Set True as soon as a fallback is needed at all (not gated on it actually finding rows) — matches the pre-pool "AI alone couldn't cover this" telemetry contract.
        cache_hit = True

        fallback_filter = list(base_filter)
        if picked_ids:
            fallback_filter.append(Question.id.notin_(picked_ids))

        cached_stmt = (
            select(Question)
            .where(*fallback_filter)
            .order_by(func.random())
            .limit(remaining * 3)
        )
        result = await db.execute(cached_stmt)
        # Over-fetched so skipping picture-based rows (see references_missing_visual) still leaves enough.
        fallback_questions = [
            q for q in result.scalars().all() if not references_missing_visual(q.question)
        ][:remaining]

        if fallback_questions:
            questions.extend(fallback_questions)
            seen_texts.extend(q.question for q in fallback_questions)

        # Deliberately NOT raised here — the caller aggregates cache_hit/ai_failed across every tier first and raises itself once that's done, so telemetry reflects the full picture.
        logger.info(
            "Served from DB cache: subject=%s, lesson=%s, difficulty=%s — %d questions (ai_failed=%s, force=%s)",
            subject, lesson or "<any>", difficulty, len(fallback_questions), ai_failed, force_cache,
        )

    ai_succeeded = not ai_failed

    return questions, cache_hit, ai_failed, ai_succeeded, generation_calls_made, duplicate_count, error_category


async def _generate_shuffle_quiz_questions(
    db: AsyncSession,
    *,
    payload: GenerateQuizRequest,
    user: User,
    telemetry_counts: dict,
) -> tuple[list[Question], bool, bool, int, int, str | None]:
    # Shuffle Mode's generation pipeline: ONE Groq call for every subject (each at its own adaptive difficulty) grounded in the textbook excerpts, retried once without excerpts if that fails, then the question pool for whatever's still short (unseen-by-this-user first, then a blind DB-cache top-up); only raises if the quiz still can't reach payload.question_count. Returns (questions, cache_hit, ai_succeeded, generation_calls_made, duplicate_count, error_category) -- the same shape generate_quiz()'s non-shuffle loop produces.
    subjects = payload.subjects or []
    subject_counts = _allocate_shuffle_subject_counts(subjects, payload.question_count)
    selected_subjects = list(subject_counts)

    difficulty_by_subject: dict[str, str] = {
        subject: await difficulty_service.get_subject_difficulty(db, user.id, subject)
        for subject in selected_subjects
    }
    subject_plan = [
        (subject, difficulty_by_subject[subject], subject_counts[subject]) for subject in selected_subjects
    ]

    logger.info(
        "Shuffle generation started requested_count=%d subjects=%d",
        payload.question_count, len(subject_plan),
    )

    generated: list[Question] = []
    used_ids: set[int] = set(payload.excluded_question_ids or [])
    used_texts: list[str] = []
    generation_calls_made = 0
    duplicate_count = 0
    ai_failed = False
    cache_hit = False
    error_category: str | None = None
    deficit: dict[str, int] = {subject: count for subject, _, count in subject_plan}

    total_deficit = sum(max(0, d) for d in deficit.values())

    # ── AI phase: one batched Groq call covering every subject. payload.force_cache skips this phase entirely.
    if not payload.force_cache and total_deficit > 0:
        ai_subject_plan = [
            (subject, difficulty, deficit[subject])
            for subject, difficulty, _count in subject_plan
            if deficit.get(subject, 0) > 0
        ]
        ai_subjects = [subject for subject, _difficulty, _count in ai_subject_plan]
        avoid_lessons_by_subject = {
            subject: await _get_recent_lessons(db, user.id, subject) for subject in ai_subjects
        }
        existing_questions_by_subject = {
            subject: await _get_existing_subject_questions(db, subject) for subject in ai_subjects
        }
        lesson_choices_by_subject = (
            {subject: curriculum_service.lessons_for(payload.grade, subject) for subject in ai_subjects}
            if curriculum_service.has_curriculum(payload.grade)
            else {}
        )
        # RAG grounding excerpts (app/services/rag_service.py), fanned out across every subject with a curriculum lesson list, bounded by ONE combined cap (RAG_MAX_LESSON_QUERIES_SHUFFLE) so a many-subject shuffle can't multiply its retrieval fan-out. Best-effort: never raises, empty on any failure.
        reference_by_lesson_by_subject = await rag_service.get_snippets_by_lesson_for_subjects(
            grade=payload.grade,
            subjects_lessons={s: lesson_choices_by_subject[s] for s in ai_subjects if lesson_choices_by_subject.get(s)},
            max_snippets_per_lesson=settings.RAG_SNIPPETS_PER_LESSON,
            max_total_lesson_queries=settings.RAG_MAX_LESSON_QUERIES_SHUFFLE,
        )
        telemetry_counts["rag_snippet_count"] = sum(
            _snippet_count(None, by_lesson) for by_lesson in reference_by_lesson_by_subject.values()
        )
        async def _ask_ai(references: dict[str, dict[str, list[str]]]) -> list[dict]:
            nonlocal generation_calls_made
            generation_calls_made += 1
            return await generate_shuffle_quiz_questions(
                grade=payload.grade,
                subject_plan=ai_subject_plan,
                avoid_lessons_by_subject=avoid_lessons_by_subject,
                existing_questions_by_subject=existing_questions_by_subject,
                lesson_choices_by_subject=lesson_choices_by_subject,
                telemetry=telemetry_counts,
                reference_by_lesson_by_subject=references,
            )

        try:
            try:
                batch = await _ask_ai(reference_by_lesson_by_subject)
            except HTTPException as grounded_exc:
                if not any(reference_by_lesson_by_subject.values()):
                    raise
                # Grounded generation failed -- try plain AI before falling back to the pool.
                logger.warning("Shuffle AI generation with textbook excerpts failed, retrying without: %s", grounded_exc.detail)
                batch = await _ask_ai({})

            for q_data in batch:
                q_text = q_data["question"]
                if _is_near_duplicate(q_text, used_texts):
                    duplicate_count += 1
                    continue
                subject = q_data["subject"]
                if deficit.get(subject, 0) <= 0:
                    # Already have enough for this subject -- an extra question the AI over-produced doesn't displace another subject's allocation.
                    continue
                question = Question(
                    question=q_text,
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    subject=subject,
                    lesson=q_data["lesson"],
                    difficulty=q_data["difficulty"],
                    grade=payload.grade,
                )
                question.pending_source_pages = q_data.get("source_pages")  # plain attribute, see provenance_columns
                db.add(question)
                generated.append(question)
                used_texts.append(q_text)
                deficit[subject] -= 1

        except HTTPException as exc:
            error_category = telemetry_service.categorize_generation_error(exc)
            ai_failed = True
            logger.warning("Shuffle AI generation failed: %s", exc.detail)
            # Deliberately not re-raised -- falls through to the DB fallback below instead of failing the whole request.

    ai_question_count = len(generated)
    total_deficit = sum(max(0, d) for d in deficit.values())
    logger.info(
        "Shuffle AI phase completed requested=%d ai_generated=%d deficit=%d",
        payload.question_count, ai_question_count, total_deficit,
    )

    # AI-created questions need a flush before their `.id` can be used to exclude them from the pool/DB-fallback queries below.
    await db.flush()
    used_ids.update(q.id for q in generated)

    # ── Pool fallback: per subject, questions this user hasn't been asked yet, only for subjects AI left short.
    for subject, difficulty, _count in subject_plan:
        remaining = deficit.get(subject, 0)
        if remaining <= 0:
            continue
        pool_questions = await _fetch_pool_questions(
            db,
            user_id=user.id,
            subject=subject,
            difficulty=difficulty,
            grade=payload.grade,
            limit=remaining,
            exclude_ids=used_ids,
            exclude_texts=used_texts,
        )
        if not pool_questions:
            continue
        generated.extend(pool_questions)
        used_ids.update(q.id for q in pool_questions)
        deficit[subject] -= len(pool_questions)
        cache_hit = True
    pool_question_count = len(generated) - ai_question_count

    # ── DB fallback #1: per-subject, only for subjects still short ───────
    for subject, _difficulty, _requested_count in subject_plan:
        remaining = deficit.get(subject, 0)
        if remaining <= 0:
            continue
        fallback = await _fetch_shuffle_fallback_questions(
            db,
            subjects=[subject],
            grade=payload.grade,
            difficulties=None,
            limit=remaining,
            exclude_ids=used_ids,
            exclude_texts=used_texts,
        )
        if not fallback:
            continue
        generated.extend(fallback)
        used_ids.update(q.id for q in fallback)
        cache_hit = True
        deficit[subject] -= len(fallback)

    # ── DB fallback #2: global across every selected shuffle subject ───── Covers both "any shuffle subject" and "rebalance onto subjects that already succeeded" in one unrestricted fetch.
    total_deficit = sum(max(0, d) for d in deficit.values())
    if total_deficit > 0:
        global_fallback = await _fetch_shuffle_fallback_questions(
            db,
            subjects=selected_subjects,
            grade=payload.grade,
            difficulties=None,
            limit=total_deficit,
            exclude_ids=used_ids,
            exclude_texts=used_texts,
        )
        if global_fallback:
            generated.extend(global_fallback)
            cache_hit = True
        logger.info(
            "Database fallback generated=%d", len(generated) - pool_question_count - ai_question_count,
        )

    if len(generated) > payload.question_count:
        generated = generated[: payload.question_count]

    if len(generated) < payload.question_count:
        logger.warning(
            "Shuffle generation short requested=%d available=%d", payload.question_count, len(generated),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Unable to generate enough unique questions for Shuffle Mode. "
                f"Requested: {payload.question_count}, Available: {len(generated)}."
            ),
        )

    random.shuffle(generated)

    ai_succeeded = not ai_failed

    logger.info(
        "Shuffle generation completed requested=%d returned=%d pool_questions=%d ai_questions=%d database_questions=%d",
        payload.question_count, len(generated), pool_question_count, ai_question_count,
        len(generated) - pool_question_count - ai_question_count,
    )

    return generated, cache_hit, ai_succeeded, generation_calls_made, duplicate_count, error_category


async def generate_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: GenerateQuizRequest,
) -> tuple[QuizSession, list[Question], bool, str, str]:
    # AI-first: every request calls Groq for a fresh set of questions, so a quiz is never silently served from the DB unless AI generation fails or the caller explicitly asks for force_cache. The caller only has to send subject + question_count -- lesson defaults to a per-question random pick, and difficulty comes from the user's accuracy history via difficulty_service.
    start_time = time.monotonic()
    user = await get_or_create_user(db, clerk_id)

    # Shuffle bypasses lesson/difficulty overrides entirely — each subject always uses its own current adaptive difficulty.
    lesson: str | None = None if payload.shuffle else payload.lesson
    preferred_lessons: list[str] | None = None
    adaptive_context: str | None = None
    reference_snippets: list[str] | None = None
    reference_by_lesson: dict[str, list[str]] | None = None
    telemetry_subject = (
        ", ".join(payload.subjects) if payload.shuffle and payload.subjects else (payload.subject or "unknown")
    )

    # subject_tier_plan is (subject, difficulty, question_count) — the unit _generate_or_cache_tier() is called once per entry for; a normal quiz has 1+ entries sharing payload.subject, while Shuffle quizzes run through the separate _generate_shuffle_quiz_questions() pipeline instead.
    subject_tier_plan: list[tuple[str, str, int]] = []
    challenge_zone_active = False

    if payload.shuffle:
        # Overwritten by the Mixed-difficulty detection below once `questions` is populated -- harmless placeholder until then.
        difficulty = difficulty_service.DEFAULT_DIFFICULTY
    else:
        subject = payload.subject
        assert subject is not None  # enforced by GenerateQuizRequest's model validator

        difficulty = payload.difficulty or (
            await difficulty_service.get_current_difficulty(db, user.id, subject, lesson, grade=payload.grade)
            if lesson is not None
            else await difficulty_service.get_subject_difficulty(db, user.id, subject)
        )

        # Challenge zone + weak-lesson targeting: only for the fully-automatic path (no explicit lesson pin or difficulty override); when active, spreads the quiz across 2-3 difficulty tiers (see CHALLENGE_ZONE_DISTRIBUTION) and nudges the AI toward the subject's weakest lessons, reusing the same single-difficulty per-tier logic just called once per tier.
        challenge_zone_active = lesson is None and payload.difficulty is None
        tier_plan: list[tuple[str, int]] = [(difficulty, payload.question_count)]

        if challenge_zone_active:
            adaptive_summary = await difficulty_service.get_subject_adaptive_summary(db, user.id, subject)
            profile = mastery_engine.get_adaptive_generation_profile(adaptive_summary["mastery_score"])
            tier_counts = mastery_engine.allocate_question_counts(profile.difficulty_distribution, payload.question_count)
            if tier_counts:
                tier_plan = list(tier_counts.items())

            lesson_scores = await difficulty_service.get_subject_lesson_mastery_scores(db, user.id, subject)
            bkt_lesson_scores = await bkt_service.get_subject_lesson_p_know_scores(db, user.id, subject)
            blended_lesson_scores = mastery_engine.blend_lesson_weakness_scores(
                lesson_scores, bkt_lesson_scores, bkt_weight=settings.BKT_LESSON_TARGETING_WEIGHT,
            )
            preferred_lessons = mastery_engine.select_preferred_lessons(blended_lesson_scores) or None
            adaptive_context = mastery_engine.describe_adaptive_context(
                adaptive_summary["mastery_score"], adaptive_summary["confidence_score"], adaptive_summary["trend_label"],
            )

        subject_tier_plan = [(subject, tier_difficulty, tier_count) for tier_difficulty, tier_count in tier_plan]

        # RAG grounding excerpts (app/services/rag_service.py) -- computed ONCE per subject here rather than per tier, since every tier in subject_tier_plan shares the same subject/lesson. Best-effort: never raises, empty on any failure.
        if lesson is not None:
            reference_snippets = await rag_service.get_snippets_for_lesson(
                grade=payload.grade, subject=subject, lesson=lesson,
                max_snippets=settings.RAG_SNIPPETS_LESSON_PINNED,
            )
        else:
            subject_lesson_choices = (
                curriculum_service.lessons_for(payload.grade, subject)
                if curriculum_service.has_curriculum(payload.grade)
                else None
            )
            if subject_lesson_choices:
                reference_by_lesson = await rag_service.get_snippets_by_lesson(
                    grade=payload.grade, subject=subject, lessons=subject_lesson_choices,
                    max_snippets_per_lesson=settings.RAG_SNIPPETS_PER_LESSON,
                    max_lessons=settings.RAG_MAX_LESSON_QUERIES,
                )
            else:
                # No curriculum data for this grade -- no fixed lesson list to query per-lesson, so fall back to one subject-level broad query.
                reference_snippets = await rag_service.get_snippets_for_lesson(
                    grade=payload.grade, subject=subject, lesson=subject,
                    max_snippets=settings.RAG_SNIPPETS_PER_LESSON,
                )

    cache_hit = payload.force_cache
    ai_failed = False
    ai_succeeded = False
    questions: list[Question] = []

    # Telemetry is recorded in the `finally` block below no matter how this function exits, purely for internal monitoring — it never affects what gets returned to the caller.
    created_session_id: int | None = None
    error_category: str | None = None
    generation_calls_made = 0
    duplicate_count = 0
    # Shuffle overwrites rag_snippet_count with its own multi-subject total (_generate_shuffle_quiz_questions).
    telemetry_counts: dict = {
        "invalid_question_count": 0,
        "rag_snippet_count": _snippet_count(reference_snippets, reference_by_lesson),
    }

    try:
        if payload.shuffle:
            (
                questions, cache_hit, ai_succeeded,
                generation_calls_made, duplicate_count, error_category,
            ) = await _generate_shuffle_quiz_questions(
                db, payload=payload, user=user, telemetry_counts=telemetry_counts,
            )
        else:
            # Cached per subject so a challenge-zone quiz (several tiers, same subject) still only fetches this once.
            recent_lessons_by_subject: dict[str, list[str]] = {}

            async def _recent_lessons_for(subj: str) -> list[str]:
                if lesson is not None:
                    return []
                if subj not in recent_lessons_by_subject:
                    recent_lessons_by_subject[subj] = await _get_recent_lessons(db, user.id, subj)
                return recent_lessons_by_subject[subj]

            seen_texts: list[str] = []
            call_budget = _CallBudget(remaining=settings.GROQ_MAX_GENERATION_CALLS_PER_REQUEST)

            for tier_subject, tier_difficulty, tier_count in subject_tier_plan:
                (
                    tier_questions, tier_cache_hit, tier_ai_failed, tier_ai_succeeded,
                    tier_calls, tier_duplicates, tier_error_category,
                ) = await _generate_or_cache_tier(
                    db,
                    payload=payload,
                    user_id=user.id,
                    subject=tier_subject,
                    lesson=lesson,
                    difficulty=tier_difficulty,
                    question_count=tier_count,
                    # A challenge-zone quiz's tiers are all the SAME subject, so once one tier's AI call genuinely failed, cascading that onto later tiers is a reasonable shortcut; deliberately keyed off `ai_failed`, not `cache_hit`, since pool-first generation makes cache_hit go True far more often without meaning AI should be skipped.
                    force_cache=payload.force_cache or ai_failed,
                    recent_lessons=await _recent_lessons_for(tier_subject),
                    seen_texts=seen_texts,
                    preferred_lessons=preferred_lessons,
                    adaptive_context=adaptive_context,
                    telemetry_counts=telemetry_counts,
                    call_budget=call_budget,
                    reference_snippets=reference_snippets,
                    reference_by_lesson=reference_by_lesson,
                )
                questions.extend(tier_questions)
                # A challenge-zone quiz spans several tiers — if ANY fell back to cache or got fresh AI questions, that's worth reflecting in the aggregate flags, not just the last tier processed.
                cache_hit = cache_hit or tier_cache_hit
                ai_failed = ai_failed or tier_ai_failed
                ai_succeeded = ai_succeeded or tier_ai_succeeded
                generation_calls_made += tier_calls
                duplicate_count += tier_duplicates
                if tier_error_category is not None:
                    error_category = tier_error_category

                if not tier_questions and tier_ai_failed:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="AI question generation failed and no cached questions are available for this topic yet.",
                    )

        # `difficulty` above is only the fallback single-tier value; for a quiz spanning more than one difficulty, reflect that as "Mixed" the same way session_lesson/session_subject do -- purely cosmetic/summary since QuestionAttempt/analytics always read each question's own Question.difficulty.
        if payload.shuffle or len(subject_tier_plan) > 1:
            distinct_tier_difficulties = {q.difficulty for q in questions}
            if len(distinct_tier_difficulties) > 1:
                difficulty = "Mixed"
            elif distinct_tier_difficulties:
                difficulty = next(iter(distinct_tier_difficulties))

        if payload.shuffle:
            distinct_subjects = {q.subject for q in questions}
            session_subject = next(iter(distinct_subjects)) if len(distinct_subjects) == 1 else "Mixed"
        else:
            assert payload.subject is not None
            session_subject = payload.subject

        if lesson is not None:
            session_lesson = lesson
        else:
            distinct_lessons = {q.lesson for q in questions}
            if len(distinct_lessons) == 1:
                session_lesson = next(iter(distinct_lessons))
            elif distinct_lessons:
                session_lesson = "Mixed"
            else:
                session_lesson = session_subject

        session = QuizSession(
            user_id=user.id,
            subject=session_subject,
            lesson=session_lesson,
            difficulty=difficulty,
            grade=payload.grade,
            question_count=len(questions),
            questions_snapshot=[_serialize_question(question) for question in questions],
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        created_session_id = session.id

        # Which textbook page each new AI question was based on -- a no-op unless RAG_PROVENANCE_COLUMNS_ENABLED.
        pending_sources = {q.id: q.pending_source_pages for q in questions if getattr(q, "pending_source_pages", None)}
        if pending_sources and provenance_columns.enabled():
            await provenance_columns.save_source_pages(db, pending_sources)
            await db.commit()

        # No per-question refresh here on purpose — every Question already has its id (from flush()'s RETURNING clause or the cache SELECT), so refreshing individually would just add an extra DB round-trip per question.

        logger.info(
            "Created QuizSession id=%d for user=%d with %d questions at lesson=%s, difficulty=%s",
            session.id, user.id, len(questions), session_lesson, difficulty,
        )
        return session, questions, cache_hit, difficulty, session_lesson

    except Exception as exc:
        if error_category is None:
            error_category = telemetry_service.categorize_generation_error(exc)
        raise
    finally:
        # record_generation_event() already catches its own errors, but this is a second safety net -- an exception in a `finally` block replaces whatever this function was about to return/raise.
        try:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            await telemetry_service.record_generation_event(
                db,
                user_id=user.id,
                session_id=created_session_id,
                subject=telemetry_subject,
                requested_question_count=payload.question_count,
                generated_question_count=len(questions),
                provider="groq",
                model_name=settings.GROQ_MODEL,
                success=ai_succeeded,
                used_cache_fallback=cache_hit,
                retry_count=max(generation_calls_made - 1, 0),
                duplicate_count=duplicate_count,
                invalid_question_count=telemetry_counts["invalid_question_count"],
                latency_ms=latency_ms,
                error_category=error_category,
                rag_snippet_count=telemetry_counts.get("rag_snippet_count"),
            )
        except Exception as telemetry_exc:  # noqa: BLE001 — see comment above
            logger.error("AI generation telemetry raised unexpectedly (non-critical): %s", telemetry_exc)


def _lesson_accuracy_breakdown_for(
    answers: list[difficulty_service.GradedAnswer],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for answer in answers:
        stats[answer.lesson]["total"] += 1
        if answer.correct:
            stats[answer.lesson]["correct"] += 1
    return {
        lesson: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round((s["correct"] / s["total"] * 100.0), 2) if s["total"] > 0 else 0.0,
        }
        for lesson, s in stats.items()
    }


def _answers_for_session(session: QuizSession, submitted: list[AnswerItem]) -> list[AnswerItem]:
    """The answers to grade, in the session's own question order.

    The client only says what it picked; WHICH questions count comes from the
    session's questions_snapshot. So an answer for a question that wasn't
    served in this session, or a second answer for the same question, is a
    422 -- otherwise a tampered request could answer one known question ten
    times for 100% (and push that into mastery/BKT). Questions the client left
    out count as unanswered (selected_answer ""), so accuracy is always out of
    the whole quiz. Sessions without a snapshot (legacy rows) fall back to the
    submitted list, still de-duplicated.
    """
    seen: set[int] = set()
    duplicates: set[int] = set()
    for a in submitted:
        if a.question_id in seen:
            duplicates.add(a.question_id)
        seen.add(a.question_id)
    if duplicates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Each question can only be answered once (duplicates: {sorted(duplicates)}).",
        )

    served_ids = [
        q["id"] for q in (session.questions_snapshot or [])
        if isinstance(q, dict) and q.get("id") is not None
    ]
    if not served_ids:
        return list(submitted)

    unknown = sorted(seen - set(served_ids))
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Answers include questions that aren't part of this quiz: {unknown}.",
        )

    by_id = {a.question_id: a for a in submitted}
    return [
        by_id.get(qid) or AnswerItem(question_id=qid, selected_answer="", response_time=0.0)
        for qid in served_ids
    ]


async def submit_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: SubmitQuizRequest,
) -> dict:
    # Grading happens entirely server-side — the frontend only ever sends what the user picked, never the correct answers, so there's nothing to tamper with client-side.
    user = await get_or_create_user(db, clerk_id)

    session = await _get_owned_session(db=db, user_id=user.id, session_id=payload.session_id)

    existing_completion_result = await db.execute(
        select(QuizCompletion).where(QuizCompletion.session_id == session.id)
    )
    if existing_completion_result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This quiz session is already completed.",
        )

    answers = _answers_for_session(session, payload.answers)
    question_ids = [a.question_id for a in answers]
    q_stmt = select(Question).where(Question.id.in_(question_ids))
    q_result = await db.execute(q_stmt)
    question_map: dict[int, Question] = {q.id: q for q in q_result.scalars().all()}

    correct_count = 0
    total_time = 0.0
    lesson_time: dict[str, float] = defaultdict(float)
    lesson_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    repeated_correct_count = 0
    repeated_wrong_count = 0
    repeated_lessons_right: set[str] = set()
    repeated_lessons_wrong: set[str] = set()
    repeated_question_ids = set(payload.repeated_question_ids)
    graded_answers: list[difficulty_service.GradedAnswer] = []
    question_results: list[dict] = []

    for answer in answers:
        question = question_map.get(answer.question_id)
        is_correct: bool | None = None
        lesson = question.lesson if question is not None else "unknown"
        is_repeated = answer.is_repeated or answer.question_id in repeated_question_ids

        if question and question.correct_answer is not None:
            is_correct = (
                str(question.correct_answer).strip().lower()
                == str(answer.selected_answer).strip().lower()
            )
            if is_correct:
                correct_count += 1

        lesson_time[lesson] += answer.response_time
        lesson_stats[lesson]["total"] += 1
        if is_correct:
            lesson_stats[lesson]["correct"] += 1

        if is_repeated and is_correct is not None:
            if is_correct:
                repeated_correct_count += 1
                repeated_lessons_right.add(lesson)
            else:
                repeated_wrong_count += 1
                repeated_lessons_wrong.add(lesson)

        # A question row removed since the quiz was served can't take an attempt
        # (FK) -- it still counts toward the total, just without a stored attempt.
        if question is not None:
            db.add(QuestionAttempt(
                session_id=session.id,
                question_id=answer.question_id,
                selected_answer=answer.selected_answer,
                correct=is_correct,
                response_time=answer.response_time,
            ))
        total_time += answer.response_time
        question_results.append({
            "question_id": answer.question_id,
            "selected_answer": answer.selected_answer,
            "correct_answer": question.correct_answer if question is not None else None,
            "is_correct": is_correct,
        })

        graded_answers.append(difficulty_service.GradedAnswer(
            subject=question.subject if question is not None else session.subject,
            lesson=lesson,
            difficulty=question.difficulty if question is not None else session.difficulty,
            correct=is_correct,
            response_time=answer.response_time,
            fingerprint=question.question_fingerprint if question is not None else "",
        ))

    total_questions = len(answers)
    accuracy = (correct_count / total_questions * 100.0) if total_questions > 0 else 0.0
    avg_response_time = total_time / total_questions if total_questions > 0 else 0.0
    lesson_time_breakdown = {
        lesson: round(seconds, 3) for lesson, seconds in lesson_time.items()
    }

    lesson_accuracy_breakdown: dict[str, dict[str, float | int]] = {}
    for lesson, stats in lesson_stats.items():
        lesson_total = stats["total"]
        lesson_correct = stats["correct"]
        lesson_accuracy_breakdown[lesson] = {
            "correct": lesson_correct,
            "total": lesson_total,
            "accuracy": round((lesson_correct / lesson_total * 100.0), 2)
            if lesson_total > 0
            else 0.0,
        }

    session.score = float(correct_count)
    session.accuracy = accuracy
    session.total_time = total_time

    completion = QuizCompletion(
        session_id=session.id,
        ended_by=payload.ended_by,
        total_time=total_time,
        score=float(correct_count),
        accuracy=accuracy,
        correct_count=correct_count,
        total_questions=total_questions,
        lesson_time_breakdown=lesson_time_breakdown,
        lesson_accuracy_breakdown=lesson_accuracy_breakdown,
        repeated_lessons_right=sorted(repeated_lessons_right),
        repeated_lessons_wrong=sorted(repeated_lessons_wrong),
        repeated_correct_count=repeated_correct_count,
        repeated_wrong_count=repeated_wrong_count,
    )

    await _upsert_progress_snapshot(
        db,
        session.id,
        remaining_time=payload.remaining_time_at_end,
        answered_count=total_questions,
        repeated_question_ids=sorted(repeated_question_ids) if repeated_question_ids else None,
        weak_lessons_hint=sorted(repeated_lessons_wrong),
        draft_answers=[a.model_dump() for a in answers],
    )

    db.add(session)
    db.add(completion)
    try:
        await db.commit()
    except IntegrityError:
        # A second submit (double tap, retry after a slow network) can pass the "already completed" check above
        # before the first one commits; quiz_completions.session_id is unique, so the loser lands here.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This quiz session is already completed.",
        )

    logger.info(
        "Quiz session %d submitted: %d/%d correct, accuracy=%.1f%%, total_time=%.1fs",
        session.id, correct_count, total_questions, accuracy, total_time,
    )

    # Feed this result back into adaptive difficulty — lesson-level first, then subject-level (which rolls up the just-updated LessonMastery rows), non-critical so it never fails the submission response. Grouped by subject since a shuffle submission's graded_answers span multiple subjects. Retakes are graded/persisted normally but must never move adaptive difficulty, since the user has already seen every correct answer.
    if not session.is_retake:
        answers_by_subject: dict[str, list[difficulty_service.GradedAnswer]] = defaultdict(list)
        for graded_answer in graded_answers:
            answers_by_subject[graded_answer.subject].append(graded_answer)

        try:
            for subj, subj_answers in answers_by_subject.items():
                subj_breakdown = _lesson_accuracy_breakdown_for(subj_answers)
                subj_total = len(subj_answers)
                subj_correct = sum(1 for a in subj_answers if a.correct)
                subj_accuracy = (subj_correct / subj_total * 100.0) if subj_total > 0 else 0.0

                await difficulty_service.update_mastery_after_submission(
                    db=db,
                    user_id=user.id,
                    subject=subj,
                    lesson_accuracy_breakdown=subj_breakdown,
                    session=session,
                    ended_by=payload.ended_by,
                    graded_answers=subj_answers,
                    grade=session.grade,
                )
                await difficulty_service.update_subject_mastery_after_submission(
                    db=db,
                    user_id=user.id,
                    subject=subj,
                    accuracy=subj_accuracy,
                    session=session,
                    ended_by=payload.ended_by,
                    graded_answers=subj_answers,
                )
        except Exception as exc:  # noqa: BLE001 — non-critical side-effect, must never fail the submission response
            logger.error("Difficulty mastery update failed (non-critical): %s", exc)

        # Separate try/except from the CEWM update above — BKT is an independent read-only signal, so a failure here shouldn't skip the difficulty-driving update above, and vice versa.
        try:
            await bkt_service.update_bkt_after_submission(
                db=db, user_id=user.id, grade=session.grade, graded_answers=graded_answers,
            )
        except Exception as exc:  # noqa: BLE001 — non-critical side-effect, must never fail the submission response
            logger.error("BKT mastery update failed (non-critical): %s", exc)

    return {
        "session_id": session.id,
        "score": float(correct_count),
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_response_time": avg_response_time,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "ended_by": payload.ended_by,
        "lesson_time_breakdown": lesson_time_breakdown,
        "lesson_accuracy_breakdown": lesson_accuracy_breakdown,
        "repeated_correct_count": repeated_correct_count,
        "repeated_wrong_count": repeated_wrong_count,
        "repeated_lessons_right": sorted(repeated_lessons_right),
        "repeated_lessons_wrong": sorted(repeated_lessons_wrong),
        "question_results": question_results,
        # Not part of SubmitQuizResponse — routes/quiz.py pops this before building the response; it only tells the caller whether to skip the post-submit analytics recompute.
        "is_retake": session.is_retake,
    }


async def _upsert_progress_snapshot(db: AsyncSession, session_id: int, **fields) -> QuizProgressSnapshot:
    """Keep exactly one progress row per session: the client autosaves every few
    seconds, and appending a row each time piled up thousands per quiz. Updates
    the newest row in place (dropping any older ones left from before this) or
    adds the first one. Not committed -- the caller commits."""
    result = await db.execute(
        select(QuizProgressSnapshot)
        .where(QuizProgressSnapshot.session_id == session_id)
        .order_by(QuizProgressSnapshot.saved_at.desc(), QuizProgressSnapshot.id.desc())
    )
    existing = list(result.scalars().all())
    if not existing:
        snapshot = QuizProgressSnapshot(session_id=session_id, **fields)
        db.add(snapshot)
        return snapshot

    snapshot, stale = existing[0], existing[1:]
    for key, value in fields.items():
        setattr(snapshot, key, value)
    snapshot.saved_at = func.now()
    if stale:
        await db.execute(delete(QuizProgressSnapshot).where(QuizProgressSnapshot.id.in_([s.id for s in stale])))
    return snapshot


async def save_quiz_progress(
    db: AsyncSession,
    clerk_id: str,
    payload: SaveProgressRequest,
) -> dict:
    user = await get_or_create_user(db, clerk_id)
    session = await _get_owned_session(db=db, user_id=user.id, session_id=payload.session_id)

    # A late autosave (in flight while the quiz was being submitted) would otherwise become the session's newest
    # snapshot and overwrite the final one submit_quiz wrote -- the sessions list reads answered_count from it.
    completed = await db.execute(select(QuizCompletion.id).where(QuizCompletion.session_id == session.id))
    if completed.first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This quiz session is already completed.",
        )

    snapshot = await _upsert_progress_snapshot(
        db,
        session.id,
        remaining_time=payload.remaining_time,
        answered_count=payload.answered_count,
        repeated_question_ids=payload.repeated_question_ids or None,
        weak_lessons_hint=payload.weak_lessons_hint or None,
        draft_answers=[a.model_dump() for a in payload.draft_answers] or None,
    )
    await db.commit()
    await db.refresh(snapshot)

    return {
        "session_id": session.id,
        "saved_at": snapshot.saved_at,
        "remaining_time": snapshot.remaining_time,
        "answered_count": snapshot.answered_count,
        "repeated_question_ids": snapshot.repeated_question_ids or [],
        "weak_lessons_hint": snapshot.weak_lessons_hint or [],
    }
