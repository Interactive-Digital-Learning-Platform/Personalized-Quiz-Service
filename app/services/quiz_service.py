import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuizSession, QuestionAttempt
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.user import User
from app.schemas.quiz import GenerateQuizRequest, SaveProgressRequest, SubmitQuizRequest
from app.services import difficulty_mastery_engine as mastery_engine
from app.services import difficulty_service, telemetry_service
from app.services.groq_service import generate_questions

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
    # Recently quizzed lessons, fed to generate_questions() as "avoid these" for variety.
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


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
    "or", "which", "what", "how", "why", "when", "where", "that", "this",
    "it", "its", "be", "been", "by", "for", "with", "as", "at", "from",
    "on", "not", "does", "do", "did", "will", "would", "can", "could",
    "following", "given", "following", "most", "one", "two", "three",
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
    # Same question if 55%+ of meaningful words overlap — catches Groq rephrasing.
    for ex in existing:
        if _jaccard_similarity(candidate, ex) >= threshold:
            return True
    return False


async def get_or_create_user(db: AsyncSession, clerk_id: str) -> User:
    stmt = select(User).where(User.clerk_id == clerk_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        logger.info("First login — creating user record for clerk_id=%s", clerk_id)
        user = User(clerk_id=clerk_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user


@dataclass
class _CallBudget:
    # Shared across every tier of one request -- caps total Groq calls
    # since each tier otherwise retries independently (this is what was
    # tripping Groq's 429 rate limit before this budget existed).
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
) -> tuple[list[Question], bool, bool, bool, int, int, str | None]:
    """Generates (or falls back to cached) `question_count` questions at one
    difficulty tier — the same AI-first/dedup/cache-fallback logic
    generate_quiz() always used for its single difficulty, now factored out
    so a challenge-zone quiz can call it once per tier. `seen_texts` is
    both read (as prior questions to avoid) and appended to in place, so
    multiple tiers in one quiz share a single cross-tier dedup pool instead
    of only deduping within each tier.

    Returns (questions, cache_hit, ai_failed, ai_succeeded,
    generation_calls_made, duplicate_count, error_category).
    """
    exclude_ids: list[int] = payload.excluded_question_ids or []
    base_filter = [Question.subject == payload.subject, Question.difficulty == difficulty]
    if lesson is not None:
        base_filter.append(Question.lesson == lesson)
    if exclude_ids:
        base_filter.append(Question.id.notin_(exclude_ids))

    cache_hit = force_cache
    ai_failed = False
    ai_succeeded = False
    questions: list[Question] = []
    generation_calls_made = 0
    duplicate_count = 0
    error_category: str | None = None

    if not cache_hit:
        try:
            existing_texts_filter = [Question.subject == payload.subject, Question.difficulty == difficulty]
            if lesson is not None:
                existing_texts_filter.append(Question.lesson == lesson)
            existing_texts_stmt = select(Question.question).where(*existing_texts_filter)
            existing_texts: list[str] = list((await db.execute(existing_texts_stmt)).scalars().all())

            # Drop near-duplicates, ask again for the remainder, up to 3 rounds.
            needed = question_count
            deduped_ai: list[dict] = []
            local_seen = seen_texts + existing_texts
            max_attempts = 3

            for attempt in range(max_attempts):
                remaining = needed - len(deduped_ai)
                if remaining <= 0:
                    break
                if not call_budget.consume():
                    logger.info(
                        "Generation call budget exhausted — stopping dedup retries [difficulty=%s]", difficulty,
                    )
                    break

                batch = await generate_questions(
                    grade=payload.grade,
                    subject=payload.subject,
                    lesson=lesson,
                    difficulty=difficulty,
                    question_count=remaining,
                    existing_questions=local_seen,
                    avoid_lessons=recent_lessons,
                    preferred_lessons=preferred_lessons,
                    adaptive_context=adaptive_context,
                    telemetry=telemetry_counts,
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
                ai_data = deduped_ai
            elif call_budget.consume():
                ai_data = await generate_questions(
                    grade=payload.grade,
                    subject=payload.subject,
                    lesson=lesson,
                    difficulty=difficulty,
                    question_count=needed,
                    avoid_lessons=recent_lessons,
                    preferred_lessons=preferred_lessons,
                    adaptive_context=adaptive_context,
                    telemetry=telemetry_counts,
                )
                generation_calls_made += 1
            else:
                # No budget left and nothing from AI -- fall back to cache like an AI failure.
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Generation call budget exhausted for this request.",
                )

            for q_data in ai_data:
                question = Question(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    subject=payload.subject,
                    lesson=lesson if lesson is not None else q_data["lesson"],
                    difficulty=difficulty,
                )
                db.add(question)
                questions.append(question)
                seen_texts.append(q_data["question"])

            await db.flush()
            ai_succeeded = True

        except HTTPException as exc:
            logger.warning(
                "AI generation failed for subject=%s, lesson=%s, difficulty=%s — falling back to DB cache: %s",
                payload.subject, lesson or "<random per question>", difficulty, exc.detail,
            )
            ai_failed = True
            cache_hit = True
            error_category = telemetry_service.categorize_generation_error(exc)

    if cache_hit:
        cached_stmt = (
            select(Question)
            .where(*base_filter)
            .order_by(func.random())
            .limit(question_count)
        )
        result = await db.execute(cached_stmt)
        questions = list(result.scalars().all())

        # Not raised here -- the caller aggregates across tiers and raises once, after.

        logger.info(
            "Served from DB cache: subject=%s, lesson=%s, difficulty=%s — %d questions (ai_failed=%s, force=%s)",
            payload.subject, lesson or "<any>", difficulty, len(questions), ai_failed, force_cache,
        )
        seen_texts.extend(q.question for q in questions)

    return questions, cache_hit, ai_failed, ai_succeeded, generation_calls_made, duplicate_count, error_category


async def generate_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: GenerateQuizRequest,
) -> tuple[QuizSession, list[Question], bool, str, str]:
    # AI-first: only falls back to the DB cache if generation fails or
    # force_cache is set. Lesson/difficulty are auto-picked unless overridden.
    start_time = time.monotonic()
    user = await get_or_create_user(db, clerk_id)

    lesson: str | None = payload.lesson

    difficulty = payload.difficulty or (
        await difficulty_service.get_current_difficulty(db, user.id, payload.subject, lesson)
        if lesson is not None
        else await difficulty_service.get_subject_difficulty(db, user.id, payload.subject)
    )

    # Challenge zone: only when neither lesson nor difficulty is overridden.
    # Spreads questions across 2-3 difficulty tiers (CHALLENGE_ZONE_DISTRIBUTION)
    # and nudges toward weak lessons, each tier reusing the same single-difficulty path.
    challenge_zone_active = lesson is None and payload.difficulty is None
    tier_plan: list[tuple[str, int]] = [(difficulty, payload.question_count)]
    preferred_lessons: list[str] | None = None
    adaptive_context: str | None = None

    if challenge_zone_active:
        adaptive_summary = await difficulty_service.get_subject_adaptive_summary(db, user.id, payload.subject)
        profile = mastery_engine.get_adaptive_generation_profile(adaptive_summary["mastery_score"])
        tier_counts = mastery_engine.allocate_question_counts(profile.difficulty_distribution, payload.question_count)
        if tier_counts:
            tier_plan = list(tier_counts.items())

        lesson_scores = await difficulty_service.get_subject_lesson_mastery_scores(db, user.id, payload.subject)
        preferred_lessons = mastery_engine.select_preferred_lessons(lesson_scores) or None
        adaptive_context = mastery_engine.describe_adaptive_context(
            adaptive_summary["mastery_score"], adaptive_summary["confidence_score"], adaptive_summary["trend_label"],
        )

    cache_hit = payload.force_cache
    ai_failed = False
    ai_succeeded = False
    questions: list[Question] = []

    # Telemetry is recorded in `finally` below regardless of outcome; purely internal.
    created_session_id: int | None = None
    error_category: str | None = None
    generation_calls_made = 0
    duplicate_count = 0
    telemetry_counts: dict = {"invalid_question_count": 0}

    try:
        recent_lessons = (
            await _get_recent_lessons(db, user.id, payload.subject)
            if lesson is None
            else []
        )
        seen_texts: list[str] = []
        call_budget = _CallBudget(remaining=settings.GROQ_MAX_GENERATION_CALLS_PER_REQUEST)

        for tier_difficulty, tier_count in tier_plan:
            (
                tier_questions, tier_cache_hit, tier_ai_failed, tier_ai_succeeded,
                tier_calls, tier_duplicates, tier_error_category,
            ) = await _generate_or_cache_tier(
                db,
                payload=payload,
                lesson=lesson,
                difficulty=tier_difficulty,
                question_count=tier_count,
                force_cache=cache_hit,
                recent_lessons=recent_lessons,
                seen_texts=seen_texts,
                preferred_lessons=preferred_lessons,
                adaptive_context=adaptive_context,
                telemetry_counts=telemetry_counts,
                call_budget=call_budget,
            )
            questions.extend(tier_questions)
            # Aggregate across tiers rather than only reflecting the last one.
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

        # Cosmetic only: session.difficulty becomes "Mixed" if tiers actually
        # varied (same idea as session_lesson becoming "Mixed"); grading always
        # reads each question's own Question.difficulty.
        if len(tier_plan) > 1:
            distinct_tier_difficulties = {q.difficulty for q in questions}
            if len(distinct_tier_difficulties) > 1:
                difficulty = "Mixed"

        if lesson is not None:
            session_lesson = lesson
        else:
            distinct_lessons = {q.lesson for q in questions}
            if len(distinct_lessons) == 1:
                session_lesson = next(iter(distinct_lessons))
            elif distinct_lessons:
                session_lesson = "Mixed"
            else:
                session_lesson = payload.subject

        session = QuizSession(
            user_id=user.id,
            subject=payload.subject,
            lesson=session_lesson,
            difficulty=difficulty,
            question_count=len(questions),
            questions_snapshot=[_serialize_question(question) for question in questions],
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        created_session_id = session.id

        # No per-question refresh -- ids are already set (flush RETURNING or cache SELECT).

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
        # Extra safety net -- an exception in `finally` would replace the real
        # return/raise, so telemetry must never blow up the request.
        try:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            await telemetry_service.record_generation_event(
                db,
                user_id=user.id,
                session_id=created_session_id,
                subject=payload.subject,
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
            )
        except Exception as telemetry_exc:  # noqa: BLE001 — see comment above
            logger.error("AI generation telemetry raised unexpectedly (non-critical): %s", telemetry_exc)


async def submit_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: SubmitQuizRequest,
) -> dict:
    # Grading is server-side -- the client never receives correct answers to tamper with.
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

    question_ids = [a.question_id for a in payload.answers]
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

    for answer in payload.answers:
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

        attempt = QuestionAttempt(
            session_id=session.id,
            question_id=answer.question_id,
            selected_answer=answer.selected_answer,
            correct=is_correct,
            response_time=answer.response_time,
        )
        db.add(attempt)
        total_time += answer.response_time

        graded_answers.append(difficulty_service.GradedAnswer(
            lesson=lesson,
            difficulty=question.difficulty if question is not None else session.difficulty,
            correct=is_correct,
            response_time=answer.response_time,
            fingerprint=question.question_fingerprint if question is not None else "",
        ))

    total_questions = len(payload.answers)
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

    final_snapshot = QuizProgressSnapshot(
        session_id=session.id,
        remaining_time=payload.remaining_time_at_end,
        answered_count=total_questions,
        repeated_question_ids=sorted(repeated_question_ids) if repeated_question_ids else None,
        weak_lessons_hint=sorted(repeated_lessons_wrong),
        draft_answers=[a.model_dump() for a in payload.answers],
    )

    db.add(session)
    db.add(completion)
    db.add(final_snapshot)
    await db.commit()

    logger.info(
        "Quiz session %d submitted: %d/%d correct, accuracy=%.1f%%, total_time=%.1fs",
        session.id, correct_count, total_questions, accuracy, total_time,
    )

    # Lesson-level first, then subject-level (which rolls up lesson mastery
    # and needs it fresh). Non-critical: must never fail the submission itself.
    try:
        await difficulty_service.update_mastery_after_submission(
            db=db,
            user_id=user.id,
            subject=session.subject,
            lesson_accuracy_breakdown=lesson_accuracy_breakdown,
            session=session,
            ended_by=payload.ended_by,
            graded_answers=graded_answers,
        )
        await difficulty_service.update_subject_mastery_after_submission(
            db=db,
            user_id=user.id,
            subject=session.subject,
            accuracy=accuracy,
            session=session,
            ended_by=payload.ended_by,
            graded_answers=graded_answers,
        )
    except Exception as exc:
        logger.error("Difficulty mastery update failed (non-critical): %s", exc)

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
    }


async def save_quiz_progress(
    db: AsyncSession,
    clerk_id: str,
    payload: SaveProgressRequest,
) -> dict:
    user = await get_or_create_user(db, clerk_id)
    session = await _get_owned_session(db=db, user_id=user.id, session_id=payload.session_id)

    snapshot = QuizProgressSnapshot(
        session_id=session.id,
        remaining_time=payload.remaining_time,
        answered_count=payload.answered_count,
        repeated_question_ids=payload.repeated_question_ids or None,
        weak_lessons_hint=payload.weak_lessons_hint or None,
        draft_answers=[a.model_dump() for a in payload.draft_answers] or None,
    )
    db.add(snapshot)
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
