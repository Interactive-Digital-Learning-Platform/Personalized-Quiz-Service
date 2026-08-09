import logging
import time
from collections import defaultdict

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuizSession, QuestionAttempt
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.user import User
from app.schemas.quiz import GenerateQuizRequest, SaveProgressRequest, SubmitQuizRequest
from app.services import difficulty_service, telemetry_service
from app.services.groq_service import generate_questions

logger = logging.getLogger(__name__)


async def _get_owned_session(db: AsyncSession, user_id: int, session_id: int) -> QuizSession:
    from fastapi import HTTPException, status

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
    # Lessons this user was recently quizzed on, most recent first — fed to
    # generate_questions() as "please avoid these" so someone who just keeps
    # hitting "generate" on the same subject still gets some topic variety.
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
    # Two questions count as "the same" if they share 55%+ of their
    # meaningful words (stop words and short filler words ignored) — catches
    # Groq rephrasing a question it was already asked not to repeat.
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


async def generate_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: GenerateQuizRequest,
) -> tuple[QuizSession, list[Question], bool, str, str]:
    # AI-first: every request calls Groq for a fresh set of questions, so a
    # quiz is never silently served from the DB unless AI generation actually
    # fails (or the caller explicitly asks for the cached fallback via
    # force_cache, e.g. after already seeing an AI failure once).
    #
    # The caller only has to send subject + question_count. Lesson and
    # difficulty are both picked automatically: lesson defaults to "let the
    # AI assign a different random lesson per question" rather than pinning
    # the whole quiz to one topic; difficulty comes from the user's accuracy
    # history via difficulty_service, tracked independently per subject (or
    # per subject+lesson if a lesson override was given).
    from fastapi import HTTPException, status

    start_time = time.monotonic()
    user = await get_or_create_user(db, clerk_id)

    lesson: str | None = payload.lesson

    difficulty = payload.difficulty or (
        await difficulty_service.get_current_difficulty(db, user.id, payload.subject, lesson)
        if lesson is not None
        else await difficulty_service.get_subject_difficulty(db, user.id, payload.subject)
    )

    exclude_ids: list[int] = payload.excluded_question_ids or []

    base_filter = [
        Question.subject == payload.subject,
        Question.difficulty == difficulty,
    ]
    if lesson is not None:
        base_filter.append(Question.lesson == lesson)
    if exclude_ids:
        base_filter.append(Question.id.notin_(exclude_ids))

    cache_hit = payload.force_cache
    ai_failed = False
    ai_succeeded = False
    questions: list[Question] = []

    # Telemetry is recorded in the `finally` block below no matter how this
    # function exits, purely for internal monitoring — it never affects what
    # gets returned to the caller.
    created_session_id: int | None = None
    error_category: str | None = None
    generation_calls_made = 0
    duplicate_count = 0
    telemetry_counts: dict = {"invalid_question_count": 0}

    try:
        if not cache_hit:
            try:
                existing_texts_filter = [
                    Question.subject == payload.subject,
                    Question.difficulty == difficulty,
                ]
                if lesson is not None:
                    existing_texts_filter.append(Question.lesson == lesson)
                existing_texts_stmt = select(Question.question).where(*existing_texts_filter)
                existing_texts: list[str] = list(
                    (await db.execute(existing_texts_stmt)).scalars().all()
                )

                recent_lessons = (
                    await _get_recent_lessons(db, user.id, payload.subject)
                    if lesson is None
                    else []
                )

                # Ask Groq for questions, drop any that are near-duplicates of
                # ones we already have, and if we're short, ask again for just
                # the remainder — up to 3 rounds. Keeps whatever unique set we
                # end up with even if a narrow topic can't fill the full count.
                needed = payload.question_count
                deduped_ai: list[dict] = []
                seen_texts: list[str] = list(existing_texts)
                max_attempts = 3

                for attempt in range(max_attempts):
                    remaining = needed - len(deduped_ai)
                    if remaining <= 0:
                        break

                    batch = await generate_questions(
                        grade=payload.grade,
                        subject=payload.subject,
                        lesson=lesson,
                        difficulty=difficulty,
                        question_count=remaining,
                        existing_questions=seen_texts,
                        avoid_lessons=recent_lessons,
                        telemetry=telemetry_counts,
                    )
                    generation_calls_made += 1

                    added = 0
                    for q in batch:
                        q_text = q["question"]
                        if not _is_near_duplicate(q_text, seen_texts):
                            deduped_ai.append(q)
                            seen_texts.append(q_text)
                            added += 1
                        else:
                            duplicate_count += 1
                            logger.info("Dedup: discarded near-duplicate question (attempt %d)", attempt + 1)

                    logger.info(
                        "Dedup attempt %d/%d: +%d accepted, %d total, %d still needed",
                        attempt + 1, max_attempts, added, len(deduped_ai), needed - len(deduped_ai),
                    )

                if deduped_ai:
                    ai_data = deduped_ai
                else:
                    ai_data = await generate_questions(
                        grade=payload.grade,
                        subject=payload.subject,
                        lesson=lesson,
                        difficulty=difficulty,
                        question_count=needed,
                        avoid_lessons=recent_lessons,
                        telemetry=telemetry_counts,
                    )
                    generation_calls_made += 1

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
                .limit(payload.question_count)
            )
            result = await db.execute(cached_stmt)
            questions = list(result.scalars().all())

            if not questions and ai_failed:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI question generation failed and no cached questions are available for this topic yet.",
                )

            logger.info(
                "Served from DB cache: subject=%s, lesson=%s, difficulty=%s — %d questions (ai_failed=%s, force=%s)",
                payload.subject, lesson or "<any>", difficulty,
                len(questions), ai_failed, payload.force_cache,
            )

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

        # No per-question refresh here on purpose — every Question object
        # already has its id (from the flush()'s RETURNING clause, or from
        # the cache SELECT), so refreshing each one individually would just
        # be an extra DB round-trip per question for no benefit. That used to
        # add several seconds per quiz over Neon's network latency.

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
        # record_generation_event() already catches its own errors, but this
        # gets a second safety net too — an exception raised inside a
        # `finally` block replaces whatever this function was about to
        # return/raise, so telemetry must never be allowed to blow up the
        # actual request even in a freak case its own handling misses.
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
    # Grading happens entirely server-side — the frontend only ever sends
    # what the user picked, never the correct answers, so there's nothing to
    # tamper with client-side.
    user = await get_or_create_user(db, clerk_id)

    session = await _get_owned_session(db=db, user_id=user.id, session_id=payload.session_id)

    existing_completion_result = await db.execute(
        select(QuizCompletion).where(QuizCompletion.session_id == session.id)
    )
    if existing_completion_result.scalar_one_or_none() is not None:
        from fastapi import HTTPException, status

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

    # Feed this result back into adaptive difficulty — subject-level (what
    # actually drives the default quiz flow) and lesson-level (for the
    # explicit-lesson-override flow). Non-critical: never let this fail the
    # submission response itself.
    try:
        await difficulty_service.update_subject_mastery_after_submission(
            db=db,
            user_id=user.id,
            subject=session.subject,
            accuracy=accuracy,
        )
        await difficulty_service.update_mastery_after_submission(
            db=db,
            user_id=user.id,
            subject=session.subject,
            lesson_accuracy_breakdown=lesson_accuracy_breakdown,
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
