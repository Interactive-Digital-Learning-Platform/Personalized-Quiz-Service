"""
services/quiz_service.py
────────────────────────
Business logic for the quiz generation and submission flow.

Key responsibility: AI-first generation for `generate_quiz`.

Every call to `/quiz/generate` triggers a fresh Groq API call so the user
always gets new questions. The DB question pool is only used as a fallback
when the AI call fails, or when the caller explicitly requests the cached
fallback via `force_cache` (used after a prior AI failure was shown in the UI).

The caller only needs to supply `subject` and `question_count`. Lesson and
difficulty are both chosen automatically:
- Lesson: each question is independently assigned a different, randomly
  varied lesson within the subject by the AI (see groq_service.generate_questions) —
  quizzes are NOT pinned to one pre-selected topic.
- Difficulty: read from `difficulty_service`'s accuracy history.
Both remain overridable if the caller explicitly supplies them.
"""
import logging
from collections import defaultdict

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question
from app.models.quiz_session import QuizSession, QuestionAttempt
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.user import User
from app.schemas.quiz import GenerateQuizRequest, SaveProgressRequest, SubmitQuizRequest
from app.services import difficulty_service
from app.services.groq_service import generate_questions

logger = logging.getLogger(__name__)


async def _get_owned_session(db: AsyncSession, user_id: int, session_id: int) -> QuizSession:
    from fastapi import HTTPException, status

    session_stmt = select(QuizSession).where(
        QuizSession.id == session_id,
        QuizSession.user_id == user_id,
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
    """
    Distinct lessons this user was recently quizzed on for this subject, most
    recent first. Passed to `generate_questions()` as `avoid_lessons` so
    repeated "just pick a subject" requests get lesson variety across sessions,
    not just within one quiz's random per-question assignment.
    """
    stmt = (
        select(QuizSession.lesson)
        .where(QuizSession.user_id == user_id, QuizSession.subject == subject)
        .order_by(QuizSession.created_at.desc())
        .limit(limit * 3)  # over-fetch before de-duping, since sessions repeat lessons
    )
    rows = (await db.execute(stmt)).scalars().all()

    seen: list[str] = []
    for lesson in rows:
        if lesson and lesson != "Mixed" and lesson not in seen:
            seen.append(lesson)
        if len(seen) >= limit:
            break
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Near-duplicate detection via word-level Jaccard similarity
# ─────────────────────────────────────────────────────────────────────────────

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
    """Return True if candidate shares ≥ threshold significant-word overlap with any existing question."""
    for ex in existing:
        if _jaccard_similarity(candidate, ex) >= threshold:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Get or create the internal User record from a Clerk ID
# ─────────────────────────────────────────────────────────────────────────────

async def get_or_create_user(db: AsyncSession, clerk_id: str) -> User:
    """
    Look up a user by their Clerk ID. Create the record if it doesn't exist yet.
    This is called on every authenticated request to lazily provision users.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. QUIZ GENERATION (AI-first, DB cache is a failure fallback only)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: GenerateQuizRequest,
) -> tuple[QuizSession, list[Question], bool, str, str]:
    """
    Generate a quiz session with questions.

    Strategy (AI-first):
    ─────────────────────
    1. Always call the Groq API to generate a fresh set of questions, so every
       quiz a user takes is genuinely new — never silently served from the DB.
    2. Only fall back to the `questions` table (subject, lesson, difficulty)
       cache if the AI call fails outright (network error, malformed output,
       etc.), or if the caller explicitly passes `force_cache=True` (the
       user-triggered fallback after seeing an AI failure in the UI).

    Lesson and difficulty are both AI/system-chosen by default — the caller
    only needs to supply `subject` and `question_count`:
    - `lesson`: if omitted (the default), each question is independently
      assigned a different, randomly varied lesson within the subject by the
      AI itself as part of the same generation call — the quiz is NOT pinned
      to one pre-selected topic. Pass `lesson` explicitly to force every
      question onto one specific topic instead (manual override).
    - `difficulty`: if omitted, `difficulty_service` reads accuracy history and
      promotes/demotes it over time — per (subject, lesson) if a single lesson
      was given, or per (subject) overall otherwise (`SubjectMastery`), since
      there's no single lesson to look up when lessons vary per question. This
      is what makes e.g. a student's Maths quizzes gradually get harder while
      their Science quizzes stay easy — tracked independently per subject.

    Returns:
        (session, questions, cache_hit, difficulty, lesson)
        - session: the newly created QuizSession ORM object
        - questions: list of Question ORM objects
        - cache_hit: True if we served from DB cache (AI failed or force_cache)
        - difficulty: the difficulty level actually used
        - lesson: the session-level lesson label — the fixed lesson if one was
          given/used, "Mixed" if the questions span multiple lessons, or the
          single common lesson if they all happen to share one
    """
    from fastapi import HTTPException, status

    user = await get_or_create_user(db, clerk_id)

    # None (the default) means: let the AI assign a different, random lesson
    # to each question within the subject, instead of pinning the whole quiz
    # to one pre-selected topic.
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

    # force_cache=True means the user explicitly chose the DB fallback after AI failed
    cache_hit = payload.force_cache
    ai_failed = False

    if not cache_hit:
        # ── Step 1: Call Groq, dedup against DB, retry until we have enough ───
        try:
            # Fetch all existing question texts for this topic so the prompt can
            # explicitly exclude them and the AI generates genuinely new content.
            # When lesson is None (random per-question mode), this spans the
            # whole subject rather than one topic.
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

            # Only meaningful in random-lesson mode — encourages variety across
            # separate quiz generations, not just within one quiz's batch.
            recent_lessons = (
                await _get_recent_lessons(db, user.id, payload.subject)
                if lesson is None
                else []
            )

            needed = payload.question_count
            deduped_ai: list[dict] = []
            # Grows with every accepted question so within-batch duplicates are also caught
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
                )

                added = 0
                for q in batch:
                    q_text = q["question"]
                    if not _is_near_duplicate(q_text, seen_texts):
                        deduped_ai.append(q)
                        seen_texts.append(q_text)
                        added += 1
                    else:
                        logger.info("Dedup: discarded near-duplicate question (attempt %d)", attempt + 1)

                logger.info(
                    "Dedup attempt %d/%d: +%d accepted, %d total, %d still needed",
                    attempt + 1, max_attempts, added, len(deduped_ai), needed - len(deduped_ai),
                )

            # Use however many unique questions we got (may be < needed if topic is narrow)
            ai_data = deduped_ai or (await generate_questions(
                grade=payload.grade,
                subject=payload.subject,
                lesson=lesson,
                difficulty=difficulty,
                question_count=needed,
                avoid_lessons=recent_lessons,
            ))

            questions: list[Question] = []
            for q_data in ai_data:
                question = Question(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    subject=payload.subject,
                    # Fixed lesson mode: same lesson for every question.
                    # Random mode: each question keeps its own AI-assigned lesson.
                    lesson=lesson if lesson is not None else q_data["lesson"],
                    difficulty=difficulty,
                )
                db.add(question)
                questions.append(question)

            # Flush to get DB-assigned IDs before creating the session
            await db.flush()

        except HTTPException as exc:
            logger.warning(
                "AI generation failed for subject=%s, lesson=%s, difficulty=%s — falling back to DB cache: %s",
                payload.subject, lesson or "<random per question>", difficulty, exc.detail,
            )
            ai_failed = True
            cache_hit = True

    if cache_hit:
        # ── Step 2: Fetch from cache, random sample of unseen questions ───────
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

    # ── Determine the session-level lesson label ───────────────────────────────
    # If a single lesson was given/used, that's the label. Otherwise (random
    # per-question mode, or a lesson-agnostic cache fallback), derive a label
    # from whatever lessons the served questions actually belong to.
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

    # ── Step 3: Create a new QuizSession ──────────────────────────────────────
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

    # NOTE: no per-question refresh here on purpose. Every Question object
    # already has its `id` populated (via the `db.flush()` RETURNING clause
    # for newly-created rows, or already-loaded from the cache SELECT), and
    # every other field was set explicitly at construction/fetch time. A
    # refresh-per-question loop here used to add one DB round-trip per
    # question — with Neon's network latency that alone added several
    # seconds per quiz for no benefit.

    logger.info(
        "Created QuizSession id=%d for user=%d with %d questions at lesson=%s, difficulty=%s",
        session.id, user.id, len(questions), session_lesson, difficulty,
    )
    return session, questions, cache_hit, difficulty, session_lesson


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUIZ SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

async def submit_quiz(
    db: AsyncSession,
    clerk_id: str,
    payload: SubmitQuizRequest,
) -> dict:
    """
    Process a quiz submission:
    1. Fetch the session (verify it belongs to this user).
    2. For each answer, compare against the correct answer in the DB.
    3. Create QuestionAttempt records.
    4. Calculate score, accuracy, total_time, avg_response_time.
    5. Update the QuizSession with the results.
    6. Return computed metrics for immediate display.

    Scoring is done server-side — the frontend only sends selected answers,
    never the correct ones, preventing cheating.
    """
    # ── Look up user ───────────────────────────────────────────────────────────
    user = await get_or_create_user(db, clerk_id)

    # ── Verify session ownership ───────────────────────────────────────────────
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

    # ── Batch-fetch all questions referenced in the submission ─────────────────
    question_ids = [a.question_id for a in payload.answers]
    q_stmt = select(Question).where(Question.id.in_(question_ids))
    q_result = await db.execute(q_stmt)
    question_map: dict[int, Question] = {q.id: q for q in q_result.scalars().all()}

    # ── Grade each answer and persist QuestionAttempt rows ────────────────────
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
            # Case-insensitive, whitespace-stripped comparison
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

    # ── Compute aggregate metrics ──────────────────────────────────────────────
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

    # ── Update QuizSession ─────────────────────────────────────────────────────
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

    # Store one final progress snapshot for resume/debug/analytics consistency.
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

    # ── Adaptive difficulty: feed this quiz's results back in ──────────────────
    # Non-critical — a failure here should never break the submission response.
    # Subject-level: drives the default random-lesson quiz flow, using overall
    # accuracy (this is what makes e.g. Maths gradually get harder while
    # Science stays easy, independently per subject).
    # Lesson-level: drives the narrower explicit-lesson-override flow.
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
    """Save an in-progress snapshot when user exits or pauses a quiz."""

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
