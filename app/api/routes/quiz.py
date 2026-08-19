import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.question import Question
from app.models.quiz_session import QuizSession
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.schemas.quiz import (
    GenerateQuizRequest,
    GenerateQuizResponse,
    QuestionOut,
    QuizSessionSummary,
    RetakeSessionResponse,
    SavedQuizResponse,
    SaveProgressRequest,
    SaveProgressResponse,
    SubmitQuizRequest,
    SubmitQuizResponse,
)
from app.services.analytics_service import update_analytics_after_submission
from app.services.quiz_service import (
    create_retake_session,
    generate_quiz,
    get_or_create_user,
    save_quiz_progress,
    submit_quiz,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz"])


@router.post(
    "/generate",
    response_model=GenerateQuizResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a quiz",
    description=(
        "Generates a set of quiz questions for the given subject, lesson, and difficulty. "
        "Uses a DB-first caching strategy: if questions already exist for these parameters, "
        "they are served directly without calling the AI API."
    ),
)
async def generate_quiz_endpoint(
    payload: GenerateQuizRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> GenerateQuizResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info(
        "Generate quiz request: clerk_id=%s, subject=%s, lesson=%s (override), difficulty=%s (override), count=%d",
        clerk_id, payload.subject, payload.lesson, payload.difficulty, payload.question_count,
    )

    session, questions, cache_hit, difficulty, lesson = await generate_quiz(
        db=db,
        clerk_id=clerk_id,
        payload=payload,
    )

    # QuestionOut never carries correct_answer — don't let it leak to the client here.
    question_out = [QuestionOut.model_validate(q) for q in questions]

    return GenerateQuizResponse(
        session_id=session.id,
        questions=question_out,
        cache_hit=cache_hit,
        difficulty=difficulty,
        lesson=lesson,
    )


@router.post(
    "/sessions/{session_id}/retake",
    response_model=RetakeSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Retake a completed quiz session",
    description=(
        "Clones the given session's questions into a brand-new session for a "
        "'Restart Quiz' attempt. Submitting the new session is graded and shown "
        "to the user like any other, but is excluded from analytics and "
        "adaptive difficulty — the user already saw the correct answers."
    ),
)
async def retake_quiz_session(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RetakeSessionResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    retake = await create_retake_session(db=db, clerk_id=clerk_id, session_id=session_id)
    return RetakeSessionResponse(session_id=retake.id)


@router.get(
    "/sessions",
    response_model=list[QuizSessionSummary],
    status_code=status.HTTP_200_OK,
    summary="List the current user's quiz sessions",
)
async def list_quiz_sessions(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[QuizSessionSummary]:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' claim.",
        )

    user = await get_or_create_user(db=db, clerk_id=clerk_id)

    sessions_result = await db.execute(
        select(QuizSession)
        .options(
            selectinload(QuizSession.completion),
            selectinload(QuizSession.progress_snapshots),
        )
        .where(
            QuizSession.user_id == user.id,
            QuizSession.deleted_at.is_(None),
            # Retakes are an implementation detail of "Restart Quiz" (same
            # questions, answers already known) — they'd just be confusing
            # duplicate entries in the user-facing sessions list.
            QuizSession.is_retake.is_(False),
        )
        .order_by(QuizSession.created_at.desc())
    )
    sessions = list(sessions_result.scalars().all())

    summaries: list[QuizSessionSummary] = []
    for session in sessions:
        latest = (
            max(session.progress_snapshots, key=lambda p: p.saved_at)
            if session.progress_snapshots
            else None
        )
        question_ids = [
            q["id"]
            for q in (session.questions_snapshot or [])
            if isinstance(q, dict) and q.get("id") is not None
        ]
        summaries.append(
            QuizSessionSummary(
                session_id=session.id,
                subject=session.subject,
                difficulty=session.difficulty,
                question_count=session.question_count,
                created_at=session.created_at,
                answered_count=latest.answered_count if latest else 0,
                is_completed=session.completion is not None,
                accuracy=session.completion.accuracy if session.completion else None,
                correct_count=session.completion.correct_count if session.completion else None,
                question_ids=question_ids,
            )
        )

    return summaries


@router.get(
    "/sessions/{session_id}",
    response_model=SavedQuizResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a saved quiz session",
    description="Retrieves a previously generated quiz session and its saved question snapshot from the database.",
)
async def get_saved_quiz_session(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedQuizResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    user = await get_or_create_user(db=db, clerk_id=clerk_id)

    session_stmt = select(QuizSession).where(
        QuizSession.id == session_id,
        QuizSession.user_id == user.id,
        QuizSession.deleted_at.is_(None),
    )
    result = await db.execute(session_stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz session not found.",
        )

    questions_data = session.questions_snapshot or []
    question_ids = [question.get("id") for question in questions_data if question.get("id") is not None]
    question_result = await db.execute(select(Question).where(Question.id.in_(question_ids)))
    question_map = {question.id: question for question in question_result.scalars().all()}

    questions = []
    for question in questions_data:
        enriched_question = dict(question)
        question_id = question.get("id")
        db_question = question_map.get(question_id) if question_id is not None else None
        if db_question is not None and enriched_question.get("correct_answer") is None:
            enriched_question["correct_answer"] = db_question.correct_answer
        questions.append(QuestionOut.model_validate(enriched_question))

    progress_result = await db.execute(
        select(QuizProgressSnapshot)
        .where(QuizProgressSnapshot.session_id == session.id)
        .order_by(QuizProgressSnapshot.saved_at.desc())
        .limit(1)
    )
    latest_progress = progress_result.scalar_one_or_none()

    completion_result = await db.execute(
        select(QuizCompletion).where(QuizCompletion.session_id == session.id)
    )
    completion = completion_result.scalar_one_or_none()

    return SavedQuizResponse(
        session_id=session.id,
        subject=session.subject,
        lesson=session.lesson,
        difficulty=session.difficulty,
        question_count=session.question_count,
        generated_at=session.created_at,
        questions=questions,
        latest_progress={
            "remaining_time": latest_progress.remaining_time,
            "answered_count": latest_progress.answered_count,
            "repeated_question_ids": latest_progress.repeated_question_ids or [],
            "weak_lessons_hint": latest_progress.weak_lessons_hint or [],
            "draft_answers": latest_progress.draft_answers or [],
            "saved_at": latest_progress.saved_at,
        }
        if latest_progress
        else None,
        completion={
            "ended_by": completion.ended_by,
            "total_time": completion.total_time,
            "score": completion.score,
            "accuracy": completion.accuracy,
            "correct_count": completion.correct_count,
            "total_questions": completion.total_questions,
            "lesson_time_breakdown": completion.lesson_time_breakdown or {},
            "lesson_accuracy_breakdown": completion.lesson_accuracy_breakdown or {},
            "repeated_lessons_right": completion.repeated_lessons_right or [],
            "repeated_lessons_wrong": completion.repeated_lessons_wrong or [],
            "repeated_correct_count": completion.repeated_correct_count,
            "repeated_wrong_count": completion.repeated_wrong_count,
            "completed_at": completion.completed_at,
        }
        if completion
        else None,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a quiz session",
)
async def delete_quiz_session(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    # We don't actually delete the row here — see the comment further down.
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' claim.",
        )

    user = await get_or_create_user(db=db, clerk_id=clerk_id)

    result = await db.execute(
        select(QuizSession).where(
            QuizSession.id == session_id,
            QuizSession.user_id == user.id,
            QuizSession.deleted_at.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or does not belong to this user.",
        )

    # This is a soft delete (just stamping deleted_at) instead of a real DELETE.
    # The row needs to stick around because quiz generation still leans on it
    # for lesson-variety history, and the difficulty-adaptation tables key off
    # of it too — losing that data on delete would make the AI forget what
    # you've already studied.
    session.deleted_at = datetime.now(UTC)
    await db.commit()

    # The old per-subject Analytics row only updates when you submit a quiz,
    # so if we don't touch it here it'll keep counting this session's answers
    # until the next submission overwrites it. Recompute it now so it's not stale.
    await update_analytics_after_submission(db=db, clerk_id=clerk_id, session_id=session_id)


@router.post(
    "/progress/save",
    response_model=SaveProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Save quiz progress",
    description=(
        "Saves in-progress quiz state so users can resume after exit/disconnect. "
        "Persists remaining time, answered count, repeated question IDs, and draft answers."
    ),
)
async def save_progress_endpoint(
    payload: SaveProgressRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SaveProgressResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    progress = await save_quiz_progress(db=db, clerk_id=clerk_id, payload=payload)
    return SaveProgressResponse(**progress)


@router.post(
    "/submit",
    response_model=SubmitQuizResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit quiz answers",
    description=(
        "Submit the user's answers for a quiz session. "
        "The server grades each answer, persists the attempts, calculates accuracy "
        "and timing metrics, and updates the user's analytics profile."
    ),
)
async def submit_quiz_endpoint(
    payload: SubmitQuizRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubmitQuizResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info(
        "Quiz submission: clerk_id=%s, session_id=%d, answer_count=%d",
        clerk_id, payload.session_id, len(payload.answers),
    )

    metrics = await submit_quiz(db=db, clerk_id=clerk_id, payload=payload)
    is_retake = metrics.pop("is_retake")

    if not is_retake:
        try:
            await update_analytics_after_submission(
                db=db,
                clerk_id=clerk_id,
                session_id=payload.session_id,
            )
        except Exception as exc:  # noqa: BLE001 — non-critical, the quiz result must still go through
            logger.error("Analytics update failed (non-critical): %s", exc)

    return SubmitQuizResponse(**metrics)


@router.post(
    "/submit-timeout",
    response_model=SubmitQuizResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit quiz on timeout",
    description=(
        "Finalizes quiz when timer reaches zero. Grades provided answers, persists "
        "attempts, and stores completion state with ended_by=timeout."
    ),
)
async def submit_timeout_quiz_endpoint(
    payload: SubmitQuizRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubmitQuizResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    timeout_payload = payload.model_copy(update={"ended_by": "timeout"})
    metrics = await submit_quiz(db=db, clerk_id=clerk_id, payload=timeout_payload)
    is_retake = metrics.pop("is_retake")

    if not is_retake:
        try:
            await update_analytics_after_submission(
                db=db,
                clerk_id=clerk_id,
                session_id=payload.session_id,
            )
        except Exception as exc:  # noqa: BLE001 — non-critical, the quiz result must still go through
            logger.error("Analytics update failed (non-critical): %s", exc)

    return SubmitQuizResponse(**metrics)
