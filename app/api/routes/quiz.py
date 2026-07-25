"""
api/routes/quiz.py
──────────────────
Quiz-related API endpoints:

    POST /quiz/generate  — Generate a quiz (DB-cached or AI-generated)
    POST /quiz/submit    — Submit answers and get scoring results
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    SavedQuizResponse,
    SaveProgressRequest,
    SaveProgressResponse,
    SubmitQuizRequest,
    SubmitQuizResponse,
)
from app.services.analytics_service import update_analytics_after_submission
from sqlalchemy.orm import selectinload

from app.services.quiz_service import (
    generate_quiz,
    get_or_create_user,
    save_quiz_progress,
    submit_quiz,
)

logger = logging.getLogger(__name__)

# All routes in this file are prefixed with /quiz
router = APIRouter(prefix="/quiz", tags=["Quiz"])


# ─────────────────────────────────────────────────────────────────────────────
# POST /quiz/generate
# ─────────────────────────────────────────────────────────────────────────────

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
    # `get_current_user` verifies the Clerk JWT and returns the decoded payload
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> GenerateQuizResponse:
    """
    Flow:
    1. Extract Clerk user ID from the verified JWT payload.
    2. Delegate to quiz_service.generate_quiz (which handles caching + AI call).
    3. Build and return the response — note we NEVER include correct_answer here.
    """
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

    # Convert ORM Question objects to Pydantic response schemas
    # IMPORTANT: correct_answer is intentionally excluded in QuestionOut
    question_out = [QuestionOut.model_validate(q) for q in questions]

    return GenerateQuizResponse(
        session_id=session.id,
        questions=question_out,
        cache_hit=cache_hit,
        difficulty=difficulty,
        lesson=lesson,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /quiz/sessions  — list all sessions for the current user
# ─────────────────────────────────────────────────────────────────────────────

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
        .where(QuizSession.user_id == user.id)
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


# ─────────────────────────────────────────────────────────────────────────────
# GET /quiz/sessions/{session_id}
# ─────────────────────────────────────────────────────────────────────────────

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
        db_question = question_map.get(question.get("id"))
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


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /quiz/sessions/{session_id}
# Deletes the quiz session and all related records (attempts, snapshots,
# completion). The aggregated Analytics table is unaffected, so historical
# subject-level accuracy is preserved.
# ─────────────────────────────────────────────────────────────────────────────

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
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or does not belong to this user.",
        )

    await db.delete(session)
    await db.commit()


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


# ─────────────────────────────────────────────────────────────────────────────
# POST /quiz/submit
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Flow:
    1. Grade all answers server-side (correct_answer fetched from DB).
    2. Persist QuestionAttempt rows.
    3. Update QuizSession with aggregate metrics.
    4. Trigger analytics upsert for the user's profile.
    5. Return computed results immediately.
    """
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

    # Grade the submission and persist attempts
    metrics = await submit_quiz(db=db, clerk_id=clerk_id, payload=payload)

    # Update the analytics table (upsert) — non-blocking best-effort
    try:
        await update_analytics_after_submission(
            db=db,
            clerk_id=clerk_id,
            session_id=payload.session_id,
        )
    except Exception as exc:
        # Analytics update failure should NOT fail the submission response
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

    try:
        await update_analytics_after_submission(
            db=db,
            clerk_id=clerk_id,
            session_id=payload.session_id,
        )
    except Exception as exc:
        logger.error("Analytics update failed (non-critical): %s", exc)

    return SubmitQuizResponse(**metrics)
