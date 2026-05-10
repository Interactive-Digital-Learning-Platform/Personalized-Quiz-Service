"""
api/routes/analytics.py
────────────────────────
Analytics-related API endpoints:

    GET /analytics/me        — User's full performance profile
    GET /analytics/feedback  — AI-generated personalised improvement suggestions
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.schemas.analytics import AIFeedbackResponse, UserAnalyticsResponse
from app.services.analytics_service import get_user_analytics
from app.services.groq_service import generate_feedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ─────────────────────────────────────────────────────────────────────────────
# GET /analytics/me
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get my analytics",
    description=(
        "Returns the authenticated user's full performance profile: "
        "overall accuracy, average response time, per-subject breakdown, "
        "and identified strong/weak subjects."
    ),
)
async def get_my_analytics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserAnalyticsResponse:
    """
    Fetches aggregated data from the `analytics` table for the current user.
    No AI call is made here — this is pure DB data, so it's fast.
    """
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /analytics/me — clerk_id=%s", clerk_id)

    analytics_data = await get_user_analytics(db=db, clerk_id=clerk_id)

    return UserAnalyticsResponse(**analytics_data)


# ─────────────────────────────────────────────────────────────────────────────
# GET/POST /analytics/feedback
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/feedback",
    response_model=AIFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Get AI feedback",
    description=(
        "Pulls the user's latest analytics from the DB, feeds it to the Groq AI, "
        "and returns personalised study suggestions, weak areas, and an encouraging note. "
        "Note: This endpoint calls the Groq API on every request (~1-2s latency)."
    ),
)
@router.post(
    "/feedback",
    response_model=AIFeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Get AI feedback",
    description=(
        "Pulls the user's latest analytics from the DB, feeds it to the Groq AI, "
        "and returns personalised study suggestions, weak areas, and an encouraging note."
    ),
)
async def get_ai_feedback(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AIFeedbackResponse:
    """
    Flow:
    1. Fetch the user's analytics from DB.
    2. If no data yet, return a friendly first-time message.
    3. Build a structured summary dict for the Groq prompt.
    4. Call groq_service.generate_feedback() and return the response.
    """
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /analytics/feedback — clerk_id=%s", clerk_id)

    # ── Fetch user's analytics ────────────────────────────────────────────────
    analytics_data = await get_user_analytics(db=db, clerk_id=clerk_id)

    # ── Handle first-time users with no quiz history ──────────────────────────
    if analytics_data["total_sessions"] == 0:
        return AIFeedbackResponse(
            weak_areas=[],
            strong_areas=[],
            suggestions=[
                "Complete your first quiz to get personalised AI feedback!",
                "Start with a subject you feel confident in to build momentum.",
            ],
            motivational_note=(
                "Welcome! Your learning journey starts with a single quiz. "
                "Take your first one to unlock personalised insights."
            ),
            generated_at=datetime.now(timezone.utc),
        )

    # ── Build the summary for the Groq prompt ─────────────────────────────────
    analytics_summary = {
        "overall_accuracy": analytics_data["overall_accuracy"],
        "overall_avg_response_time": analytics_data["overall_avg_response_time"],
        "total_sessions_completed": analytics_data["total_sessions"],
        "subjects": analytics_data["subjects"],
        "strong_subjects": analytics_data["strong_subjects"],
        "weak_subjects": analytics_data["weak_subjects"],
    }

    # ── Call Groq for AI feedback ──────────────────────────────────────────────
    feedback_data = await generate_feedback(analytics_summary)

    return AIFeedbackResponse(
        weak_areas=feedback_data.get("weak_areas", []),
        strong_areas=feedback_data.get("strong_areas", []),
        suggestions=feedback_data.get("suggestions", []),
        motivational_note=feedback_data.get("motivational_note", ""),
        generated_at=datetime.now(timezone.utc),
    )
