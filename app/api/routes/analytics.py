import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, require_admin_or_dev
from app.schemas.analytics import AIFeedbackResponse, UserAnalyticsResponse
from app.schemas.system_analytics import AIGenerationAnalyticsResponse
from app.services.analytics_service import get_user_analytics
from app.services.groq_service import generate_feedback
from app.services.telemetry_service import get_ai_generation_analytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/me",
    response_model=UserAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get my analytics",
    description=(
        "Returns the authenticated user's full performance profile: overall "
        "accuracy (weighted across every answered question, not averaged "
        "per-subject), average response time, correct/incorrect/unanswered "
        "question totals, per-subject breakdown, and identified strong/weak "
        "subjects."
    ),
)
async def get_my_analytics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserAnalyticsResponse:
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /analytics/me — clerk_id=%s", clerk_id)

    analytics_data = await get_user_analytics(db=db, clerk_id=clerk_id)

    return UserAnalyticsResponse(**analytics_data)


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
    clerk_id: str = current_user.get("sub", "")
    if not clerk_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the 'sub' (user ID) claim.",
        )

    logger.info("GET /analytics/feedback — clerk_id=%s", clerk_id)

    analytics_data = await get_user_analytics(db=db, clerk_id=clerk_id)

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

    # Trimmed down on purpose — the full subjects[] payload carries every
    # topic's mastery/difficulty/response-time breakdown, which got big
    # enough on multi-subject accounts to blow past Groq's tokens-per-minute
    # limit. `recommendations` below already has the detailed, actionable
    # stuff, so the AI doesn't need the raw data too.
    lean_subjects = [
        {
            "subject": s["subject"],
            "accuracy": s["accuracy"],
            "weak_topic": s["weak_topic"],
            "current_difficulty": s["current_difficulty"],
            "trend": s["performance_trend"]["trend"],
        }
        for s in analytics_data["subjects"]
    ]
    analytics_summary = {
        "overall_accuracy": analytics_data["overall_accuracy"],
        "overall_avg_response_time": analytics_data["overall_avg_response_time"],
        "total_sessions_completed": analytics_data["total_sessions"],
        "subjects": lean_subjects,
        "strong_subjects": analytics_data["strong_subjects"],
        "weak_subjects": analytics_data["weak_subjects"],
        "recommendations": analytics_data["recommendations"],
    }

    feedback_data = await generate_feedback(analytics_summary)

    return AIFeedbackResponse(
        weak_areas=feedback_data.get("weak_areas", []),
        strong_areas=feedback_data.get("strong_areas", []),
        suggestions=feedback_data.get("suggestions", []),
        motivational_note=feedback_data.get("motivational_note", ""),
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/system/ai-generation",
    response_model=AIGenerationAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="[Admin] Groq quiz-generation telemetry",
    description=(
        "Internal, system-wide technical analytics for the Groq quiz-generation "
        "pipeline — success/failure/cache-fallback rates, retries, duplicate/invalid "
        "question counts, and generation latency (including p95). Restricted to "
        "development environments or listed admin users (see ADMIN_CLERK_IDS). "
        "This is NOT part of, and never appears in, GET /analytics/me."
    ),
)
async def get_ai_generation_telemetry(
    start_date: datetime | None = Query(
        default=None, description="Only include events at/after this UTC timestamp."
    ),
    end_date: datetime | None = Query(
        default=None, description="Only include events at/before this UTC timestamp."
    ),
    _admin: dict = Depends(require_admin_or_dev),
    db: AsyncSession = Depends(get_db),
) -> AIGenerationAnalyticsResponse:
    logger.info(
        "GET /analytics/system/ai-generation — start_date=%s, end_date=%s", start_date, end_date,
    )
    data = await get_ai_generation_analytics(db=db, start_date=start_date, end_date=end_date)
    return AIGenerationAnalyticsResponse(**data)
