import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import rate_limit
from app.core.security import ClerkId, require_admin_or_dev
from app.schemas.analytics import (
    AIFeedbackResponse,
    DailyAnalyticsResponse,
    UserAnalyticsResponse,
)
from app.schemas.system_analytics import AIGenerationAnalyticsResponse
from app.services.analytics.daily_service import get_daily_analytics
from app.services.analytics_service import get_user_analytics
from app.services.groq_service import generate_feedback
from app.services.telemetry_service import get_ai_generation_analytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# 14 lets the profile compare this week against the week before.
DAILY_WINDOW_DAYS = {7, 14, 30}

# clerk_id -> (summary hash, expires at [monotonic], response). The feedback only
# changes when the analytics behind it do, so reopening the profile shouldn't
# cost a Groq call each time. Per process, like the rate limiter.
_feedback_cache: dict[str, tuple[str, float, AIFeedbackResponse]] = {}


def _summary_hash(summary: dict) -> str:
    return hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()


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
    clerk_id: ClerkId,
    db: AsyncSession = Depends(get_db),
) -> UserAnalyticsResponse:
    logger.info("GET /analytics/me — clerk_id=%s", clerk_id)

    analytics_data = await get_user_analytics(db=db, clerk_id=clerk_id)

    return UserAnalyticsResponse(**analytics_data)


@router.get(
    "/daily",
    response_model=DailyAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get my day-by-day activity",
    description=(
        "Per-day quizzes completed, questions answered, accuracy, study time and "
        "average time per question for the last `days` days (7, 14 or 30), bucketed by "
        "the student's local calendar day in `tz` (IANA name, e.g. Asia/Colombo). "
        "Every day in the window is returned, including days with no activity."
    ),
)
async def get_my_daily_analytics(
    clerk_id: ClerkId,
    days: int = Query(7, description="Window length: 7, 14 or 30 days."),
    tz: str = Query("UTC", description="IANA timezone name used for day boundaries."),
    db: AsyncSession = Depends(get_db),
) -> DailyAnalyticsResponse:
    if days not in DAILY_WINDOW_DAYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"days must be one of {sorted(DAILY_WINDOW_DAYS)}.",
        )
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown timezone {tz!r}; expected an IANA name such as 'Asia/Colombo'.",
        )

    logger.info("GET /analytics/daily — clerk_id=%s days=%d tz=%s", clerk_id, days, tz)

    daily_data = await get_daily_analytics(db, clerk_id, days=days, tz=zone)
    return DailyAnalyticsResponse(**daily_data)


@router.get(
    "/feedback",
    response_model=AIFeedbackResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit("ai_feedback", "RATE_LIMIT_AI_FEEDBACK"))],
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
    dependencies=[Depends(rate_limit("ai_feedback", "RATE_LIMIT_AI_FEEDBACK"))],
    summary="Get AI feedback",
    description=(
        "Pulls the user's latest analytics from the DB, feeds it to the Groq AI, "
        "and returns personalised study suggestions, weak areas, and an encouraging note."
    ),
)
async def get_ai_feedback(
    clerk_id: ClerkId,
    db: AsyncSession = Depends(get_db),
) -> AIFeedbackResponse:
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
            generated_at=datetime.now(UTC),
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

    summary_hash = _summary_hash(analytics_summary)
    cached = _feedback_cache.get(clerk_id)
    if cached and cached[0] == summary_hash and cached[1] > time.monotonic():
        return cached[2]

    feedback_data = await generate_feedback(analytics_summary)

    response = AIFeedbackResponse(
        weak_areas=feedback_data.get("weak_areas", []),
        strong_areas=feedback_data.get("strong_areas", []),
        suggestions=feedback_data.get("suggestions", []),
        motivational_note=feedback_data.get("motivational_note", ""),
        generated_at=datetime.now(UTC),
    )
    # Don't pin the generic fallback (Groq returned non-JSON) for hours -- only real feedback.
    if response.weak_areas or response.strong_areas:
        _feedback_cache[clerk_id] = (
            summary_hash, time.monotonic() + settings.AI_FEEDBACK_CACHE_TTL_SECONDS, response,
        )
    return response


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
