"""
services/telemetry_service.py
────────────────────────────────
Internal, technical telemetry for the Groq quiz-generation pipeline —
completely separate from the user-facing learning analytics in
analytics_service.py. One AIGenerationEvent row is recorded per
POST /quiz/generate request (see quiz_service.generate_quiz()), regardless
of whether generation succeeded, failed, or fell back to the DB cache.
Read back via the admin-only GET /analytics/system/ai-generation endpoint —
never exposed through GET /analytics/me.

Never stores API keys, JWTs, prompts, model responses, or raw exception
messages — only the safe, fixed-vocabulary `error_category` (see
ERROR_CATEGORIES) and plain counts/timings.
"""
import logging

from fastapi import HTTPException
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_generation_event import AIGenerationEvent

logger = logging.getLogger(__name__)

ERROR_CATEGORIES = (
    "timeout", "provider_error", "invalid_json", "validation_error", "database_error", "unknown",
)


def categorize_generation_error(exc: BaseException) -> str:
    """
    Maps an exception raised during AI quiz generation to one of
    ERROR_CATEGORIES. This is the ONLY thing derived from the exception that
    ever reaches the database — the exception's own message/detail is used
    here transiently to classify it, then discarded.
    """
    if isinstance(exc, SQLAlchemyError):
        return "database_error"

    if isinstance(exc, HTTPException):
        detail = str(exc.detail).lower()
        if "timeout" in detail or "timed out" in detail:
            return "timeout"
        if "malformed json" in detail:
            return "invalid_json"
        if "empty question list" in detail or "no valid questions" in detail:
            return "validation_error"
        if "ai service error" in detail:
            return "provider_error"
        return "provider_error"

    return "unknown"


async def record_generation_event(
    db: AsyncSession,
    *,
    user_id: int | None,
    session_id: int | None,
    subject: str,
    requested_question_count: int,
    generated_question_count: int,
    provider: str,
    model_name: str,
    success: bool,
    used_cache_fallback: bool,
    retry_count: int,
    duplicate_count: int,
    invalid_question_count: int,
    latency_ms: float,
    error_category: str | None,
) -> None:
    """
    Best-effort telemetry write — deliberately swallows every exception.
    A failure recording telemetry must NEVER break quiz generation itself
    (see quiz_service.generate_quiz(), which calls this from a `finally`
    block after the real response or exception has already been decided).
    """
    try:
        event = AIGenerationEvent(
            user_id=user_id,
            session_id=session_id,
            subject=subject,
            requested_question_count=requested_question_count,
            generated_question_count=generated_question_count,
            provider=provider,
            model_name=model_name,
            success=success,
            used_cache_fallback=used_cache_fallback,
            retry_count=retry_count,
            duplicate_count=duplicate_count,
            invalid_question_count=invalid_question_count,
            latency_ms=latency_ms,
            error_category=error_category,
        )
        db.add(event)
        await db.commit()
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see docstring
        logger.error("Failed to record AI generation telemetry (non-critical): %s", exc)
        try:
            await db.rollback()
        except Exception:
            pass


async def get_ai_generation_analytics(
    db: AsyncSession,
    *,
    start_date=None,
    end_date=None,
) -> dict:
    """
    Aggregate AI-generation telemetry across ALL users — this is a
    system-wide, admin-only view (see routes/analytics.py's
    GET /analytics/system/ai-generation), never scoped to one user like
    GET /analytics/me is.

    p95_generation_latency_ms uses PostgreSQL's percentile_cont() directly
    against Postgres. SQLite (used by this project's test suite) has no
    equivalent built-in function, so the identical statistic — linear-
    interpolation continuous percentile, matching percentile_cont's own
    definition — is computed in Python from the same filtered latency
    values when running on SQLite, keeping the two paths numerically
    consistent even though only one of them is the literal SQL feature
    requested.
    """
    filters = []
    if start_date is not None:
        filters.append(AIGenerationEvent.created_at >= start_date)
    if end_date is not None:
        filters.append(AIGenerationEvent.created_at <= end_date)

    agg_stmt = select(
        func.count(AIGenerationEvent.id).label("total_requests"),
        func.sum(cast(AIGenerationEvent.success, Integer)).label("successful"),
        func.sum(cast(AIGenerationEvent.used_cache_fallback, Integer)).label("cache_fallback"),
        func.sum(AIGenerationEvent.retry_count).label("total_retries"),
        func.sum(AIGenerationEvent.duplicate_count).label("total_duplicates"),
        func.sum(AIGenerationEvent.invalid_question_count).label("total_invalid"),
        func.avg(AIGenerationEvent.latency_ms).label("avg_latency"),
        func.avg(AIGenerationEvent.generated_question_count).label("avg_generated"),
    ).where(*filters)
    row = (await db.execute(agg_stmt)).one()

    # Postgres can return SUM()/CASE-derived aggregates as decimal.Decimal
    # rather than plain int (see analytics_service.py's Decimal notes for
    # the same recurring issue) — explicitly convert rather than relying on
    # `or 0` alone, which only masks the type when the result happens to be zero.
    total_requests = int(row.total_requests or 0)
    successful = int(row.successful or 0)
    cache_fallback = int(row.cache_fallback or 0)
    failed = total_requests - successful

    if total_requests == 0:
        p95 = 0.0
    elif db.bind.dialect.name == "postgresql":
        p95_stmt = select(
            func.percentile_cont(0.95).within_group(AIGenerationEvent.latency_ms.asc())
        ).where(*filters)
        p95_value = (await db.execute(p95_stmt)).scalar_one()
        p95 = round(float(p95_value or 0.0), 2)
    else:
        latencies_stmt = (
            select(AIGenerationEvent.latency_ms)
            .where(*filters)
            .order_by(AIGenerationEvent.latency_ms.asc())
        )
        latencies = [float(v) for v in (await db.execute(latencies_stmt)).scalars().all()]
        p95 = round(_percentile_cont(latencies, 0.95), 2)

    return {
        "total_generation_requests": total_requests,
        "successful_ai_generations": successful,
        "failed_ai_generations": failed,
        "cache_fallback_count": cache_fallback,
        "generation_retry_count": int(row.total_retries or 0),
        "duplicate_question_count": int(row.total_duplicates or 0),
        "invalid_question_count": int(row.total_invalid or 0),
        "average_generation_latency_ms": round(float(row.avg_latency or 0.0), 2),
        "p95_generation_latency_ms": p95,
        "average_questions_generated": round(float(row.avg_generated or 0.0), 2),
        "ai_success_rate": round(successful / total_requests * 100.0, 2) if total_requests > 0 else 0.0,
        "cache_fallback_rate": (
            round(cache_fallback / total_requests * 100.0, 2) if total_requests > 0 else 0.0
        ),
        "start_date": start_date,
        "end_date": end_date,
    }


def _percentile_cont(sorted_values: list[float], pct: float) -> float:
    """
    Pure-Python re-implementation of PostgreSQL's percentile_cont — linear
    interpolation between the two closest ranks — used only as the SQLite
    fallback above so both code paths compute the exact same statistic.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac
