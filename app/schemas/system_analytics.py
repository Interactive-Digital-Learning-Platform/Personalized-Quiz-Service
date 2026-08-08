"""
schemas/system_analytics.py
─────────────────────────────
Pydantic v2 schemas for INTERNAL, system-wide technical analytics —
distinct from the per-user, learning-focused schemas in analytics.py.
Currently just the Groq quiz-generation telemetry (see
app/models/ai_generation_event.py and app/services/telemetry_service.py),
served only via the admin-only GET /analytics/system/ai-generation endpoint.
"""
from datetime import datetime
from pydantic import BaseModel


class AIGenerationAnalyticsResponse(BaseModel):
    """
    Aggregate telemetry across ALL users' POST /quiz/generate requests,
    optionally scoped to a date range (see start_date/end_date below).
    """
    total_generation_requests: int
    successful_ai_generations: int
    failed_ai_generations: int
    cache_fallback_count: int
    generation_retry_count: int
    duplicate_question_count: int
    invalid_question_count: int
    average_generation_latency_ms: float
    # Computed via PostgreSQL's percentile_cont() against the real DB — see
    # telemetry_service.get_ai_generation_analytics().
    p95_generation_latency_ms: float
    average_questions_generated: float
    ai_success_rate: float      # Percentage (0.0 - 100.0)
    cache_fallback_rate: float  # Percentage (0.0 - 100.0)

    # Echoes the date-range filter actually applied (None means unbounded on
    # that side), so callers can confirm what window this response covers.
    start_date: datetime | None
    end_date: datetime | None
