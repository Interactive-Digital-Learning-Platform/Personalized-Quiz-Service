from datetime import datetime

from pydantic import BaseModel


class AIGenerationAnalyticsResponse(BaseModel):
    total_generation_requests: int
    successful_ai_generations: int
    failed_ai_generations: int
    cache_fallback_count: int
    generation_retry_count: int
    duplicate_question_count: int
    invalid_question_count: int
    average_generation_latency_ms: float
    p95_generation_latency_ms: float
    average_questions_generated: float
    ai_success_rate: float
    cache_fallback_rate: float

    start_date: datetime | None
    end_date: datetime | None
