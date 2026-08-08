"""
models/ai_generation_event.py
────────────────────────────────
SQLAlchemy ORM model for the `ai_generation_events` table.

One row per POST /quiz/generate request — recorded regardless of whether
Groq succeeded, failed, or was skipped in favor of the DB cache (see
app/services/telemetry_service.py and quiz_service.generate_quiz()). This is
INTERNAL, technical telemetry for the generation pipeline itself, entirely
separate from the user-facing learning analytics in models/analytics.py —
never exposed via GET /analytics/me, only via the admin-only
GET /analytics/system/ai-generation endpoint.

Deliberately stores nothing sensitive: no API keys, JWTs, prompts, model
responses, or raw exception messages — only categorical/numeric data (see
`error_category`, which is always one of a small fixed set of safe labels).
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AIGenerationEvent(Base):
    __tablename__ = "ai_generation_events"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Who / what session (nullable: telemetry must survive even if we can't
    # attribute it to a session — e.g. generation failed before one was
    # created — or, in principle, to a user) ───────────────────────────────────
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="SET NULL"), nullable=True
    )

    # ── Request parameters ───────────────────────────────────────────────────
    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    requested_question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Provider / model ──────────────────────────────────────────────────────
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="groq")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # ── Outcome ────────────────────────────────────────────────────────────────
    # Whether the AI actually produced usable questions this request —
    # independent of used_cache_fallback (a deliberate force_cache request
    # never even attempts AI, so both are meaningfully distinct signals; see
    # telemetry_service.py for the exact semantics).
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    used_cache_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # One of: "timeout" | "provider_error" | "invalid_json" | "validation_error"
    # | "database_error" | "unknown" — NEVER the raw exception message. NULL
    # when the request had no error (success, or an intentional cache-only
    # request that never attempted AI at all).
    error_category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
