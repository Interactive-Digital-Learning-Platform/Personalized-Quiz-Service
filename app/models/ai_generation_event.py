from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AIGenerationEvent(Base):
    __tablename__ = "ai_generation_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="SET NULL"), nullable=True
    )

    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    requested_question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="groq")
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    used_cache_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # One of: timeout | provider_error | invalid_json | validation_error |
    # database_error | unknown. Never the raw exception message — this table
    # is safe to expose to admins without leaking internals.
    error_category: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
