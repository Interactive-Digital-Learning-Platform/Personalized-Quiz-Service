"""
models/analytics.py
────────────────────
SQLAlchemy ORM model for the `analytics` table.

One row per (user, subject) pair — this is an upsert-style aggregate table.
Every time a user submits a quiz on a subject, we recalculate and update their
row rather than inserting a new one, keeping the table compact and queries fast.

`weak_topic` stores the single worst-performing topic for that subject as a
simple string (sufficient for MVP; can be expanded to JSONB array later).
"""
from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Analytics(Base):
    __tablename__ = "analytics"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Foreign Key to User ───────────────────────────────────────────────────
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Dimension ─────────────────────────────────────────────────────────────
    # Tracked per subject so we can report "weak in Maths but strong in Science"
    subject: Mapped[str] = mapped_column(String(100), nullable=False)

    # ── Aggregated Metrics ────────────────────────────────────────────────────
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_response_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # The topic within this subject where the user scores lowest
    weak_topic: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),   # Automatically updated on every UPDATE statement
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="analytics")  # noqa: F821
