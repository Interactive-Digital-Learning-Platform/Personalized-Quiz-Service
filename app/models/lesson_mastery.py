"""
models/lesson_mastery.py
─────────────────────────
SQLAlchemy ORM model for the `lesson_mastery` table.

One row per (user, subject, lesson) triple. This is the persistent state that
drives adaptive difficulty: instead of the client choosing "easy/medium/hard",
`difficulty_service.py` reads/writes this table to decide what difficulty the
next quiz for that subject+lesson should use, based on the user's accuracy
history over time.
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class LessonMastery(Base):
    __tablename__ = "lesson_mastery"
    __table_args__ = (
        UniqueConstraint("user_id", "subject", "lesson", name="uq_lesson_mastery_user_subject_lesson"),
    )

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Foreign Key to User ───────────────────────────────────────────────────
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Dimension ─────────────────────────────────────────────────────────────
    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), index=True, nullable=False)

    # ── Adaptive State ────────────────────────────────────────────────────────
    # The difficulty the NEXT quiz for this (subject, lesson) should be generated at.
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False, default="easy")

    # Accuracy from the most recent completed quiz that touched this lesson.
    last_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Consecutive strong/weak sessions — used to decide when to bump the
    # difficulty up or down (see difficulty_service.py for thresholds).
    consecutive_strong: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consecutive_weak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Timestamps ────────────────────────────────────────────────────────────
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="lesson_masteries")  # noqa: F821
