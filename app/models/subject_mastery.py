"""
models/subject_mastery.py
──────────────────────────
SQLAlchemy ORM model for the `subject_mastery` table.

One row per (user, subject) pair. This is the persistent state that drives
adaptive difficulty for the default random-lesson quiz flow: since each quiz
question is independently assigned a different random lesson, there's no
single lesson to look up a difficulty for ahead of generation. Instead, this
table tracks the student's overall performance in each SUBJECT directly, so
e.g. a student doing well in Maths gets gradually harder Maths quizzes over
time, while a student struggling in Science stays at "easy" until they
improve — independently per subject, per student.

`LessonMastery` still exists separately for the narrower case where a caller
explicitly requests one specific lesson (manual override / focused practice).
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SubjectMastery(Base):
    __tablename__ = "subject_mastery"
    __table_args__ = (
        UniqueConstraint("user_id", "subject", name="uq_subject_mastery_user_subject"),
    )

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Foreign Key to User ───────────────────────────────────────────────────
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Dimension ─────────────────────────────────────────────────────────────
    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)

    # ── Adaptive State ────────────────────────────────────────────────────────
    # The difficulty the NEXT random-lesson quiz for this subject should use.
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False, default="easy")

    # Overall accuracy from the most recent completed quiz in this subject.
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
    user: Mapped["User"] = relationship("User", back_populates="subject_masteries")  # noqa: F821
