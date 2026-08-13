from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SubjectMastery(Base):
    __tablename__ = "subject_mastery"
    __table_args__ = (
        UniqueConstraint("user_id", "subject", name="uq_subject_mastery_user_subject"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)

    # Difficulty the next random-lesson quiz in this subject should use.
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False, default="easy")
    last_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    consecutive_strong: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consecutive_weak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # See difficulty_mastery_engine.py for the formulas. Defaults match a brand-new student.
    mastery_score: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    fluency_score: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recent_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    previous_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    trend_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    trend_label: Mapped[str] = mapped_column(String(32), nullable=False, default="insufficient_data")
    retention_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    last_mastery_update: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="subject_masteries")  # noqa: F821
