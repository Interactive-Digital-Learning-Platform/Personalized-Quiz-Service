from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class SkillBKTState(Base):
    # Bayesian Knowledge Tracing state per (user, subject, lesson, grade)
    # skill — a read-only probabilistic P(know) signal, independent of
    # LessonMastery's Continuous Evidence-Weighted System (which drives
    # difficulty selection). See app/services/bkt_service.py.
    __tablename__ = "skill_bkt_state"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "subject", "lesson", "grade", name="uq_skill_bkt_state_user_subject_lesson_grade"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    # Null for legacy attempts pre-dating the curriculum taxonomy — same
    # precedent as LessonMastery.grade.
    grade: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)

    # Default mirrors settings.BKT_P_INIT (can't reference settings directly
    # in a column default) — keep the two in sync if that setting changes.
    p_know: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    opportunities: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="skill_bkt_states")
