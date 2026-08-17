from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.quiz_session import QuizSession


class QuizProgressSnapshot(Base):
    __tablename__ = "quiz_progress_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    remaining_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    answered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repeated_question_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    weak_lessons_hint: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    draft_answers: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)

    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[QuizSession] = relationship("QuizSession", back_populates="progress_snapshots")


class QuizCompletion(Base):
    __tablename__ = "quiz_completions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    ended_by: Mapped[str] = mapped_column(String(32), nullable=False)
    total_time: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)

    lesson_time_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    lesson_accuracy_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    repeated_lessons_right: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    repeated_lessons_wrong: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    repeated_correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repeated_wrong_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[QuizSession] = relationship("QuizSession", back_populates="completion")
