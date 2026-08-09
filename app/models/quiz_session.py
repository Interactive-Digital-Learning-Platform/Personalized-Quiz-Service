from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.question import Question
    from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
    from app.models.user import User


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    questions_snapshot: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # "Deleting" a session from the app just sets this — the row sticks
    # around because analytics and future quiz generation both still read
    # from session history. Only the user-facing list/detail endpoints
    # filter it out.
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    user: Mapped["User"] = relationship("User", back_populates="quiz_sessions")  # noqa: F821
    attempts: Mapped[list["QuestionAttempt"]] = relationship(
        "QuestionAttempt", back_populates="session", cascade="all, delete-orphan"
    )
    progress_snapshots: Mapped[list["QuizProgressSnapshot"]] = relationship(
        "QuizProgressSnapshot", back_populates="session", cascade="all, delete-orphan"
    )
    completion: Mapped["QuizCompletion | None"] = relationship(
        "QuizCompletion", back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class QuestionAttempt(Base):
    __tablename__ = "question_attempts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), index=True, nullable=False
    )

    selected_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    response_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    session: Mapped["QuizSession"] = relationship("QuizSession", back_populates="attempts")
    question: Mapped["Question"] = relationship("Question", back_populates="attempts")  # noqa: F821
