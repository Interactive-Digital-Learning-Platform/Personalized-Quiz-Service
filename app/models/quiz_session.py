"""
models/quiz_session.py
──────────────────────
SQLAlchemy ORM models for `quiz_sessions` and `question_attempts`.

A QuizSession records a user starting a quiz (subject, lesson, difficulty).
A QuestionAttempt records the user's answer to a single question within a session.

Separating attempts from sessions lets us do per-question analytics easily.
"""
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
    """
    One row = one quiz attempt by a user.
    Scores are stored after submission (`POST /quiz/submit`).
    """
    __tablename__ = "quiz_sessions"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Foreign Key to User ───────────────────────────────────────────────────
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Quiz Metadata ─────────────────────────────────────────────────────────
    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    questions_snapshot: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)

    # ── Results (populated on submit) ─────────────────────────────────────────
    score: Mapped[float | None] = mapped_column(Float, nullable=True)        # e.g. 7 (out of 10)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)     # e.g. 70.0 (percent)
    total_time: Mapped[float | None] = mapped_column(Float, nullable=True)   # Total seconds taken

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
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
    """
    One row = one question answered within a quiz session.
    Stores what the user selected, whether it was correct, and how long they took.
    """
    __tablename__ = "question_attempts"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Foreign Keys ─────────────────────────────────────────────────────────
    session_id: Mapped[int] = mapped_column(
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # ── Answer Data ───────────────────────────────────────────────────────────
    selected_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # ── Timing ────────────────────────────────────────────────────────────────
    # Seconds the user spent on this specific question (sent by the frontend)
    response_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    session: Mapped["QuizSession"] = relationship("QuizSession", back_populates="attempts")
    question: Mapped["Question"] = relationship("Question", back_populates="attempts")  # noqa: F821
