from typing import TYPE_CHECKING

from sqlalchemy import JSON, Index, Integer, String, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.services.question_fingerprint import compute_question_fingerprint

if TYPE_CHECKING:
    from app.models.quiz_session import QuestionAttempt


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        # The question pool is read by (subject, difficulty) on every quiz
        # generation request (see quiz_service._fetch_pool_questions) — the
        # single-column indexes below cover other lookups but not this
        # combined filter efficiently.
        Index("ix_questions_subject_difficulty", "subject", "difficulty"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    question: Mapped[str] = mapped_column(String, nullable=False)
    options: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    correct_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)

    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), index=True, nullable=False)

    # Null for legacy rows generated before the curriculum taxonomy existed.
    # Scopes lesson identity alongside (subject, lesson) so same-named lessons
    # in different grades (e.g. "Percentages" in both Grade 10 and 11 Math)
    # aren't conflated.
    grade: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)

    # Hash of (subject, lesson, question text) that identifies "the same or an
    # equivalent question" across rows — every AI generation creates fresh
    # rows rather than reusing one, so this is how we later spot repeats.
    # Filled in automatically by the listener below if not set explicitly.
    question_fingerprint: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    attempts: Mapped[list["QuestionAttempt"]] = relationship(
        "QuestionAttempt", back_populates="question", cascade="all, delete-orphan"
    )


@event.listens_for(Question, "before_insert")
def _populate_question_fingerprint(mapper, connection, target: "Question") -> None:
    if not target.question_fingerprint:
        target.question_fingerprint = compute_question_fingerprint(
            target.subject, target.lesson, target.question
        )
