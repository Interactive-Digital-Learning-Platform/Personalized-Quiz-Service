"""
models/question.py
──────────────────
SQLAlchemy ORM model for the `questions` table.

Questions are stored globally (not tied to a single session) so they can be
reused across multiple quiz sessions — this is the core of the DB-first
caching strategy for `/quiz/generate`.

The `options` column uses SQLAlchemy's generic JSON type so the model works
with both PostgreSQL and the local SQLite development fallback.
"""
from sqlalchemy import event, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.services.question_fingerprint import compute_question_fingerprint


class Question(Base):
    __tablename__ = "questions"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Content ───────────────────────────────────────────────────────────────
    question: Mapped[str] = mapped_column(String, nullable=False)

    # `options` stores the list of answer choices as a JSON array, e.g.:
    #   ["Option A", "Option B", "Option C", "Option D"]
    options: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)

    correct_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Metadata (used for caching/filtering & analytics) ─────────────────────
    # These fields allow us to look up existing questions before calling Groq.
    subject: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    lesson: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), index=True, nullable=False)

    # Deterministic sha256 hash of (normalized subject, lesson, question text)
    # — see services/question_fingerprint.py. Identifies "the same or an
    # equivalent question" across separate rows, since every AI generation
    # call creates fresh rows rather than reusing an existing one. Indexed
    # for efficient GROUP BY in the repeated-question analytics query.
    # Auto-populated by the before_insert listener below if not set
    # explicitly, so no calling code needs to remember to compute it.
    question_fingerprint: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    # ── Relationships ─────────────────────────────────────────────────────────
    # A question can appear in many attempts across many sessions
    attempts: Mapped[list["QuestionAttempt"]] = relationship(  # noqa: F821
        "QuestionAttempt", back_populates="question", cascade="all, delete-orphan"
    )


@event.listens_for(Question, "before_insert")
def _populate_question_fingerprint(mapper, connection, target: "Question") -> None:
    if not target.question_fingerprint:
        target.question_fingerprint = compute_question_fingerprint(
            target.subject, target.lesson, target.question
        )
