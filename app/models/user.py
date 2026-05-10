"""
models/user.py
──────────────
SQLAlchemy ORM model for the `users` table.

A User is created the first time a Clerk-authenticated user hits our API.
We store their Clerk ID as the canonical identifier — we never rely on our
internal `id` for authentication, only for DB foreign-key relationships.
"""
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    # ── Primary Key ───────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # ── Clerk Identity ────────────────────────────────────────────────────────
    # The `sub` claim from the Clerk JWT (e.g. "user_2abc...")
    # Unique constraint ensures one DB record per Clerk account.
    clerk_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    # ── Profile ───────────────────────────────────────────────────────────────
    username: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    # server_default=func.now() means the DB sets this, not Python —
    # so it uses the DB server's clock consistently.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    # Lazy="dynamic" is replaced in SQLAlchemy 2.x with `lazy="select"` (default)
    quiz_sessions: Mapped[list["QuizSession"]] = relationship(  # noqa: F821
        "QuizSession", back_populates="user", cascade="all, delete-orphan"
    )
    analytics: Mapped[list["Analytics"]] = relationship(  # noqa: F821
        "Analytics", back_populates="user", cascade="all, delete-orphan"
    )
