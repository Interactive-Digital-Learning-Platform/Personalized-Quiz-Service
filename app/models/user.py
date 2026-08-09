from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    clerk_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)

    username: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    quiz_sessions: Mapped[list["QuizSession"]] = relationship(  # noqa: F821
        "QuizSession", back_populates="user", cascade="all, delete-orphan"
    )
    analytics: Mapped[list["Analytics"]] = relationship(  # noqa: F821
        "Analytics", back_populates="user", cascade="all, delete-orphan"
    )
    lesson_masteries: Mapped[list["LessonMastery"]] = relationship(  # noqa: F821
        "LessonMastery", back_populates="user", cascade="all, delete-orphan"
    )
    subject_masteries: Mapped[list["SubjectMastery"]] = relationship(  # noqa: F821
        "SubjectMastery", back_populates="user", cascade="all, delete-orphan"
    )
