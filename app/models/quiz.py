from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, func
from sqlalchemy.orm import relationship
from app.database.session import Base


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    grade = Column(String, nullable=False)
    subject = Column(String, nullable=False)
    lesson = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    count = Column(Integer, nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    meta = Column(JSON, nullable=True)

    questions = relationship("Question", back_populates="quiz", cascade="all, delete-orphan")
