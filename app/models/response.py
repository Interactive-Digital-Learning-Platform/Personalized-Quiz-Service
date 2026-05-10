from sqlalchemy import Column, Integer, ForeignKey, String, Float, Boolean, DateTime, func
from app.database.session import Base


class UserResponse(Base):
    __tablename__ = "user_responses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    quiz_id = Column(Integer, ForeignKey("quiz_sessions.id"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    chosen_answer = Column(String, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    response_time = Column(Float, nullable=True)
    answered_at = Column(DateTime(timezone=True), server_default=func.now())
