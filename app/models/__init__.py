"""
models/__init__.py
──────────────────
Import all models here so SQLAlchemy's metadata knows about them when
`Base.metadata.create_all()` is called at startup.

This is the single import point — any new model added to the project must also
be imported here.
"""
from app.models.user import User  # noqa: F401
from app.models.question import Question  # noqa: F401
from app.models.quiz_session import QuizSession, QuestionAttempt  # noqa: F401
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot  # noqa: F401
from app.models.analytics import Analytics  # noqa: F401
from app.models.lesson_mastery import LessonMastery  # noqa: F401
from app.models.subject_mastery import SubjectMastery  # noqa: F401
from app.models.ai_generation_event import AIGenerationEvent  # noqa: F401

__all__ = [
	"User",
	"Question",
	"QuizSession",
	"QuestionAttempt",
	"QuizProgressSnapshot",
	"QuizCompletion",
	"Analytics",
	"LessonMastery",
	"SubjectMastery",
	"AIGenerationEvent",
]
