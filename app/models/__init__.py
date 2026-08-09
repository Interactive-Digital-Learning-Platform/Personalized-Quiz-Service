# Every model needs to be imported here, or SQLAlchemy won't know it exists
# when create_all() runs at startup — add new models to this list too.
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
