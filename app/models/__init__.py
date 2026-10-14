# Every model needs to be imported here, or SQLAlchemy won't know it exists
# when create_all() runs at startup — add new models to this list too.
from app.models.ai_generation_event import AIGenerationEvent
from app.models.analytics import Analytics
from app.models.lesson_mastery import LessonMastery
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.skill_bkt_state import SkillBKTState
from app.models.subject_mastery import SubjectMastery
from app.models.user import User

__all__ = [
	"AIGenerationEvent",
	"Analytics",
	"LessonMastery",
	"Question",
	"QuestionAttempt",
	"QuizCompletion",
	"QuizProgressSnapshot",
	"QuizSession",
	"SkillBKTState",
	"SubjectMastery",
	"User",
]
