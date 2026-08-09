from typing import Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class GenerateQuizRequest(BaseModel):
    grade: int = Field(10, ge=1, le=13, description="School grade level (1–13)")
    subject: str = Field(..., min_length=1, max_length=100)
    lesson: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Optional. If omitted (the default), each question is independently "
            "assigned a different, randomly varied lesson within the subject by "
            "the AI — the quiz is NOT pinned to one topic. Set this to force "
            "every question onto one specific lesson instead."
        ),
    )
    difficulty: str | None = Field(
        default=None,
        description=(
            "easy | medium | hard — optional. If omitted, the server picks the "
            "difficulty automatically based on the user's accuracy history for "
            "this subject/lesson (see difficulty_service.py)."
        ),
    )
    question_count: int = Field(..., ge=1, le=30, description="Number of questions to generate")
    excluded_question_ids: list[int] = Field(
        default_factory=list,
        description="Question IDs the user has already seen — excluded from the cache sample to ensure variety",
    )
    force_cache: bool = Field(
        default=False,
        description="Skip AI generation and serve directly from the DB question pool (user-initiated fallback after AI failure)",
    )

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str | None) -> str | None:
        if v is None:
            return None
        allowed = {"easy", "medium", "hard"}
        if v.lower() not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}")
        return v.lower()


class QuestionOut(BaseModel):
    id: int
    question: str
    options: list[Any] | None = None
    subject: str
    lesson: str
    difficulty: str
    correct_answer: str | None = None

    model_config = {"from_attributes": True}


class GenerateQuizResponse(BaseModel):
    session_id: int
    questions: list[QuestionOut]
    cache_hit: bool = False
    difficulty: str
    lesson: str


class SavedQuizResponse(BaseModel):
    session_id: int
    subject: str
    lesson: str
    difficulty: str
    question_count: int
    generated_at: datetime
    questions: list[QuestionOut]
    latest_progress: dict | None = None
    completion: dict | None = None


class QuizSessionSummary(BaseModel):
    session_id: int
    subject: str
    difficulty: str
    question_count: int
    created_at: datetime
    answered_count: int
    is_completed: bool
    accuracy: float | None = None
    correct_count: int | None = None
    question_ids: list[int] = Field(
        default_factory=list,
        description="IDs of questions served in this session — used by the client to request fresh questions",
    )


class AnswerItem(BaseModel):
    question_id: int
    selected_answer: str
    response_time: float = Field(..., ge=0.0)
    is_repeated: bool = False


class ProgressAnswerDraft(BaseModel):
    question_id: int
    selected_answer: str | None = None
    response_time: float | None = Field(default=None, ge=0.0)


class SaveProgressRequest(BaseModel):
    session_id: int
    remaining_time: float | None = Field(default=None, ge=0.0)
    answered_count: int = Field(default=0, ge=0)
    repeated_question_ids: list[int] = Field(default_factory=list)
    weak_lessons_hint: list[str] = Field(default_factory=list)
    draft_answers: list[ProgressAnswerDraft] = Field(default_factory=list)


class SaveProgressResponse(BaseModel):
    session_id: int
    saved_at: datetime
    remaining_time: float | None
    answered_count: int
    repeated_question_ids: list[int]
    weak_lessons_hint: list[str]


class SubmitQuizRequest(BaseModel):
    session_id: int
    answers: list[AnswerItem] = Field(..., min_length=1)
    ended_by: Literal["submitted", "timeout"] = "submitted"
    remaining_time_at_end: float | None = Field(default=None, ge=0.0)
    repeated_question_ids: list[int] = Field(default_factory=list)


class SubmitQuizResponse(BaseModel):
    session_id: int
    score: float
    accuracy: float
    total_time: float
    avg_response_time: float
    correct_count: int
    total_questions: int
    ended_by: str
    lesson_time_breakdown: dict[str, float]
    lesson_accuracy_breakdown: dict[str, dict[str, float | int]]
    repeated_correct_count: int
    repeated_wrong_count: int
    repeated_lessons_right: list[str]
    repeated_lessons_wrong: list[str]
