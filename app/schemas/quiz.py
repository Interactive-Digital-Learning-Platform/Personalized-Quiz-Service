"""
schemas/quiz.py
───────────────
Pydantic v2 schemas for quiz generation and submission endpoints.

Schema design decisions:
- `GenerateQuizRequest` uses literal types for difficulty to enforce valid values.
- `QuestionOut` uses `model_config from_attributes` so it can be built from ORM objects.
- `SubmitQuizRequest` contains a nested list of per-question answers + timings.
- `SubmitQuizResponse` is returned immediately after submission with computed metrics.
"""
from typing import Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ── Quiz Generation ────────────────────────────────────────────────────────────

class GenerateQuizRequest(BaseModel):
    """Input payload for POST /quiz/generate"""
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
    """
    A single question returned to the frontend.
    `correct_answer` is included so the frontend can compare submitted answers.
    """
    id: int
    question: str
    options: list[Any] | None = None   # List of answer choices
    subject: str
    lesson: str
    difficulty: str
    correct_answer: str | None = None

    model_config = {"from_attributes": True}


class GenerateQuizResponse(BaseModel):
    """Response payload for POST /quiz/generate"""
    session_id: int
    questions: list[QuestionOut]
    # Indicates whether these questions came from DB cache or fresh AI generation
    cache_hit: bool = False
    # The difficulty actually used — chosen automatically from the user's
    # accuracy history unless the caller explicitly overrode it.
    difficulty: str
    # Session-level lesson label. Normally "Mixed" — by default each question
    # is independently assigned a different, random lesson within the subject
    # (see each QuestionOut's own `lesson` for the per-question value). Only a
    # single lesson name if the caller explicitly overrode it, or if every
    # served question happened to land on the same lesson.
    lesson: str


class SavedQuizResponse(BaseModel):
    """Response payload for retrieving a previously generated quiz session."""
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
    """Lightweight session summary for the practice list."""
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


# ── Quiz Submission ────────────────────────────────────────────────────────────

class AnswerItem(BaseModel):
    """One answer entry — one per question in the quiz."""
    question_id: int
    selected_answer: str
    # Time spent on this question in seconds (measured by the frontend)
    response_time: float = Field(..., ge=0.0)
    is_repeated: bool = False


class ProgressAnswerDraft(BaseModel):
    question_id: int
    selected_answer: str | None = None
    response_time: float | None = Field(default=None, ge=0.0)


class SaveProgressRequest(BaseModel):
    """Input payload for POST /quiz/progress/save"""

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
    """Input payload for POST /quiz/submit"""
    session_id: int
    answers: list[AnswerItem] = Field(..., min_length=1)
    ended_by: Literal["submitted", "timeout"] = "submitted"
    remaining_time_at_end: float | None = Field(default=None, ge=0.0)
    repeated_question_ids: list[int] = Field(default_factory=list)


class SubmitQuizResponse(BaseModel):
    """Computed results returned immediately after submission."""
    session_id: int
    score: float           # Number of correct answers
    accuracy: float        # Percentage correct (0.0 – 100.0)
    total_time: float      # Sum of all response_times in seconds
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
