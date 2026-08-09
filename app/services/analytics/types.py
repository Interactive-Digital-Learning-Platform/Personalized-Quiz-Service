from dataclasses import dataclass
from datetime import datetime

from app.models.analytics import Analytics
from app.models.subject_mastery import SubjectMastery


@dataclass(frozen=True, slots=True)
class TopicRow:
    subject: str
    topic: str
    total_attempted: int
    total_correct: int
    last_attempted_at: datetime


@dataclass(frozen=True, slots=True)
class ResponseTimeRow:
    subject: str
    topic: str
    correct: bool
    response_time: float


@dataclass(frozen=True, slots=True)
class TrendAttemptRow:
    session_id: int
    subject: str
    topic: str
    correct: bool


@dataclass(frozen=True, slots=True)
class RepeatedAttemptRow:
    attempt_id: int
    correct: bool
    fingerprint: str
    subject: str
    topic: str
    difficulty: str


@dataclass(frozen=True, slots=True)
class DifficultyAttemptRow:
    subject: str
    difficulty: str
    total_attempted: int
    total_correct: int
    avg_response_time: float


@dataclass(frozen=True, slots=True)
class DifficultySessionRow:
    subject: str
    difficulty: str
    completed_sessions: int


@dataclass(frozen=True, slots=True)
class TopicDifficultyRow:
    subject: str
    topic: str
    difficulty: str
    total_attempted: int
    total_correct: int


@dataclass(frozen=True, slots=True)
class GrowthSessionRow:
    session_id: int
    created_at: datetime
    completion_id: int | None
    last_saved_at: datetime | None


@dataclass(frozen=True, slots=True)
class GrowthAttemptRow:
    session_id: int
    correct: bool
    subject: str
    topic: str
    session_created_at: datetime


@dataclass(slots=True)
class SessionCompletionStats:
    total_sessions: int
    completed_sessions: int
    timed_out_sessions: int
    incomplete_sessions: int
    abandoned_sessions: int
    average_session_duration_seconds: float
    average_questions_per_session: float
    completion_rate: float
    timeout_rate: float


@dataclass(slots=True)
class GradedTotals:
    total_correct_answers: int
    total_incorrect_answers: int
    total_unanswered_questions: int
    total_questions_attempted: int
    overall_accuracy: float


@dataclass(slots=True)
class RawAnalyticsData:
    # Everything queries.py fetches for one user, in one pass — every
    # orchestration service reads from this same bundle instead of issuing
    # its own queries, which is what keeps the total query count fixed no
    # matter how many subjects/topics/sessions the user has.
    analytics_rows: list[Analytics]
    mastery_by_subject: dict[str, SubjectMastery]
    session_stats: SessionCompletionStats
    graded_totals: GradedTotals
    topic_rows: list[TopicRow]
    response_time_rows: list[ResponseTimeRow]
    session_completed_at: dict[int, datetime]
    trend_attempt_rows: list[TrendAttemptRow]
    repeated_attempt_rows: list[RepeatedAttemptRow]
    difficulty_attempt_rows: list[DifficultyAttemptRow]
    difficulty_session_rows: list[DifficultySessionRow]
    topic_difficulty_rows: list[TopicDifficultyRow]
    growth_session_rows: list[GrowthSessionRow]
    growth_attempt_rows: list[GrowthAttemptRow]
