"""
services/analytics/types.py
──────────────────────────────
Typed internal data structures for the GET /analytics/me pipeline.

Two categories:
1. "Row" dataclasses — one per DB query in queries.py, shaping raw
   SQLAlchemy Result rows into plain, typed, IDE-friendly objects instead of
   passing Row/RowMapping objects (or ad-hoc dicts) between layers.
2. Aggregate dataclasses — typed containers for values computed directly by
   a single SQL aggregate query (no further per-row grouping needed), and
   the RawAnalyticsData bundle that carries everything queries.py fetched
   into the orchestration layer in one pass.

None of these are Pydantic models — they never cross the API boundary
directly (the route only ever returns UserAnalyticsResponse, built from a
plain dict at the very end) — plain dataclasses are enough here and keep
this internal layer decoupled from the API schema layer.
"""
from dataclasses import dataclass
from datetime import datetime

from app.models.analytics import Analytics
from app.models.subject_mastery import SubjectMastery


# ─────────────────────────────────────────────────────────────────────────────
# Per-query row shapes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TopicRow:
    """One row of the per-(subject, topic) graded-attempt aggregate."""
    subject: str
    topic: str
    total_attempted: int
    total_correct: int
    last_attempted_at: datetime


@dataclass(frozen=True, slots=True)
class ResponseTimeRow:
    """One VALID (already filtered) response time, with enough context to
    be grouped by overall/subject/topic scope."""
    subject: str
    topic: str
    correct: bool
    response_time: float


@dataclass(frozen=True, slots=True)
class TrendAttemptRow:
    """One graded attempt belonging to a COMPLETED session, for performance-
    trend session-bucketing."""
    session_id: int
    subject: str
    topic: str
    correct: bool


@dataclass(frozen=True, slots=True)
class RepeatedAttemptRow:
    """One graded attempt, chronologically ordered, for repeated-question/
    repeated-mistake grouping by Question.question_fingerprint."""
    attempt_id: int
    correct: bool
    fingerprint: str
    subject: str
    topic: str
    difficulty: str


@dataclass(frozen=True, slots=True)
class DifficultyAttemptRow:
    """One row of the per-(subject, difficulty) graded-attempt aggregate."""
    subject: str
    difficulty: str
    total_attempted: int
    total_correct: int
    avg_response_time: float


@dataclass(frozen=True, slots=True)
class DifficultySessionRow:
    """One row of the per-(subject, difficulty) distinct-completed-session count."""
    subject: str
    difficulty: str
    completed_sessions: int


@dataclass(frozen=True, slots=True)
class TopicDifficultyRow:
    """One row of the per-(subject, topic, difficulty) graded-attempt
    aggregate — used only to feed topic-level mastery's difficulty_score."""
    subject: str
    topic: str
    difficulty: str
    total_attempted: int
    total_correct: int


@dataclass(frozen=True, slots=True)
class GrowthSessionRow:
    """One session started within the growth rolling window, with its
    completion/abandonment-relevant fields."""
    session_id: int
    created_at: datetime
    completion_id: int | None
    last_saved_at: datetime | None


@dataclass(frozen=True, slots=True)
class GrowthAttemptRow:
    """One graded attempt within the growth rolling window."""
    session_id: int
    correct: bool
    subject: str
    topic: str
    session_created_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate containers (already fully computed by a single SQL query)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SessionCompletionStats:
    """Session-completion figures — see AnalyticsSummaryService for the
    classification rules (completed/incomplete/timed_out/abandoned)."""
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
    """Weighted, attempt-level correct/incorrect/unanswered totals — the
    source of truth for overall_accuracy (see AnalyticsSummaryService)."""
    total_correct_answers: int
    total_incorrect_answers: int
    total_unanswered_questions: int
    total_questions_attempted: int
    overall_accuracy: float


# ─────────────────────────────────────────────────────────────────────────────
# The full raw-data bundle handed to the orchestration layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class RawAnalyticsData:
    """
    Everything queries.py fetches for one user, in ONE pass — see
    orchestrator.py. Every one of the 8 orchestration services reads from
    this same bundle rather than issuing its own queries, which is what
    keeps the total query count fixed regardless of how many subjects/
    topics/sessions the user has (see queries.py's module docstring for the
    full list and orchestrator.py for the final count).
    """
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
