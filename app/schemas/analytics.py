"""
schemas/analytics.py
─────────────────────
Pydantic v2 schemas for analytics and AI feedback endpoints.
"""
from datetime import datetime
from pydantic import BaseModel


# ── GET /analytics/me ─────────────────────────────────────────────────────────

class PerformanceTrend(BaseModel):
    """
    Whether a scope (overall/subject/topic) is improving, declining, or
    stable, comparing two non-overlapping periods of completed sessions.
    Both accuracies are WEIGHTED (total correct / total attempted across
    every session in the period), never an average of each session's own
    percentage — see scoring_service.compute_performance_trend().
    """
    current_period_accuracy: float    # Percentage (0.0 - 100.0)
    previous_period_accuracy: float   # Percentage (0.0 - 100.0)
    accuracy_change: float            # current - previous, percentage points
    current_period_sessions: int
    previous_period_sessions: int
    # "improving" | "declining" | "stable" | "insufficient_data"
    trend: str
    # "recent_sessions" (latest N completed sessions vs the N before them) |
    # "weekly" (current calendar week vs previous calendar week, UTC) |
    # "insufficient_data" (neither method had enough graded attempts in both
    # periods — see ANALYTICS_TREND_MIN_ATTEMPTS_PER_PERIOD)
    method: str


class MasteryComponents(BaseModel):
    """
    The five inputs behind mastery_score, each already clamped to 0-100 —
    see app/services/mastery_service.py for exactly how each is derived.
    Returned alongside mastery_score so the result is explainable rather
    than an opaque number.
    """
    accuracy_score: float             # Weighted historical accuracy
    recent_performance_score: float   # Accuracy from recent completed sessions
    difficulty_score: float           # Difficulty tier, adjusted by performance at it
    retention_score: float            # Repeated-question mistake-correction rate
    consistency_score: float          # Inverted variation across completed sessions


class RepeatedQuestionAnalytics(BaseModel):
    """
    Whether the student corrected previous mistakes when shown the same or
    an equivalent question again — see scoring_service.
    compute_repeated_question_group_stats()/aggregate_repeated_question_stats()
    and analytics_service.get_user_analytics()'s repeated-question section.

    "Same or equivalent question" is identified by Question.question_fingerprint
    (see app/services/question_fingerprint.py) — a deterministic hash of
    normalized (subject, lesson, question text), since this app generates
    fresh Question rows on every quiz rather than reusing one row's ID.
    Questions attempted only once are ignored entirely (not part of any count
    below).
    """
    # Attempts after the first, across every repeated-question group in scope.
    repeated_question_count: int
    repeated_correct_count: int      # ...of which were themselves correct
    repeated_incorrect_count: int    # ...of which were themselves incorrect
    # A later CORRECT attempt whose immediately preceding attempt (on the
    # same fingerprint) was incorrect.
    corrected_previous_mistakes: int
    # A later INCORRECT attempt whose immediately preceding attempt (on the
    # same fingerprint) was also incorrect.
    repeated_same_mistakes: int
    # corrected_previous_mistakes / (corrected_previous_mistakes +
    # repeated_same_mistakes) * 100 — 0.0 if that denominator is 0 (e.g. every
    # repeat was correct->correct, so no mistake was ever there to fix or repeat).
    mistake_correction_rate: float


class TopicAnalytics(BaseModel):
    """
    Analytics for a single topic/lesson within a subject — derived from each
    QUESTION's own `lesson` field (see Question model), not from a session's
    single session-level `lesson` — a quiz's questions can each belong to a
    different lesson within the same subject.
    """
    topic: str                  # "Unknown Topic" if the question had no lesson set
    total_attempted: int         # Graded attempts (correct IS NOT NULL) for this topic
    total_correct: int
    total_incorrect: int
    accuracy: float              # Percentage (0.0 – 100.0)
    avg_response_time: float     # Seconds
    last_attempted_at: datetime  # UTC
    # "strong" | "developing" | "weak" | "insufficient_data"
    # (insufficient_data overrides accuracy when total_attempted is below
    # ANALYTICS_TOPIC_MIN_ATTEMPTS — see scoring_service.classify_topic_status)
    status: str

    # ── Response-time statistics ────────────────────────────────────────────
    # All computed only from valid response times for this topic (see
    # analytics_service.get_user_analytics()'s response-time section) — a
    # negative, null, or implausibly large response_time never counts.
    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    # "fast_and_accurate" | "fast_but_inaccurate" | "slow_and_accurate" |
    # "slow_and_inaccurate" | "balanced" | "insufficient_data" — see
    # scoring_service.classify_answering_behavior()
    answering_behavior: str

    # Trend for THIS topic specifically — "sessions" here means completed
    # sessions that had at least one graded attempt on this topic (a topic
    # can appear across sessions that also cover other topics/lessons in the
    # same subject).
    performance_trend: PerformanceTrend

    # Repeated-question stats scoped to fingerprint groups whose subject AND
    # lesson match this topic (a fingerprint always carries a single fixed
    # subject/lesson, since it's derived from them).
    repeated_question_analytics: RepeatedQuestionAnalytics

    # ── Mastery score ────────────────────────────────────────────────────────
    # See app/services/mastery_service.py. None/"insufficient_data" if this
    # topic has fewer than ANALYTICS_MASTERY_MIN_ATTEMPTS graded attempts.
    mastery_score: float | None
    # "beginner" | "developing" | "proficient" | "advanced" | "insufficient_data"
    mastery_level: str
    mastery_components: MasteryComponents | None


class SubjectDifficultyPerformance(BaseModel):
    """
    One difficulty level's performance within a subject — grouped by each
    QUESTION's own `difficulty` (not QuizSession.difficulty), so a session
    that happens to mix difficulty levels (not possible via the app's own
    generation flow today, but not assumed here) is still split correctly
    across the right buckets. `completed_sessions` counts distinct completed
    sessions with at least one graded attempt at this specific difficulty —
    a mixed-difficulty session would count toward more than one bucket here,
    which is the correct behavior for that edge case.
    """
    difficulty: str   # "easy" | "medium" | "hard"
    total_attempted: int
    total_correct: int
    accuracy: float              # Percentage (0.0 - 100.0)
    avg_response_time: float     # Seconds; only valid response times count
    completed_sessions: int


class SubjectAnalytics(BaseModel):
    """Analytics for a single subject."""
    subject: str
    accuracy: float            # Percentage (0.0 – 100.0)
    avg_response_time: float   # Seconds
    # Kept temporarily for backwards compatibility — now DERIVED from the
    # lowest-accuracy topic in `topics` that isn't "insufficient_data" (falls
    # back to the legacy Analytics.weak_topic value if no topic qualifies).
    # New code should prefer `topics` directly.
    weak_topic: str | None
    # The difficulty the student is currently on for this subject — driven by
    # SubjectMastery (see difficulty_service.py). "easy" if they haven't
    # completed a quiz in this subject yet.
    current_difficulty: str
    # Per-topic breakdown, sorted by accuracy ascending (weakest first),
    # capped at ANALYTICS_MAX_TOPICS_PER_SUBJECT entries.
    topics: list[TopicAnalytics] = []

    # ── Response-time statistics ────────────────────────────────────────────
    # See TopicAnalytics — same fields, computed for this subject as a whole
    # from all of its valid response times (not just its shown topics).
    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    answering_behavior: str

    # Trend for this subject — "sessions" means this subject's own completed
    # sessions (every QuizSession belongs to exactly one subject).
    performance_trend: PerformanceTrend

    # Repeated-question stats scoped to fingerprint groups whose subject
    # matches this subject (across all of that subject's topics/lessons).
    repeated_question_analytics: RepeatedQuestionAnalytics

    # ── Difficulty-level analytics ──────────────────────────────────────────
    # Performance broken down per difficulty level actually attempted in this
    # subject (see SubjectDifficultyPerformance) — NOT sorted by accuracy,
    # ordered easy -> medium -> hard.
    difficulty_performance: list[SubjectDifficultyPerformance] = []
    # The remaining fields all come from difficulty_service.
    # describe_subject_mastery() — a read-only projection of the SAME
    # SubjectMastery row and constants that actually drive promotion/
    # demotion (see difficulty_service.py). GET /analytics/me never writes
    # to SubjectMastery; it only describes whatever state already exists
    # there (safe "easy"/0/0 defaults if the subject has no mastery row yet).
    consecutive_strong_quizzes: int
    consecutive_weak_quizzes: int
    promotion_threshold: float    # Accuracy %, e.g. 80.0
    demotion_threshold: float     # Accuracy %, e.g. 40.0
    quizzes_required_for_promotion: int
    # 0-100: consecutive_strong_quizzes / quizzes_required_for_promotion,
    # forced to 0.0 if already at the highest difficulty (there's no next
    # level to progress toward).
    promotion_progress_percentage: float
    # The difficulty this subject would advance to next; equals
    # current_difficulty itself if already at the highest level (no further
    # promotion possible).
    next_difficulty: str
    # Short, human-readable status derived from the fields above — see
    # difficulty_service.describe_subject_mastery().
    difficulty_status_message: str

    # ── Mastery score ────────────────────────────────────────────────────────
    # See app/services/mastery_service.py. None/"insufficient_data" if this
    # subject has fewer than ANALYTICS_MASTERY_MIN_ATTEMPTS graded attempts.
    mastery_score: float | None
    # "beginner" | "developing" | "proficient" | "advanced" | "insufficient_data"
    mastery_level: str
    mastery_components: MasteryComponents | None


class EffortComponents(BaseModel):
    """See app/services/growth_service.py:compute_effort_score()."""
    completed_quiz_count_score: float
    attempted_question_count_score: float
    active_learning_days_score: float
    completion_rate_score: float
    weak_topic_attempts_score: float


class ConsistencyComponents(BaseModel):
    """See app/services/growth_service.py:compute_consistency_score()."""
    active_days_score: float
    session_spacing_score: float
    completion_rate_score: float
    score_stability_score: float
    low_abandonment_score: float


class ImprovementComponents(BaseModel):
    """See app/services/growth_service.py:compute_improvement_score()."""
    recent_accuracy_change_score: float
    topic_improvement_score: float
    repeated_mistake_correction_score: float


class GrowthComponents(BaseModel):
    effort: EffortComponents
    consistency: ConsistencyComponents
    improvement: ImprovementComponents


class GrowthAnalytics(BaseModel):
    """
    Growth-oriented analytics — deliberately NOT a ranking or a raw-score
    proxy. Describes whether the student is putting in effort, showing up
    consistently, and actually improving, over a rolling window (see
    ANALYTICS_GROWTH_WINDOW_DAYS) — current mastery contributes only a
    modest, fixed 15% weight so it can't dominate the result. See
    app/services/growth_service.py for the full formula and the reasoning
    behind each design choice.

    All fields are None/"insufficient_data" together when the user has
    fewer than ANALYTICS_GROWTH_MIN_ATTEMPTS_IN_WINDOW graded attempts in
    the rolling window — there isn't enough recent activity to describe a
    trajectory, regardless of how much all-time history exists.

    Nothing here is persisted — every field is computed fresh from stored
    quiz/attempt activity on each request.
    """
    effort_score: float | None
    consistency_score: float | None
    improvement_score: float | None
    # The user's overall mastery — the average of subjects[].mastery_score
    # across subjects with enough data to have one (see analytics_service.
    # get_user_analytics()'s growth section). Neutral (50.0) if no subject
    # has enough data yet; this does NOT gate the rest of growth, since
    # growth describes recent trajectory, which doesn't require established
    # per-subject mastery.
    mastery_score: float | None
    growth_score: float | None
    # "starting" | "growing" | "strong_growth" | "exceptional_growth" | "insufficient_data"
    growth_level: str
    components: GrowthComponents | None


class RecommendationSupportingMetrics(BaseModel):
    """The concrete numbers behind a recommendation's `reason` — present so
    the recommendation is explainable rather than an opaque suggestion."""
    accuracy: float | None
    attempts: int
    trend: str | None
    repeated_mistakes: int


class Recommendation(BaseModel):
    """
    One deterministic, database-driven recommendation — see
    app/services/recommendation_service.py. Never produced by calling Groq;
    every field here is derived directly from other already-computed
    analytics fields on this same response.
    """
    priority: int   # 1 = most urgent; ranked, never a raw score
    # "weak_topic" | "declining_subject" | "repeated_mistake" |
    # "careless_guessing" | "slow_response" | "incomplete_quiz" |
    # "difficulty_ready_for_promotion" | "maintain_strong_subject"
    type: str
    subject: str | None   # None for overall-scope recommendations (e.g. incomplete_quiz)
    topic: str | None
    reason: str
    recommended_action: str
    recommended_difficulty: str | None   # None when a difficulty isn't relevant to this type
    supporting_metrics: RecommendationSupportingMetrics


class UserAnalyticsResponse(BaseModel):
    """
    Full analytics profile for the authenticated user.

    `overall_accuracy` and `overall_avg_response_time` are computed directly
    from QuestionAttempt rows (weighted across every attempt), NOT from an
    unweighted average of each subject's own accuracy — see
    analytics_service.get_user_analytics() for why that distinction matters.
    """
    overall_accuracy: float
    overall_avg_response_time: float
    total_sessions: int

    # ── Response-time statistics (overall, across every subject/topic) ─────────
    # Computed only from valid response times — see
    # analytics_service.get_user_analytics()'s response-time section and
    # scoring_service.is_valid_response_time()/classify_answering_behavior().
    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    # "fast_and_accurate" | "fast_but_inaccurate" | "slow_and_accurate" |
    # "slow_and_inaccurate" | "balanced" | "insufficient_data"
    answering_behavior: str
    # Grand total of questions across all of the user's sessions — equal to
    # total_correct_answers + total_incorrect_answers + total_unanswered_questions.
    total_questions_attempted: int
    total_correct_answers: int
    total_incorrect_answers: int
    # Inferred per-session from (session.question_count - recorded attempts),
    # not from any attempt explicitly marked incorrect. See requirement notes
    # in analytics_service.py.
    total_unanswered_questions: int

    # ── Session completion analytics ───────────────────────────────────────────
    # Categories OVERLAP by design — see analytics_service.get_user_analytics()
    # for the full classification rules:
    #   - completed_sessions + incomplete_sessions == total_sessions (disjoint).
    #   - timed_out_sessions is a SUBSET of completed_sessions (a quiz that hit
    #     the timer still counts as completed, but is also flagged here).
    #   - abandoned_sessions is a SUBSET of incomplete_sessions.
    completed_sessions: int
    incomplete_sessions: int
    timed_out_sessions: int
    abandoned_sessions: int
    completion_rate: float   # Percentage (0.0 – 100.0)
    timeout_rate: float      # Percentage (0.0 – 100.0)
    average_session_duration_seconds: float  # Across completed sessions only
    average_questions_per_session: float     # Across all sessions

    subjects: list[SubjectAnalytics]
    # Subjects ordered best → worst accuracy
    strong_subjects: list[str]
    weak_subjects: list[str]

    # Trend across every completed session the user has, regardless of subject.
    performance_trend: PerformanceTrend

    # Repeated-question stats across every subject/topic for this user.
    repeated_question_analytics: RepeatedQuestionAnalytics

    # Growth-oriented analytics (effort/consistency/improvement + mastery) —
    # see GrowthAnalytics and app/services/growth_service.py.
    growth: GrowthAnalytics

    # Deterministic, database-driven recommendations (max
    # ANALYTICS_RECOMMENDATION_MAX_COUNT, priority-ordered, deduplicated by
    # subject/topic) — see app/services/recommendation_service.py. Empty
    # list when there's insufficient data to recommend anything.
    recommendations: list[Recommendation] = []


# ── GET /analytics/feedback ────────────────────────────────────────────────────

class AIFeedbackResponse(BaseModel):
    """AI-generated improvement suggestions from Groq."""
    weak_areas: list[str]        # Subject/topic areas needing work
    strong_areas: list[str]      # What the user is doing well
    suggestions: list[str]       # Concrete, actionable improvement tips
    motivational_note: str       # A short encouraging message from the AI
    generated_at: datetime       # When this feedback was generated
