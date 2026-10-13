from datetime import date, datetime

from pydantic import BaseModel


class PerformanceTrend(BaseModel):
    current_period_accuracy: float
    previous_period_accuracy: float
    accuracy_change: float
    current_period_sessions: int
    previous_period_sessions: int
    trend: str
    method: str


class AccuracyHistoryPoint(BaseModel):
    # One completed quiz in the same session set performance_trend is computed
    # from (completed, non-deleted, non-retake, graded), oldest first.
    session_id: int
    completed_at: datetime
    subject: str
    correct: int
    total: int
    accuracy: float


class DailyActivityPoint(BaseModel):
    # One local calendar day. accuracy / avg_response_time are null on days with no graded answers.
    date: date
    quizzes_completed: int
    questions_answered: int
    correct_answers: int
    accuracy: float | None
    study_time_seconds: float
    avg_response_time: float | None
    # Daily Progress Score (0-100) and its three components; all null on days with no graded answers.
    # baseline_accuracy is the student's accuracy over everything before this day (null with no prior history).
    baseline_accuracy: float | None
    progress_score: float | None
    score_accuracy: float | None
    score_improvement: float | None
    score_effort: float | None


class DailyAnalyticsResponse(BaseModel):
    # GET /analytics/daily: completed, non-deleted, non-retake quizzes (the same set the rest of analytics uses)
    # bucketed by the local date they were completed on in `timezone`, oldest day first, zero-activity days included.
    days: int
    timezone: str
    start_date: date
    end_date: date
    active_days: int
    total_quizzes_completed: int
    total_questions_answered: int
    total_correct_answers: int
    accuracy: float | None
    total_study_time_seconds: float
    avg_response_time: float | None
    weekly_progress_score: float | None  # mean of the active days' progress_score
    daily: list[DailyActivityPoint]


class MasteryComponents(BaseModel):
    accuracy_score: float
    recent_performance_score: float
    difficulty_score: float
    retention_score: float
    consistency_score: float


class AdaptiveMasteryDetail(BaseModel):
    # Output of the Continuous Evidence-Weighted Mastery System
    # (SubjectMastery / app/services/difficulty_mastery_engine.py) — the
    # actual driver of difficulty selection. Distinct from
    # mastery_score/mastery_level/mastery_components above, which are a
    # separate, analytics-only display metric (app/services/mastery_service.py)
    # that never affects difficulty.
    mastery_score: float
    fluency_score: float
    confidence_score: float
    evidence_count: int
    recent_accuracy: float | None
    previous_accuracy: float | None
    trend_score: float | None
    trend_label: str
    retention_score: float | None
    last_mastery_update: datetime | None


class RepeatedQuestionAnalytics(BaseModel):
    # "Repeated" means the same underlying question (by fingerprint) shown
    # again — questions only ever seen once aren't counted here at all.
    repeated_question_count: int
    repeated_correct_count: int
    repeated_incorrect_count: int
    corrected_previous_mistakes: int
    repeated_same_mistakes: int
    mistake_correction_rate: float


class BKTMasteryDetail(BaseModel):
    # Bayesian Knowledge Tracing state (SkillBKTState) — a read-only P(know)
    # signal, independent of mastery_score/mastery_level above (a different,
    # weighted-average formula) and of adaptive_mastery on SubjectAnalytics
    # below (the CEWM system that actually drives difficulty).
    p_know: float
    mastery_label: str
    opportunities: int
    last_updated: datetime | None


class TopicAnalytics(BaseModel):
    topic: str
    total_attempted: int
    total_correct: int
    total_incorrect: int
    accuracy: float
    avg_response_time: float
    last_attempted_at: datetime
    status: str

    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    answering_behavior: str

    performance_trend: PerformanceTrend
    repeated_question_analytics: RepeatedQuestionAnalytics

    mastery_score: float | None
    mastery_level: str
    mastery_components: MasteryComponents | None
    bkt_mastery: BKTMasteryDetail | None


class SubjectDifficultyPerformance(BaseModel):
    difficulty: str
    total_attempted: int
    total_correct: int
    accuracy: float
    avg_response_time: float
    completed_sessions: int


class SubjectAnalytics(BaseModel):
    subject: str
    accuracy: float
    avg_response_time: float
    weak_topic: str | None
    current_difficulty: str
    topics: list[TopicAnalytics] = []

    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    answering_behavior: str

    performance_trend: PerformanceTrend
    repeated_question_analytics: RepeatedQuestionAnalytics

    difficulty_performance: list[SubjectDifficultyPerformance] = []
    consecutive_strong_quizzes: int
    consecutive_weak_quizzes: int
    promotion_threshold: float
    demotion_threshold: float
    quizzes_required_for_promotion: int
    promotion_progress_percentage: float
    promotion_readiness: float
    # The two requirements behind promotion_readiness (= the lower of the two progress ratios): the adaptive mastery
    # score vs the level-up threshold, and questions answered vs the minimum evidence. *_required are null at the
    # highest difficulty.
    promotion_mastery_score: float = 0.0
    promotion_mastery_required: float | None = None
    promotion_questions_answered: int = 0
    promotion_questions_required: int | None = None
    next_difficulty: str
    difficulty_status_message: str

    mastery_score: float | None
    mastery_level: str
    mastery_components: MasteryComponents | None
    adaptive_mastery: AdaptiveMasteryDetail | None
    mastered_skill_count: int
    total_skill_count: int


class EffortComponents(BaseModel):
    completed_quiz_count_score: float
    attempted_question_count_score: float
    active_learning_days_score: float
    completion_rate_score: float
    weak_topic_attempts_score: float


class ConsistencyComponents(BaseModel):
    active_days_score: float
    session_spacing_score: float
    completion_rate_score: float
    score_stability_score: float
    low_abandonment_score: float


class ImprovementComponents(BaseModel):
    recent_accuracy_change_score: float
    topic_improvement_score: float
    repeated_mistake_correction_score: float


class GrowthComponents(BaseModel):
    effort: EffortComponents
    consistency: ConsistencyComponents
    improvement: ImprovementComponents


class GrowthAnalytics(BaseModel):
    effort_score: float | None
    consistency_score: float | None
    improvement_score: float | None
    mastery_score: float | None
    growth_score: float | None
    growth_level: str
    components: GrowthComponents | None


class RecommendationSupportingMetrics(BaseModel):
    accuracy: float | None
    attempts: int
    trend: str | None
    repeated_mistakes: int


class Recommendation(BaseModel):
    priority: int
    type: str
    subject: str | None
    topic: str | None
    reason: str
    recommended_action: str
    recommended_difficulty: str | None
    supporting_metrics: RecommendationSupportingMetrics


class UserAnalyticsResponse(BaseModel):
    overall_accuracy: float
    overall_avg_response_time: float
    total_sessions: int

    median_response_time: float
    fastest_response_time: float
    slowest_response_time: float
    correct_answer_avg_response_time: float
    incorrect_answer_avg_response_time: float
    response_time_standard_deviation: float
    answering_behavior: str
    total_questions_attempted: int
    total_correct_answers: int
    total_incorrect_answers: int
    total_unanswered_questions: int

    # completed_sessions + incomplete_sessions == total_sessions.
    # timed_out_sessions is a subset of completed_sessions (still counts as
    # completed, just also flagged). abandoned_sessions is a subset of
    # incomplete_sessions.
    completed_sessions: int
    incomplete_sessions: int
    timed_out_sessions: int
    abandoned_sessions: int
    completion_rate: float
    timeout_rate: float
    average_session_duration_seconds: float
    average_questions_per_session: float

    subjects: list[SubjectAnalytics]
    strong_subjects: list[str]
    weak_subjects: list[str]

    performance_trend: PerformanceTrend
    accuracy_history: list[AccuracyHistoryPoint] = []
    repeated_question_analytics: RepeatedQuestionAnalytics
    growth: GrowthAnalytics
    recommendations: list[Recommendation] = []


class AIFeedbackResponse(BaseModel):
    weak_areas: list[str]
    strong_areas: list[str]
    suggestions: list[str]
    motivational_note: str
    generated_at: datetime
