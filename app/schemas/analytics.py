from datetime import datetime
from pydantic import BaseModel


class PerformanceTrend(BaseModel):
    current_period_accuracy: float
    previous_period_accuracy: float
    accuracy_change: float
    current_period_sessions: int
    previous_period_sessions: int
    trend: str
    method: str


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
    next_difficulty: str
    difficulty_status_message: str

    mastery_score: float | None
    mastery_level: str
    mastery_components: MasteryComponents | None
    adaptive_mastery: AdaptiveMasteryDetail | None


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
    repeated_question_analytics: RepeatedQuestionAnalytics
    growth: GrowthAnalytics
    recommendations: list[Recommendation] = []


class AIFeedbackResponse(BaseModel):
    weak_areas: list[str]
    strong_areas: list[str]
    suggestions: list[str]
    motivational_note: str
    generated_at: datetime
