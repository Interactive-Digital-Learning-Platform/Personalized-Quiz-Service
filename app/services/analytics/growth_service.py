from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.services import growth_service as growth_formulas
from app.services.analytics.queries import as_utc
from app.services.analytics.types import GrowthAttemptRow, GrowthSessionRow


class GrowthAnalyticsService:
    # Gathers the inputs app.services.growth_service's pure formula needs for
    # the `growth` section. Effort/consistency come from their own rolling-
    # window queries; improvement and the overall mastery input reuse data
    # TrendAnalyticsService/RepeatedMistakeAnalyticsService/
    # SubjectAnalyticsService already computed, rather than recomputing anything.

    def __init__(
        self,
        *,
        growth_session_rows: list[GrowthSessionRow],
        growth_attempt_rows: list[GrowthAttemptRow],
        topics_by_subject: dict[str, list[dict]],
        overall_performance_trend: dict,
        overall_repeated_question_analytics: dict,
        subjects_data: list[dict],
    ):
        self._growth_session_rows = growth_session_rows
        self._growth_attempt_rows = growth_attempt_rows
        self._topics_by_subject = topics_by_subject
        self._overall_trend = overall_performance_trend
        self._overall_repeated = overall_repeated_question_analytics
        self._subjects_data = subjects_data

    def build(self) -> dict:
        total_sessions_in_window = len(self._growth_session_rows)
        completed_quiz_count = sum(1 for row in self._growth_session_rows if row.completion_id is not None)
        completion_rate = growth_formulas.compute_rate_score(completed_quiz_count, total_sessions_in_window)

        # Same "incomplete + inactive longer than the threshold" definition
        # as the all-time abandoned_sessions figure, just scoped to this window.
        abandoned_cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.ANALYTICS_ABANDONED_AFTER_HOURS)
        abandoned_in_window = sum(
            1
            for row in self._growth_session_rows
            if row.completion_id is None and as_utc(row.last_saved_at or row.created_at) < abandoned_cutoff
        )
        abandonment_free_rate = growth_formulas.compute_rate_score(
            total_sessions_in_window - abandoned_in_window, total_sessions_in_window
        )

        total_attempts_in_window = len(self._growth_attempt_rows)
        active_dates = sorted({as_utc(row.session_created_at).date() for row in self._growth_attempt_rows})
        active_learning_days = len(active_dates)

        weak_topic_keys = {
            (subject, t["topic"])
            for subject, topics in self._topics_by_subject.items()
            for t in topics
            if t["status"] == "weak"
        }
        weak_topic_attempts = sum(
            1 for row in self._growth_attempt_rows if (row.subject, row.topic) in weak_topic_keys
        )

        session_stats: dict[int, dict[str, int]] = {}
        for row in self._growth_attempt_rows:
            entry = session_stats.setdefault(row.session_id, {"correct": 0, "total": 0})
            entry["total"] += 1
            if row.correct:
                entry["correct"] += 1
        session_accuracies = [
            s["correct"] / s["total"] * 100.0 for s in session_stats.values() if s["total"] > 0
        ]

        accuracy_change = (
            self._overall_trend["accuracy_change"] if self._overall_trend["method"] != "insufficient_data" else None
        )
        improving_topic_count = sum(
            1 for topics in self._topics_by_subject.values() for t in topics
            if t["performance_trend"]["trend"] == "improving"
        )
        declining_topic_count = sum(
            1 for topics in self._topics_by_subject.values() for t in topics
            if t["performance_trend"]["trend"] == "declining"
        )
        repeated_denominator = (
            self._overall_repeated["corrected_previous_mistakes"] + self._overall_repeated["repeated_same_mistakes"]
        )
        repeated_mistake_rate = (
            self._overall_repeated["mistake_correction_rate"] if repeated_denominator > 0 else None
        )

        # Overall mastery = average of subjects[].mastery_score across
        # subjects with enough data to have one. This doesn't gate growth
        # itself — growth describes recent trajectory, not established mastery.
        subject_mastery_scores = [s["mastery_score"] for s in self._subjects_data if s["mastery_score"] is not None]
        mastery_score = (
            round(sum(subject_mastery_scores) / len(subject_mastery_scores), 2)
            if subject_mastery_scores else settings.ANALYTICS_MASTERY_NEUTRAL_SCORE
        )

        return growth_formulas.build_growth_analytics(
            total_attempts_in_window=total_attempts_in_window,
            min_attempts_in_window=settings.ANALYTICS_GROWTH_MIN_ATTEMPTS_IN_WINDOW,
            completed_quiz_count=completed_quiz_count,
            attempted_question_count=total_attempts_in_window,
            active_learning_days=active_learning_days,
            completion_rate=completion_rate,
            weak_topic_attempts=weak_topic_attempts,
            has_weak_topics=len(weak_topic_keys) > 0,
            active_dates=active_dates,
            abandonment_free_rate=abandonment_free_rate,
            session_accuracies=session_accuracies,
            consistency_min_sessions=settings.ANALYTICS_MASTERY_CONSISTENCY_MIN_SESSIONS,
            accuracy_change=accuracy_change,
            improving_topic_count=improving_topic_count,
            declining_topic_count=declining_topic_count,
            repeated_mistake_correction_rate=repeated_mistake_rate,
            mastery_score=mastery_score,
            window_days=settings.ANALYTICS_GROWTH_WINDOW_DAYS,
            target_quizzes=settings.ANALYTICS_GROWTH_EFFORT_TARGET_QUIZZES,
            target_questions=settings.ANALYTICS_GROWTH_EFFORT_TARGET_QUESTIONS,
            target_weak_topic_attempts=settings.ANALYTICS_GROWTH_EFFORT_TARGET_WEAK_TOPIC_ATTEMPTS,
            neutral_score=settings.ANALYTICS_MASTERY_NEUTRAL_SCORE,
            effort_weight=settings.ANALYTICS_GROWTH_EFFORT_WEIGHT,
            consistency_weight=settings.ANALYTICS_GROWTH_CONSISTENCY_WEIGHT,
            improvement_weight=settings.ANALYTICS_GROWTH_IMPROVEMENT_WEIGHT,
            mastery_weight=settings.ANALYTICS_GROWTH_MASTERY_WEIGHT,
            growing_threshold=settings.ANALYTICS_GROWTH_GROWING_THRESHOLD,
            strong_threshold=settings.ANALYTICS_GROWTH_STRONG_THRESHOLD,
            exceptional_threshold=settings.ANALYTICS_GROWTH_EXCEPTIONAL_THRESHOLD,
        )
