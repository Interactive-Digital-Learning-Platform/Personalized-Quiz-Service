"""
services/analytics/mastery_service.py
────────────────────────────────────────
MasteryScoreService — orchestrates app.services.mastery_service's pure
formula (build_mastery_analytics et al.) for the subject/topic scopes,
gathering the inputs each needs from the topic-difficulty query results.
Contains no scoring formula of its own — see app/services/mastery_service.py
for that (imported below as `mastery_formulas` to keep the two distinct
"mastery_service" modules unambiguous at the call sites in this file).
"""
from app.core.config import settings
from app.services import difficulty_service
from app.services import mastery_service as mastery_formulas
from app.services.analytics.types import TopicDifficultyRow


class MasteryScoreService:
    def __init__(self, topic_difficulty_rows: list[TopicDifficultyRow]):
        # Per-(subject, topic, difficulty) accuracy — used only to feed a
        # topic's difficulty_score component. Not derived from LessonMastery
        # — see queries.fetch_topic_difficulty_rows()'s docstring for why.
        self._topic_difficulty_accuracy: dict[tuple[str, str, str], float] = {}
        for row in topic_difficulty_rows:
            if row.total_attempted > 0:
                self._topic_difficulty_accuracy[(row.subject, row.topic, row.difficulty)] = round(
                    row.total_correct / row.total_attempted * 100.0, 2
                )

    def score_topic(
        self,
        *,
        subject: str,
        topic: str,
        total_attempted: int,
        accuracy: float,
        trend: dict,
        repeated: dict,
        most_recent_difficulty: str,
        session_accuracies: list[float],
    ) -> dict:
        accuracy_at_difficulty = self._topic_difficulty_accuracy.get((subject, topic, most_recent_difficulty))
        return self._build(
            total_attempted=total_attempted,
            accuracy=accuracy,
            recent_performance=self._recent_performance(trend, accuracy),
            current_difficulty=most_recent_difficulty,
            accuracy_at_difficulty=accuracy_at_difficulty,
            repeated=repeated,
            session_accuracies=session_accuracies,
        )

    def score_subject(
        self,
        *,
        total_attempted: int,
        accuracy: float,
        trend: dict,
        repeated: dict,
        current_difficulty: str,
        accuracy_at_difficulty: float | None,
        session_accuracies: list[float],
    ) -> dict:
        return self._build(
            total_attempted=total_attempted,
            accuracy=accuracy,
            recent_performance=self._recent_performance(trend, accuracy),
            current_difficulty=current_difficulty,
            accuracy_at_difficulty=accuracy_at_difficulty,
            repeated=repeated,
            session_accuracies=session_accuracies,
        )

    @staticmethod
    def _recent_performance(trend: dict, accuracy: float) -> float:
        """recent_performance_score: the trend's current-period accuracy
        when it's meaningful, else this scope's own overall accuracy as the
        best available estimate of "recent" performance."""
        return trend["current_period_accuracy"] if trend["method"] != "insufficient_data" else accuracy

    @staticmethod
    def _build(
        *, total_attempted, accuracy, recent_performance, current_difficulty,
        accuracy_at_difficulty, repeated, session_accuracies,
    ) -> dict:
        return mastery_formulas.build_mastery_analytics(
            total_attempted=total_attempted,
            accuracy=accuracy,
            recent_performance_score=recent_performance,
            current_difficulty=current_difficulty,
            accuracy_at_current_difficulty=accuracy_at_difficulty,
            corrected_previous_mistakes=repeated["corrected_previous_mistakes"],
            repeated_same_mistakes=repeated["repeated_same_mistakes"],
            mistake_correction_rate=repeated["mistake_correction_rate"],
            session_accuracies=session_accuracies,
            min_attempts=settings.ANALYTICS_MASTERY_MIN_ATTEMPTS,
            base_difficulty_scores=difficulty_service.DIFFICULTY_BASE_SCORES,
            neutral_score=settings.ANALYTICS_MASTERY_NEUTRAL_SCORE,
            consistency_min_sessions=settings.ANALYTICS_MASTERY_CONSISTENCY_MIN_SESSIONS,
            accuracy_weight=settings.ANALYTICS_MASTERY_ACCURACY_WEIGHT,
            recent_performance_weight=settings.ANALYTICS_MASTERY_RECENT_PERFORMANCE_WEIGHT,
            difficulty_weight=settings.ANALYTICS_MASTERY_DIFFICULTY_WEIGHT,
            retention_weight=settings.ANALYTICS_MASTERY_RETENTION_WEIGHT,
            consistency_weight=settings.ANALYTICS_MASTERY_CONSISTENCY_WEIGHT,
            developing_threshold=settings.ANALYTICS_MASTERY_DEVELOPING_THRESHOLD,
            proficient_threshold=settings.ANALYTICS_MASTERY_PROFICIENT_THRESHOLD,
            advanced_threshold=settings.ANALYTICS_MASTERY_ADVANCED_THRESHOLD,
        )
