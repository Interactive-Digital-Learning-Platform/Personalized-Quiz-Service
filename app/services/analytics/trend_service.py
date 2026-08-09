from datetime import datetime, timezone

from app.core.config import settings
from app.services.analytics.types import TrendAttemptRow
from app.services.scoring_service import compute_performance_trend


class TrendAnalyticsService:
    # Buckets graded attempts from completed sessions into per-session
    # {correct, total} maps for overall/subject/topic, then hands off to
    # scoring_service.compute_performance_trend() for the actual formula.

    def __init__(self, trend_attempt_rows: list[TrendAttemptRow], session_completed_at: dict[int, datetime]):
        self._session_completed_at = session_completed_at
        self._now = datetime.now(timezone.utc)

        self.overall_session_stats: dict[int, dict[str, int]] = {}
        self.subject_session_stats: dict[str, dict[int, dict[str, int]]] = {}
        self.topic_session_stats: dict[tuple[str, str], dict[int, dict[str, int]]] = {}

        for row in trend_attempt_rows:
            self._bump(self.overall_session_stats, row.session_id, row.correct)
            self._bump(self.subject_session_stats.setdefault(row.subject, {}), row.session_id, row.correct)
            self._bump(
                self.topic_session_stats.setdefault((row.subject, row.topic), {}),
                row.session_id, row.correct,
            )

    @staticmethod
    def _bump(stats: dict[int, dict[str, int]], session_id: int, is_correct: bool) -> None:
        entry = stats.setdefault(session_id, {"correct": 0, "total": 0})
        entry["total"] += 1
        if is_correct:
            entry["correct"] += 1

    def trend_for(self, session_stats: dict[int, dict[str, int]]) -> dict:
        return compute_performance_trend(
            session_stats,
            self._session_completed_at,
            window_size=settings.ANALYTICS_TREND_SESSION_WINDOW,
            min_attempts_per_period=settings.ANALYTICS_TREND_MIN_ATTEMPTS_PER_PERIOD,
            change_threshold=settings.ANALYTICS_TREND_CHANGE_THRESHOLD,
            now=self._now,
        )
