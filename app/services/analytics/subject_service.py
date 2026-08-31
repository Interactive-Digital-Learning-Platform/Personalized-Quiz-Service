from app.core.config import settings
from app.models.analytics import Analytics
from app.models.subject_mastery import SubjectMastery
from app.services import difficulty_service
from app.services.analytics.mastery_service import MasteryScoreService
from app.services.analytics.repeated_mistake_service import RepeatedMistakeAnalyticsService
from app.services.analytics.summary_service import compute_answering_behavior, compute_response_time_stats
from app.services.analytics.trend_service import TrendAnalyticsService
from app.services.analytics.types import DifficultyAttemptRow, DifficultySessionRow, ResponseTimeRow


def _build_adaptive_mastery_detail(mastery: SubjectMastery | None) -> dict | None:
    # Straight projection of the SubjectMastery row the real adaptive engine
    # writes (difficulty_service.py / difficulty_mastery_engine.py) — no
    # recomputation here, so this can never drift from what actually drove
    # difficulty selection. None for a subject with no SubjectMastery row
    # yet (never attempted) rather than a row full of made-up defaults.
    if mastery is None:
        return None
    return {
        "mastery_score": mastery.mastery_score,
        "fluency_score": mastery.fluency_score,
        "confidence_score": mastery.confidence_score,
        "evidence_count": mastery.evidence_count,
        "recent_accuracy": mastery.recent_accuracy,
        "previous_accuracy": mastery.previous_accuracy,
        "trend_score": mastery.trend_score,
        "trend_label": mastery.trend_label,
        "retention_score": mastery.retention_score,
        "last_mastery_update": mastery.last_mastery_update,
    }


def build_difficulty_performance_by_subject(
    attempt_rows: list[DifficultyAttemptRow], session_rows: list[DifficultySessionRow],
) -> dict[str, dict[str, dict]]:
    by_subject: dict[str, dict[str, dict]] = {}
    for row in attempt_rows:
        accuracy = (
            round(row.total_correct / row.total_attempted * 100.0, 2) if row.total_attempted > 0 else 0.0
        )
        by_subject.setdefault(row.subject, {})[row.difficulty] = {
            "difficulty": row.difficulty,
            "total_attempted": row.total_attempted,
            "total_correct": row.total_correct,
            "accuracy": accuracy,
            "avg_response_time": row.avg_response_time,
            "completed_sessions": 0,
        }
    for row in session_rows:
        bucket = by_subject.get(row.subject, {}).get(row.difficulty)
        if bucket is not None:
            bucket["completed_sessions"] = row.completed_sessions
    return by_subject


class SubjectAnalyticsService:
    def __init__(
        self,
        *,
        analytics_rows: list[Analytics],
        mastery_by_subject: dict[str, SubjectMastery],
        topics_by_subject: dict[str, list[dict]],
        subject_response_time_rows: dict[str, list[ResponseTimeRow]],
        overall_median_response_time: float,
        trend_service: TrendAnalyticsService,
        repeated_mistake_service: RepeatedMistakeAnalyticsService,
        mastery_service: MasteryScoreService,
        difficulty_performance_by_subject: dict[str, dict[str, dict]],
        max_topics: int,
    ):
        self._rows = analytics_rows
        self._mastery_by_subject = mastery_by_subject
        self._topics_by_subject = topics_by_subject
        self._subject_response_time_rows = subject_response_time_rows
        self._overall_median_response_time = overall_median_response_time
        self._trend = trend_service
        self._repeated = repeated_mistake_service
        self._mastery = mastery_service
        self._difficulty_performance_by_subject = difficulty_performance_by_subject
        self._max_topics = max_topics

    def _derive_weak_topic(self, subject: str, legacy_weak_topic: str | None) -> str | None:
        # Now derived from the lowest-accuracy topic that isn't
        # "insufficient_data" (falling back to the legacy Analytics.weak_topic
        # if none qualify) — read from the FULL topic list, already sorted
        # ascending by accuracy, so a topic beyond the shown top-N can still win.
        eligible = [t for t in self._topics_by_subject.get(subject, []) if t["status"] != "insufficient_data"]
        if eligible:
            return eligible[0]["topic"]
        return legacy_weak_topic

    def build(self) -> tuple[list[dict], list[str], list[str]]:
        sorted_rows = sorted(self._rows, key=lambda r: r.accuracy, reverse=True)
        midpoint = len(sorted_rows) // 2
        strong_subjects = [r.subject for r in sorted_rows[:midpoint]] if sorted_rows else []
        weak_subjects = [r.subject for r in sorted_rows[midpoint:]] if sorted_rows else []

        subjects_data = [self._build_one(r) for r in self._rows]
        return subjects_data, strong_subjects, weak_subjects

    def _build_one(self, r: Analytics) -> dict:
        subject = r.subject
        rt_stats = compute_response_time_stats(self._subject_response_time_rows.get(subject, []))
        mastery_info = difficulty_service.describe_subject_mastery(self._mastery_by_subject.get(subject))
        difficulty_buckets = self._difficulty_performance_by_subject.get(subject, {})
        difficulty_performance = [
            difficulty_buckets[d] for d in difficulty_service.DIFFICULTY_LEVELS if d in difficulty_buckets
        ]

        subject_session_stats = self._trend.subject_session_stats.get(subject, {})
        trend = self._trend.trend_for(subject_session_stats)
        repeated = self._repeated.for_subject(subject)

        topics = self._topics_by_subject.get(subject, [])
        total_attempted = sum(t["total_attempted"] for t in topics)
        # Computed from the FULL topic list (before the `topics[:max_topics]`
        # slice below), so this rollup isn't undercounted by the display cap.
        bkt_topics = [t["bkt_mastery"] for t in topics if t.get("bkt_mastery") is not None]
        mastered_skill_count = sum(1 for b in bkt_topics if b["mastery_label"] == "mastered")
        total_skill_count = len(bkt_topics)
        session_accuracies = [
            s["correct"] / s["total"] * 100.0 for s in subject_session_stats.values() if s["total"] > 0
        ]
        accuracy_at_difficulty_bucket = difficulty_buckets.get(mastery_info["current_difficulty"])
        accuracy_at_difficulty = (
            accuracy_at_difficulty_bucket["accuracy"] if accuracy_at_difficulty_bucket is not None else None
        )
        mastery = self._mastery.score_subject(
            total_attempted=total_attempted,
            accuracy=r.accuracy,
            trend=trend,
            repeated=repeated,
            current_difficulty=mastery_info["current_difficulty"],
            accuracy_at_difficulty=accuracy_at_difficulty,
            session_accuracies=session_accuracies,
        )

        return {
            "subject": subject,
            "accuracy": r.accuracy,
            "avg_response_time": rt_stats["avg_response_time"],
            "weak_topic": self._derive_weak_topic(subject, r.weak_topic),
            "current_difficulty": mastery_info["current_difficulty"],
            "topics": topics[: self._max_topics],
            "median_response_time": rt_stats["median_response_time"],
            "fastest_response_time": rt_stats["fastest_response_time"],
            "slowest_response_time": rt_stats["slowest_response_time"],
            "correct_answer_avg_response_time": rt_stats["correct_answer_avg_response_time"],
            "incorrect_answer_avg_response_time": rt_stats["incorrect_answer_avg_response_time"],
            "response_time_standard_deviation": rt_stats["response_time_standard_deviation"],
            "answering_behavior": compute_answering_behavior(rt_stats, r.accuracy, self._overall_median_response_time),
            "performance_trend": trend,
            "repeated_question_analytics": repeated,
            "difficulty_performance": difficulty_performance,
            "consecutive_strong_quizzes": mastery_info["consecutive_strong_quizzes"],
            "consecutive_weak_quizzes": mastery_info["consecutive_weak_quizzes"],
            "promotion_threshold": mastery_info["promotion_threshold"],
            "demotion_threshold": mastery_info["demotion_threshold"],
            "quizzes_required_for_promotion": mastery_info["quizzes_required_for_promotion"],
            "promotion_progress_percentage": mastery_info["promotion_progress_percentage"],
            "promotion_readiness": mastery_info["promotion_readiness"],
            "next_difficulty": mastery_info["next_difficulty"],
            "difficulty_status_message": mastery_info["difficulty_status_message"],
            "mastery_score": mastery["mastery_score"],
            "mastery_level": mastery["mastery_level"],
            "mastery_components": mastery["mastery_components"],
            "adaptive_mastery": _build_adaptive_mastery_detail(self._mastery_by_subject.get(subject)),
            "mastered_skill_count": mastered_skill_count,
            "total_skill_count": total_skill_count,
        }
