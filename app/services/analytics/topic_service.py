from app.core.config import settings
from app.services import difficulty_service
from app.services.analytics.mastery_service import MasteryScoreService
from app.services.analytics.repeated_mistake_service import RepeatedMistakeAnalyticsService
from app.services.analytics.summary_service import compute_answering_behavior, compute_response_time_stats
from app.services.analytics.trend_service import TrendAnalyticsService
from app.services.analytics.types import ResponseTimeRow, TopicRow
from app.services.scoring_service import classify_topic_status


class TopicAnalyticsService:
    # Owns subjects[].topics. Grouped by each QUESTION's own lesson field —
    # never QuizSession.lesson, which is just a session-level summary label
    # (often "Mixed") since one session can span multiple lessons.

    def __init__(self, topic_rows: list[TopicRow]):
        self._topic_rows = topic_rows

    def build(self) -> dict[str, list[dict]]:
        # Returns {subject: [topic_dict, ...]} with base fields only —
        # response-time/trend/repeated-mistake/mastery are added by the
        # attach_* methods below, in the order orchestrator.py calls them.
        min_topic_attempts = settings.ANALYTICS_TOPIC_MIN_ATTEMPTS
        topics_by_subject: dict[str, list[dict]] = {}
        for row in self._topic_rows:
            total_incorrect = row.total_attempted - row.total_correct
            accuracy = (
                round(row.total_correct / row.total_attempted * 100.0, 2)
                if row.total_attempted > 0 else 0.0
            )
            topics_by_subject.setdefault(row.subject, []).append({
                "topic": row.topic,
                "total_attempted": row.total_attempted,
                "total_correct": row.total_correct,
                "total_incorrect": total_incorrect,
                "accuracy": accuracy,
                "last_attempted_at": row.last_attempted_at,
                "status": classify_topic_status(accuracy, row.total_attempted, min_topic_attempts),
            })
        return topics_by_subject

    @staticmethod
    def attach_response_time(
        topics_by_subject: dict[str, list[dict]],
        topic_response_time_rows: dict[tuple[str, str], list[ResponseTimeRow]],
        overall_median_response_time: float,
    ) -> None:
        for subject, topics in topics_by_subject.items():
            for topic_dict in topics:
                rt = compute_response_time_stats(
                    topic_response_time_rows.get((subject, topic_dict["topic"]), [])
                )
                topic_dict["avg_response_time"] = rt["avg_response_time"]
                topic_dict["median_response_time"] = rt["median_response_time"]
                topic_dict["fastest_response_time"] = rt["fastest_response_time"]
                topic_dict["slowest_response_time"] = rt["slowest_response_time"]
                topic_dict["correct_answer_avg_response_time"] = rt["correct_answer_avg_response_time"]
                topic_dict["incorrect_answer_avg_response_time"] = rt["incorrect_answer_avg_response_time"]
                topic_dict["response_time_standard_deviation"] = rt["response_time_standard_deviation"]
                topic_dict["answering_behavior"] = compute_answering_behavior(
                    rt, topic_dict["accuracy"], overall_median_response_time
                )

    @staticmethod
    def attach_trend(topics_by_subject: dict[str, list[dict]], trend_service: TrendAnalyticsService) -> None:
        for subject, topics in topics_by_subject.items():
            for topic_dict in topics:
                topic_dict["performance_trend"] = trend_service.trend_for(
                    trend_service.topic_session_stats.get((subject, topic_dict["topic"]), {})
                )

    @staticmethod
    def attach_repeated_mistakes(
        topics_by_subject: dict[str, list[dict]], repeated_service: RepeatedMistakeAnalyticsService,
    ) -> None:
        for subject, topics in topics_by_subject.items():
            for topic_dict in topics:
                topic_dict["repeated_question_analytics"] = repeated_service.for_topic(subject, topic_dict["topic"])

    @staticmethod
    def attach_mastery(
        topics_by_subject: dict[str, list[dict]],
        mastery_service: MasteryScoreService,
        trend_service: TrendAnalyticsService,
        repeated_service: RepeatedMistakeAnalyticsService,
    ) -> None:
        # Must run after attach_trend/attach_repeated_mistakes since mastery
        # reuses their outputs.
        for subject, topics in topics_by_subject.items():
            for topic_dict in topics:
                topic = topic_dict["topic"]
                current_difficulty = repeated_service.most_recent_topic_difficulty.get(
                    (subject, topic), difficulty_service.DEFAULT_DIFFICULTY
                )
                session_accuracies = [
                    s["correct"] / s["total"] * 100.0
                    for s in trend_service.topic_session_stats.get((subject, topic), {}).values()
                    if s["total"] > 0
                ]
                mastery = mastery_service.score_topic(
                    subject=subject,
                    topic=topic,
                    total_attempted=topic_dict["total_attempted"],
                    accuracy=topic_dict["accuracy"],
                    trend=topic_dict["performance_trend"],
                    repeated=topic_dict["repeated_question_analytics"],
                    most_recent_difficulty=current_difficulty,
                    session_accuracies=session_accuracies,
                )
                topic_dict["mastery_score"] = mastery["mastery_score"]
                topic_dict["mastery_level"] = mastery["mastery_level"]
                topic_dict["mastery_components"] = mastery["mastery_components"]

    @staticmethod
    def sort_topics(topics_by_subject: dict[str, list[dict]]) -> None:
        # Ascending by accuracy (weakest first), tie-broken by topic name for
        # a stable, deterministic order.
        for topics in topics_by_subject.values():
            topics.sort(key=lambda t: (t["accuracy"], t["topic"]))
