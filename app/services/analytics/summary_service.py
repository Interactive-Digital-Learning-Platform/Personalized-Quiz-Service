from app.core.config import settings
from app.services.analytics.types import GradedTotals, ResponseTimeRow, SessionCompletionStats
from app.services.scoring_service import classify_answering_behavior, compute_median_and_stddev


def compute_response_time_stats(entries: list[ResponseTimeRow]) -> dict:
    # avg/median/fastest/slowest/stddev plus correct-vs-incorrect breakdown —
    # shared by the overall/subject/topic scopes, each just passing in a
    # differently-filtered `entries` list. median/stddev run in Python
    # (compute_median_and_stddev) since SQLite has no equivalent SQL function.
    values = [e.response_time for e in entries]
    correct_values = [e.response_time for e in entries if e.correct]
    incorrect_values = [e.response_time for e in entries if not e.correct]
    median, stddev = compute_median_and_stddev(values)
    return {
        "count": len(values),
        "avg_response_time": round(sum(values) / len(values), 3) if values else 0.0,
        "median_response_time": median,
        "fastest_response_time": round(min(values), 3) if values else 0.0,
        "slowest_response_time": round(max(values), 3) if values else 0.0,
        "correct_answer_avg_response_time": (
            round(sum(correct_values) / len(correct_values), 3) if correct_values else 0.0
        ),
        "incorrect_answer_avg_response_time": (
            round(sum(incorrect_values) / len(incorrect_values), 3) if incorrect_values else 0.0
        ),
        "response_time_standard_deviation": stddev,
    }


def compute_answering_behavior(rt_stats: dict, accuracy: float, overall_median_response_time: float) -> str:
    return classify_answering_behavior(
        avg_response_time=rt_stats["avg_response_time"],
        accuracy=accuracy,
        valid_attempt_count=rt_stats["count"],
        overall_median_response_time=overall_median_response_time,
        min_attempts=settings.ANALYTICS_BEHAVIOR_MIN_ATTEMPTS,
        accurate_threshold=settings.ANALYTICS_ACCURATE_THRESHOLD,
        balanced_ratio=settings.ANALYTICS_BALANCED_TIME_RATIO,
    )


class AnalyticsSummaryService:
    # Builds the overall (non-subject-scoped) summary section. Every session
    # is either completed or incomplete (always summing to total_sessions);
    # timed_out is a subset of completed, abandoned a subset of incomplete —
    # they overlap those two buckets rather than adding new ones.
    # overall_accuracy is total_correct/total_attempted across every graded
    # attempt, not an average of each subject's own accuracy — that would
    # weight a 2-attempt subject the same as a 200-attempt one.

    def __init__(
        self,
        session_stats: SessionCompletionStats,
        graded_totals: GradedTotals,
        overall_response_time_rows: list[ResponseTimeRow],
    ):
        self._session_stats = session_stats
        self._graded_totals = graded_totals
        self.response_time_stats = compute_response_time_stats(overall_response_time_rows)
        # What every subject/topic scope compares its own pace against.
        self.overall_median_response_time = (
            self.response_time_stats["median_response_time"]
            if self.response_time_stats["count"] > 0
            else settings.ANALYTICS_MEDIAN_FALLBACK_SECONDS
        )

    def build(self) -> dict:
        rt = self.response_time_stats
        s = self._session_stats
        g = self._graded_totals
        behavior = compute_answering_behavior(rt, g.overall_accuracy, self.overall_median_response_time)

        return {
            "overall_accuracy": g.overall_accuracy,
            "overall_avg_response_time": rt["avg_response_time"],
            "median_response_time": rt["median_response_time"],
            "fastest_response_time": rt["fastest_response_time"],
            "slowest_response_time": rt["slowest_response_time"],
            "correct_answer_avg_response_time": rt["correct_answer_avg_response_time"],
            "incorrect_answer_avg_response_time": rt["incorrect_answer_avg_response_time"],
            "response_time_standard_deviation": rt["response_time_standard_deviation"],
            "answering_behavior": behavior,
            "total_sessions": s.total_sessions,
            "total_questions_attempted": g.total_questions_attempted,
            "total_correct_answers": g.total_correct_answers,
            "total_incorrect_answers": g.total_incorrect_answers,
            "total_unanswered_questions": g.total_unanswered_questions,
            "completed_sessions": s.completed_sessions,
            "incomplete_sessions": s.incomplete_sessions,
            "timed_out_sessions": s.timed_out_sessions,
            "abandoned_sessions": s.abandoned_sessions,
            "completion_rate": s.completion_rate,
            "timeout_rate": s.timeout_rate,
            "average_session_duration_seconds": s.average_session_duration_seconds,
            "average_questions_per_session": s.average_questions_per_session,
        }
