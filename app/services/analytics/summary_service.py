"""
services/analytics/summary_service.py
────────────────────────────────────────
AnalyticsSummaryService — the OVERALL (non-subject-scoped) summary fields
for GET /analytics/me: weighted accuracy, response-time statistics and
answering behavior, and session-completion figures.

Also hosts two small pure functions shared by SubjectAnalyticsService and
TopicAnalyticsService — response-time stats and answering-behavior
classification are the same formula at every scope (overall/subject/topic),
just narrowed to that scope's own rows, so the formula lives here ONCE
rather than being reimplemented per scope.
"""
from app.core.config import settings
from app.services.analytics.types import GradedTotals, ResponseTimeRow, SessionCompletionStats
from app.services.scoring_service import classify_answering_behavior, compute_median_and_stddev


def compute_response_time_stats(entries: list[ResponseTimeRow]) -> dict:
    """
    avg/median/fastest(min)/slowest(max)/stddev response time, plus a
    correct-vs-incorrect breakdown, computed from whatever scope's list of
    ResponseTimeRow is passed in — overall/subject/topic all call this the
    same way, just with a differently-filtered `entries` list.

    median/stddev are computed here in Python (via scoring_service.
    compute_median_and_stddev) rather than SQL because they can't be
    correctly recombined from grouped sub-aggregates (a median of a union
    isn't derivable from its parts' medians), and because SQLite — this
    project's test database — has no percentile_cont/stddev_samp
    equivalent at all.
    """
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
    """
    Thin wrapper around scoring_service.classify_answering_behavior() that
    reads its threshold/weight configuration from settings once here, so
    every call site (overall/subject/topic) stays consistent by construction.
    "fast" is always relative to the user's OVERALL median response time,
    never a subject's or topic's own — see classify_answering_behavior()'s
    own docstring for why.
    """
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
    """
    Builds the overall summary section of GET /analytics/me.

    ── Session-completion classification ───────────────────────────────────
    Every QuizSession falls into exactly one of two DISJOINT buckets
    (always summing to total_sessions): completed (a QuizCompletion row
    exists) or incomplete (it doesn't). Two further categories OVERLAP those
    buckets rather than adding new ones: timed_out is a SUBSET of completed
    (a quiz that hit the timer still counts as completed, but is also
    flagged); abandoned is a SUBSET of incomplete (inactive longer than
    ANALYTICS_ABANDONED_AFTER_HOURS). Deleted sessions are never counted —
    QuizSession rows are hard-deleted, so there's no soft-delete flag to filter.

    ── Weighted accuracy ────────────────────────────────────────────────────
    overall_accuracy is total_correct_answers / total_questions_attempted
    across every graded attempt — deliberately NOT an average of each
    subject's own accuracy, which would weight a subject with 2 attempts
    the same as one with 200.
    """

    def __init__(
        self,
        session_stats: SessionCompletionStats,
        graded_totals: GradedTotals,
        overall_response_time_rows: list[ResponseTimeRow],
    ):
        self._session_stats = session_stats
        self._graded_totals = graded_totals
        self.response_time_stats = compute_response_time_stats(overall_response_time_rows)
        # Shared reference every subject/topic scope compares its OWN pace
        # against — see compute_answering_behavior()'s docstring. Falls back
        # to a configured constant when there's no valid response time yet.
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
