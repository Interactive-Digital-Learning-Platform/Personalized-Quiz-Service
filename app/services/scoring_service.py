import itertools
import statistics
from datetime import datetime, timedelta


def calculate_score(correct_count: int, total_questions: int) -> float:
    return float(correct_count)


def calculate_accuracy(correct_count: int, total_questions: int) -> float:
    if total_questions == 0:
        return 0.0
    return round((correct_count / total_questions) * 100.0, 2)


def calculate_avg_response_time(timings: list[float]) -> float:
    if not timings:
        return 0.0
    return round(sum(timings) / len(timings), 3)


def identify_weak_topic(
    question_topics: dict[int, str | None],
    wrong_question_ids: list[int],
) -> str | None:
    if not wrong_question_ids:
        return None

    topic_counts: dict[str, int] = {}
    for qid in wrong_question_ids:
        topic = question_topics.get(qid)
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    if not topic_counts:
        return None

    return max(topic_counts, key=lambda t: topic_counts[t])


def compute_performance_band(accuracy: float) -> str:
    if accuracy >= 85.0:
        return "Excellent"
    elif accuracy >= 70.0:
        return "Good"
    elif accuracy >= 50.0:
        return "Fair"
    else:
        return "Needs Improvement"


def classify_topic_status(accuracy: float, total_attempted: int, min_attempts: int) -> str:
    # Too few attempts always wins over accuracy — not enough data to call it
    # either way. The 41% cutoff (not 40%) closes a gap the original spec
    # left undefined between "weak <= 40" and "developing 41-69.99" for
    # fractional accuracy values like 40.5%.
    if total_attempted < min_attempts:
        return "insufficient_data"
    if accuracy >= 70.0:
        return "strong"
    if accuracy >= 41.0:
        return "developing"
    return "weak"


def is_valid_response_time(response_time: float | None, max_seconds: float) -> bool:
    return response_time is not None and 0 <= response_time <= max_seconds


def compute_median_and_stddev(values: list[float]) -> tuple[float, float]:
    # Plain Python rather than Postgres's percentile_cont/stddev_samp,
    # because the test suite runs against SQLite, which has no equivalent —
    # doing it here means the tested code path and the real one are the same
    # code, not two implementations that can quietly drift apart.
    if not values:
        return 0.0, 0.0
    median = round(statistics.median(values), 3)
    stddev = round(statistics.stdev(values), 3) if len(values) >= 2 else 0.0
    return median, stddev


def classify_answering_behavior(
    *,
    avg_response_time: float,
    accuracy: float,
    valid_attempt_count: int,
    overall_median_response_time: float,
    min_attempts: int,
    accurate_threshold: float,
    balanced_ratio: float,
) -> str:
    # "Fast" is always judged against the user's own overall median response
    # time, never a per-subject/topic one — so a subject is compared to the
    # user's normal pace, not to itself. "Balanced" catches anything close
    # enough to that median (within balanced_ratio) regardless of accuracy.
    if valid_attempt_count < min_attempts:
        return "insufficient_data"

    if overall_median_response_time > 0:
        deviation_ratio = (
            abs(avg_response_time - overall_median_response_time) / overall_median_response_time
        )
        if deviation_ratio <= balanced_ratio:
            return "balanced"

    fast = avg_response_time <= overall_median_response_time
    accurate = accuracy >= accurate_threshold

    if fast and accurate:
        return "fast_and_accurate"
    if fast:
        return "fast_but_inaccurate"
    if accurate:
        return "slow_and_accurate"
    return "slow_and_inaccurate"


def start_of_iso_week(dt: datetime) -> datetime:
    monday = dt - timedelta(days=dt.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def compute_performance_trend(
    session_stats: dict[int, dict[str, int]],
    session_completed_at: dict[int, datetime],
    *,
    window_size: int,
    min_attempts_per_period: int,
    change_threshold: float,
    now: datetime,
) -> dict:
    # Compares recent activity against whatever came right before it, using
    # WEIGHTED accuracy (total correct / total attempted across the whole
    # period) so a 2-question session doesn't count as much as a 20-question
    # one. Uses the latest N completed sessions vs the N before them once
    # there's enough history (2x window_size); otherwise falls back to this
    # week vs last week. Either way, if a period ends up with too few graded
    # attempts, the result is "insufficient_data" rather than a shaky number.
    sessions_sorted = sorted(
        session_stats.keys(), key=lambda sid: session_completed_at[sid], reverse=True
    )

    def _period_totals(session_ids: list[int]) -> tuple[int, int]:
        correct = sum(session_stats[sid]["correct"] for sid in session_ids)
        total = sum(session_stats[sid]["total"] for sid in session_ids)
        return correct, total

    if len(sessions_sorted) >= 2 * window_size:
        method = "recent_sessions"
        current_ids = sessions_sorted[:window_size]
        previous_ids = sessions_sorted[window_size: 2 * window_size]
    else:
        method = "weekly"
        week_start = start_of_iso_week(now)
        previous_week_start = week_start - timedelta(days=7)
        current_ids = [sid for sid in sessions_sorted if session_completed_at[sid] >= week_start]
        previous_ids = [
            sid for sid in sessions_sorted
            if previous_week_start <= session_completed_at[sid] < week_start
        ]

    current_correct, current_total = _period_totals(current_ids)
    previous_correct, previous_total = _period_totals(previous_ids)

    if current_total < min_attempts_per_period or previous_total < min_attempts_per_period:
        return {
            "current_period_accuracy": 0.0,
            "previous_period_accuracy": 0.0,
            "accuracy_change": 0.0,
            "current_period_sessions": len(current_ids),
            "previous_period_sessions": len(previous_ids),
            "trend": "insufficient_data",
            "method": "insufficient_data",
        }

    # Round once from the raw fractions so accuracy_change can't drift from
    # current - previous due to each side being rounded independently first.
    current_accuracy_raw = current_correct / current_total * 100.0
    previous_accuracy_raw = previous_correct / previous_total * 100.0
    accuracy_change = round(current_accuracy_raw - previous_accuracy_raw, 2)

    if accuracy_change >= change_threshold:
        trend = "improving"
    elif accuracy_change <= -change_threshold:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "current_period_accuracy": round(current_accuracy_raw, 2),
        "previous_period_accuracy": round(previous_accuracy_raw, 2),
        "accuracy_change": accuracy_change,
        "current_period_sessions": len(current_ids),
        "previous_period_sessions": len(previous_ids),
        "trend": trend,
        "method": method,
    }


def compute_repeated_question_group_stats(ordered_corrects: list[bool]) -> dict[str, int]:
    # One "group" = every attempt at the same question (by fingerprint),
    # oldest first. Needs at least 2 entries — callers must have already
    # dropped groups with a single attempt, since there's no "previous
    # attempt" to compare against otherwise. A correct -> anything
    # transition isn't a "mistake" event either way, so it only counts
    # toward repeated_question_count/repeated_correct_count, not the
    # corrected/repeated-mistake counters below.
    repeated = ordered_corrects[1:]
    repeated_correct_count = sum(1 for c in repeated if c)
    repeated_incorrect_count = len(repeated) - repeated_correct_count

    corrected_previous_mistakes = 0
    repeated_same_mistakes = 0
    for prev_correct, cur_correct in itertools.pairwise(ordered_corrects):
        if not prev_correct and cur_correct:
            corrected_previous_mistakes += 1
        elif not prev_correct and not cur_correct:
            repeated_same_mistakes += 1

    return {
        "repeated_question_count": len(repeated),
        "repeated_correct_count": repeated_correct_count,
        "repeated_incorrect_count": repeated_incorrect_count,
        "corrected_previous_mistakes": corrected_previous_mistakes,
        "repeated_same_mistakes": repeated_same_mistakes,
    }


def aggregate_repeated_question_stats(group_stats: list[dict[str, int]]) -> dict:
    totals: dict[str, int | float] = {
        "repeated_question_count": 0,
        "repeated_correct_count": 0,
        "repeated_incorrect_count": 0,
        "corrected_previous_mistakes": 0,
        "repeated_same_mistakes": 0,
    }
    for stats in group_stats:
        for key in totals:
            totals[key] += stats[key]

    denominator = totals["corrected_previous_mistakes"] + totals["repeated_same_mistakes"]
    totals["mistake_correction_rate"] = (
        round(totals["corrected_previous_mistakes"] / denominator * 100.0, 2)
        if denominator > 0
        else 0.0
    )
    return totals
