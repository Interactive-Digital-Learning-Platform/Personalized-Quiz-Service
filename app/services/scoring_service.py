"""
services/scoring_service.py
────────────────────────────
Pure calculation functions for quiz scoring.

These are intentionally stateless pure functions — they take numbers and return
numbers. No DB access, no Groq calls. This makes them trivially unit-testable.

Called by quiz_service.py after grading individual attempts.
"""
import statistics
from datetime import datetime, timedelta


def calculate_score(correct_count: int, total_questions: int) -> float:
    """
    Returns the raw score (number of correct answers).
    Kept as a dedicated function for clarity and future extensibility
    (e.g. weighted scoring per difficulty).
    """
    return float(correct_count)


def calculate_accuracy(correct_count: int, total_questions: int) -> float:
    """
    Returns accuracy as a percentage (0.0 – 100.0).

    Example: 7 correct out of 10 → 70.0
    """
    if total_questions == 0:
        return 0.0
    return round((correct_count / total_questions) * 100.0, 2)


def calculate_avg_response_time(timings: list[float]) -> float:
    """
    Returns the average response time in seconds.

    `timings` is a list of per-question response times sent by the frontend.
    """
    if not timings:
        return 0.0
    return round(sum(timings) / len(timings), 3)


def identify_weak_topic(
    question_topics: dict[int, str | None],
    wrong_question_ids: list[int],
) -> str | None:
    """
    Given a mapping of question_id → topic/lesson, and the list of incorrectly
    answered question IDs, find the topic with the most wrong answers.

    Returns the weak topic name, or None if there's insufficient data.

    Args:
        question_topics: {question_id: topic_string}
        wrong_question_ids: list of question IDs answered incorrectly

    Example:
        question_topics = {1: "Algebra", 2: "Algebra", 3: "Geometry"}
        wrong_question_ids = [1, 2]
        → returns "Algebra"
    """
    if not wrong_question_ids:
        return None

    topic_counts: dict[str, int] = {}
    for qid in wrong_question_ids:
        topic = question_topics.get(qid)
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    if not topic_counts:
        return None

    # Return the topic with the highest error count
    return max(topic_counts, key=lambda t: topic_counts[t])


def compute_performance_band(accuracy: float) -> str:
    """
    Maps an accuracy percentage to a human-readable performance band.
    Used for display purposes in the frontend.

    Returns: "Excellent" | "Good" | "Fair" | "Needs Improvement"
    """
    if accuracy >= 85.0:
        return "Excellent"
    elif accuracy >= 70.0:
        return "Good"
    elif accuracy >= 50.0:
        return "Fair"
    else:
        return "Needs Improvement"


def classify_topic_status(accuracy: float, total_attempted: int, min_attempts: int) -> str:
    """
    Classifies a single topic/lesson's mastery status for the per-subject
    topic breakdown in GET /analytics/me.

    `total_attempted < min_attempts` always wins regardless of accuracy —
    too few data points to draw a reliable conclusion either way.

    Boundaries (accuracy in percent, 0-100):
      - insufficient_data: total_attempted < min_attempts
      - weak:               accuracy < 41   (i.e. <= ~40.99)
      - developing:         41 <= accuracy < 70   (i.e. 41 to 69.99)
      - strong:             accuracy >= 70

    Note: the spec phrases the boundaries as "weak: <= 40" and
    "developing: 41 to 69.99", which leaves an undefined gap between 40 and
    41 for non-integer accuracy values (e.g. 40.5%, possible with certain
    attempt counts). Using "< 41" for the weak/developing boundary closes
    that gap while matching the given ranges for every value they explicitly
    define.
    """
    if total_attempted < min_attempts:
        return "insufficient_data"
    if accuracy >= 70.0:
        return "strong"
    if accuracy >= 41.0:
        return "developing"
    return "weak"


def is_valid_response_time(response_time: float | None, max_seconds: float) -> bool:
    """
    A response time is valid for analytics purposes if it's present,
    non-negative, and not absurdly large (e.g. a client bug, or a device that
    slept mid-question). Raw QuestionAttempt.response_time rows are NEVER
    modified because of this — this predicate only decides what gets counted
    at calculation time (see analytics_service.get_user_analytics()).
    """
    return response_time is not None and 0 <= response_time <= max_seconds


def compute_median_and_stddev(values: list[float]) -> tuple[float, float]:
    """
    Returns (median, sample_stddev), both rounded to 3 decimal places.

    Sample standard deviation (matching Postgres's stddev_samp) needs at least
    2 values; returns 0.0 for 0 or 1 values rather than raising.

    Computed here in plain Python, rather than via SQL percentile_cont/
    stddev_samp, because SQLite (used by this project's test suite) has no
    equivalent built-in functions at all. A Postgres-only SQL implementation
    would mean the tested code path and the production code path are never
    actually the same code — exactly the kind of divergence that has already
    caused real bugs in this codebase (see the Decimal-vs-float notes in
    analytics_service.py). Values are already validity-filtered and bounded to
    one user's own attempts before reaching this function, so the in-memory
    computation stays cheap.
    """
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
    """
    Classifies a scope's (overall/subject/topic) answering behavior.

    "fast" is always relative to the user's OVERALL median response time —
    never a per-subject or per-topic median — so e.g. a subject is judged
    against the user's normal pace, not against itself.

    "balanced": when avg_response_time falls within `balanced_ratio` of the
    overall median (e.g. 0.15 = within +/-15%), timing is neither meaningfully
    fast nor slow, regardless of accuracy — this is the one value not
    otherwise reachable from the fast/accurate boolean combinations below.
    """
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
    """
    Monday 00:00:00 of the ISO calendar week containing `dt`.
    `dt` must already be UTC-aware — this does no timezone conversion of its
    own, it only truncates to the start of that week in whatever zone `dt`
    is expressed in (callers pass UTC datetimes throughout this codebase).
    """
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
    """
    Compares a scope's (overall/subject/topic) most recent activity against
    the activity immediately before it, using WEIGHTED accuracy (total
    correct / total attempted across every session in a period) — never an
    average of each session's own percentage, which would let a 2-question
    session count as much as a 20-question one.

    `session_stats`: {session_id: {"correct": int, "total": int}} — only
    graded attempts relevant to this scope (e.g. just one topic's attempts,
    for a topic-level trend), from COMPLETED sessions only.
    `session_completed_at`: {session_id: datetime} — UTC, for every session
    appearing in `session_stats`.

    Method selection:
      - "recent_sessions": used when at least 2x `window_size` sessions
        exist for this scope — the latest `window_size` sessions (by
        completed_at) vs the `window_size` immediately before them.
      - "weekly": used otherwise — the current ISO calendar week (Monday
        00:00 UTC through `now`) vs the previous one (UTC).
      - "insufficient_data": if either resulting period ends up with fewer
        than `min_attempts_per_period` graded attempts, regardless of which
        method was attempted — a trend can't be judged reliably on that
        little data.
    """
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

    # Round once, from the raw (unrounded) fractions, so accuracy_change
    # can't drift from current_period_accuracy - previous_period_accuracy
    # due to independently rounding each side first.
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
    """
    Stats for ONE repeated-question group (all attempts sharing the same
    Question.question_fingerprint), already ordered chronologically.

    `ordered_corrects`: one bool per attempt, oldest first — MUST have at
    least 2 entries; callers must have already dropped groups with only a
    single attempt ("ignore questions attempted only once" — a lone attempt
    has no "previous attempt" to compare against).

    - repeated_question_count: every attempt after the first in this group.
    - repeated_correct_count / repeated_incorrect_count: how many of those
      repeated attempts were themselves correct/incorrect (always sums back
      to repeated_question_count).
    - corrected_previous_mistakes: a later CORRECT attempt whose immediately
      preceding attempt was incorrect.
    - repeated_same_mistakes: a later INCORRECT attempt whose immediately
      preceding attempt was also incorrect.

    A correct->anything transition isn't a "mistake" event either way, so it
    contributes to repeated_question_count/repeated_correct_count but not to
    corrected_previous_mistakes/repeated_same_mistakes.
    """
    repeated = ordered_corrects[1:]
    repeated_correct_count = sum(1 for c in repeated if c)
    repeated_incorrect_count = len(repeated) - repeated_correct_count

    corrected_previous_mistakes = 0
    repeated_same_mistakes = 0
    for prev_correct, cur_correct in zip(ordered_corrects, ordered_corrects[1:]):
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
    """
    Sums per-group stats (from compute_repeated_question_group_stats) across
    every qualifying fingerprint group in a scope (overall/subject/topic),
    and derives mistake_correction_rate from the totals.
    """
    totals = {
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
