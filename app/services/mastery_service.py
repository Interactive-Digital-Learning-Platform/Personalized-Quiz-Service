"""
services/mastery_service.py
─────────────────────────────
A dedicated, self-contained home for the mastery-score formula used by
GET /analytics/me's subjects[] and subjects[].topics[] entries.

Every function here is a PURE function — no DB access, no async, no
dependency on any other service — so the formula itself can be unit-tested
completely independently of the database (see tests/test_mastery_service.py).
analytics_service.get_user_analytics() is responsible for gathering the raw
inputs (accuracy, recent-period accuracy, difficulty state, repeated-question
counts, per-session accuracies) and handing them to build_mastery_analytics()
here; this module never queries anything itself and never writes to
SubjectMastery or any other table — mastery_score is purely descriptive.

Formula (each component clamped to 0-100 before weighting, and the final
score clamped again after):

    mastery_score =
        accuracy_score            * ANALYTICS_MASTERY_ACCURACY_WEIGHT +
        recent_performance_score  * ANALYTICS_MASTERY_RECENT_PERFORMANCE_WEIGHT +
        difficulty_score          * ANALYTICS_MASTERY_DIFFICULTY_WEIGHT +
        retention_score           * ANALYTICS_MASTERY_RETENTION_WEIGHT +
        consistency_score         * ANALYTICS_MASTERY_CONSISTENCY_WEIGHT

All weights and thresholds are read from app.core.config.settings by the
caller and passed in explicitly — nothing in this module hardcodes them,
so they stay configurable in exactly one place.
"""
import statistics


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_mastery_score(
    accuracy_score: float,
    recent_performance_score: float,
    difficulty_score: float,
    retention_score: float,
    consistency_score: float,
    *,
    accuracy_weight: float,
    recent_performance_weight: float,
    difficulty_weight: float,
    retention_weight: float,
    consistency_weight: float,
) -> float:
    """
    The weighted formula itself. Every input is clamped to 0-100
    independently before weighting (a caller passing an out-of-range value
    can't skew the result beyond what a single component honestly allows),
    and the weighted sum is clamped again at the end as a final safety net.
    """
    weighted_total = (
        _clamp(accuracy_score) * accuracy_weight
        + _clamp(recent_performance_score) * recent_performance_weight
        + _clamp(difficulty_score) * difficulty_weight
        + _clamp(retention_score) * retention_weight
        + _clamp(consistency_score) * consistency_weight
    )
    return _clamp(round(weighted_total, 2))


def classify_mastery_level(
    score: float,
    *,
    developing_threshold: float,
    proficient_threshold: float,
    advanced_threshold: float,
) -> str:
    """
    "beginner" | "developing" | "proficient" | "advanced" — half-open
    intervals: [0, developing_threshold) -> beginner,
    [developing_threshold, proficient_threshold) -> developing,
    [proficient_threshold, advanced_threshold) -> proficient,
    [advanced_threshold, 100] -> advanced.
    """
    if score < developing_threshold:
        return "beginner"
    if score < proficient_threshold:
        return "developing"
    if score < advanced_threshold:
        return "proficient"
    return "advanced"


def compute_difficulty_score(
    current_difficulty: str,
    accuracy_at_current_difficulty: float | None,
    *,
    base_scores: dict[str, float],
    neutral_score: float,
) -> float:
    """
    Base score for the difficulty tier itself (see
    difficulty_service.DIFFICULTY_BASE_SCORES), adjusted by how well the
    student is actually performing at that tier — a simple average of the
    two, so a student parked at "hard" but scoring poorly there doesn't get
    full marks just for being on the hardest tier, and a student doing very
    well at "easy" isn't scored as low as the tier alone would suggest.

    `accuracy_at_current_difficulty` is None when the scope has no graded
    attempts at its own current difficulty yet (e.g. just promoted/demoted) —
    `neutral_score` stands in for that missing half of the average.
    """
    base = base_scores.get(current_difficulty, neutral_score)
    performance = neutral_score if accuracy_at_current_difficulty is None else accuracy_at_current_difficulty
    return _clamp((base + performance) / 2.0)


def compute_retention_score(
    corrected_previous_mistakes: int,
    repeated_same_mistakes: int,
    mistake_correction_rate: float,
    *,
    neutral_score: float,
) -> float:
    """
    Directly the repeated-question mistake_correction_rate (see
    scoring_service.aggregate_repeated_question_stats()) when any repeated-
    question data exists, else `neutral_score` — a rate of exactly 0.0 is
    ambiguous on its own (it's the same value used both for "every repeat
    was a repeated mistake" and "there were no repeats to measure at all"),
    so the counts are checked explicitly to tell those two cases apart.
    """
    if corrected_previous_mistakes + repeated_same_mistakes == 0:
        return neutral_score
    return _clamp(mistake_correction_rate)


def compute_consistency_score(
    session_accuracies: list[float],
    *,
    min_sessions: int,
    neutral_score: float,
) -> float:
    """
    100 minus the sample standard deviation of per-session accuracy
    percentages — a student whose completed-session scores barely vary
    scores near 100 here; one who swings wildly session to session scores
    lower. Requires at least `min_sessions` completed sessions with graded
    attempts to be meaningful; below that, `neutral_score` is used instead.
    """
    if len(session_accuracies) < min_sessions:
        return neutral_score
    stdev = statistics.stdev(session_accuracies)
    return _clamp(100.0 - stdev)


def build_mastery_analytics(
    *,
    total_attempted: int,
    accuracy: float,
    recent_performance_score: float,
    current_difficulty: str,
    accuracy_at_current_difficulty: float | None,
    corrected_previous_mistakes: int,
    repeated_same_mistakes: int,
    mistake_correction_rate: float,
    session_accuracies: list[float],
    min_attempts: int,
    base_difficulty_scores: dict[str, float],
    neutral_score: float,
    consistency_min_sessions: int,
    accuracy_weight: float,
    recent_performance_weight: float,
    difficulty_weight: float,
    retention_weight: float,
    consistency_weight: float,
    developing_threshold: float,
    proficient_threshold: float,
    advanced_threshold: float,
) -> dict:
    """
    Orchestrates the full mastery calculation for ONE scope (a subject or a
    topic) from already-gathered primitives — see analytics_service.
    get_user_analytics() for where each input comes from. Purely a
    computation: never touches the DB, never mutates SubjectMastery/
    LessonMastery, and has no knowledge of "subject" vs "topic" — the caller
    decides what scope's numbers to pass in.

    Returns:
        {
            "mastery_score": float | None,    # None if total_attempted < min_attempts
            "mastery_level": str,             # "insufficient_data" in that case
            "mastery_components": dict | None,
        }
    """
    if total_attempted < min_attempts:
        return {
            "mastery_score": None,
            "mastery_level": "insufficient_data",
            "mastery_components": None,
        }

    accuracy_score = _clamp(accuracy)
    recent_score = _clamp(recent_performance_score)
    difficulty_score = compute_difficulty_score(
        current_difficulty, accuracy_at_current_difficulty,
        base_scores=base_difficulty_scores, neutral_score=neutral_score,
    )
    retention_score = compute_retention_score(
        corrected_previous_mistakes, repeated_same_mistakes, mistake_correction_rate,
        neutral_score=neutral_score,
    )
    consistency_score = compute_consistency_score(
        session_accuracies, min_sessions=consistency_min_sessions, neutral_score=neutral_score,
    )

    mastery_score = compute_mastery_score(
        accuracy_score, recent_score, difficulty_score, retention_score, consistency_score,
        accuracy_weight=accuracy_weight,
        recent_performance_weight=recent_performance_weight,
        difficulty_weight=difficulty_weight,
        retention_weight=retention_weight,
        consistency_weight=consistency_weight,
    )
    mastery_level = classify_mastery_level(
        mastery_score,
        developing_threshold=developing_threshold,
        proficient_threshold=proficient_threshold,
        advanced_threshold=advanced_threshold,
    )

    return {
        "mastery_score": mastery_score,
        "mastery_level": mastery_level,
        "mastery_components": {
            "accuracy_score": accuracy_score,
            "recent_performance_score": recent_score,
            "difficulty_score": difficulty_score,
            "retention_score": retention_score,
            "consistency_score": consistency_score,
        },
    }
