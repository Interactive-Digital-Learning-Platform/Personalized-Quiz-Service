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
    # Weighted sum of the 5 components below, each clamped to 0-100 first so
    # one out-of-range input can't skew the result beyond what it honestly
    # should — then clamped again at the end as a final safety net.
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
    # Averages the tier's base score with how well the student actually
    # performs at that tier, so being parked at "hard" but doing badly there
    # doesn't score full marks just for the tier, and doing great at "easy"
    # isn't penalized down to the tier's low base score.
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
    # A correction rate of exactly 0.0 is ambiguous on its own (could mean
    # "every repeat was still wrong" or "there were no repeats at all"), so
    # the counts are checked explicitly to tell those two cases apart.
    if corrected_previous_mistakes + repeated_same_mistakes == 0:
        return neutral_score
    return _clamp(mistake_correction_rate)


def compute_consistency_score(
    session_accuracies: list[float],
    *,
    min_sessions: int,
    neutral_score: float,
) -> float:
    # 100 minus the standard deviation of per-session accuracy — scores that
    # barely vary session to session land near 100, wild swings score lower.
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
    # Runs the full mastery calculation for one subject or topic from
    # already-gathered numbers — never touches the DB, has no idea whether
    # it's scoring a subject or a topic, just crunches whatever it's given.
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
