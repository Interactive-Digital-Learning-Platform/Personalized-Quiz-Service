import statistics
from datetime import date

from app.services.mastery_service import compute_consistency_score as _score_stability


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _normalize_to_target(raw_count: int, target: int) -> float:
    if target <= 0:
        return 0.0
    return _clamp(raw_count / target * 100.0)


def compute_rate_score(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return _clamp(numerator / denominator * 100.0)


def compute_session_spacing_score(active_dates: list[date], *, neutral_score: float) -> float:
    # 100 minus how erratically spaced the gaps between active-study days
    # are — evenly spaced study days score near 100, bursty/random spacing
    # scores lower. Needs at least 3 distinct active dates to say anything;
    # below that, neutral_score is used instead.
    if len(active_dates) < 3:
        return neutral_score
    ordered = sorted(set(active_dates))
    if len(ordered) < 3:
        return neutral_score
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:])]
    mean_gap = statistics.mean(gaps)
    if mean_gap == 0:
        return 100.0
    stdev_gap = statistics.stdev(gaps)
    return round(_clamp(100.0 - (stdev_gap / mean_gap * 100.0)), 2)


def compute_effort_score(
    *,
    completed_quiz_count: int,
    attempted_question_count: int,
    active_learning_days: int,
    completion_rate: float | None,
    weak_topic_attempts: int,
    has_weak_topics: bool,
    window_days: int,
    target_quizzes: int,
    target_questions: int,
    target_weak_topic_attempts: int,
    neutral_score: float,
) -> tuple[float, dict[str, float]]:
    # Equal-weighted average of 5 components. Deliberately ignores accuracy
    # entirely — someone who shows up and genuinely attempts questions scores
    # well here even before they're accurate. Each count is normalized
    # against a target and capped at 100, so raw volume alone can't inflate
    # the score forever. weak_topic_attempts_score is neutral (not zero) when
    # there are no weak topics right now — nothing to practice isn't a lack
    # of effort.
    components = {
        "completed_quiz_count_score": round(_normalize_to_target(completed_quiz_count, target_quizzes), 2),
        "attempted_question_count_score": round(
            _normalize_to_target(attempted_question_count, target_questions), 2
        ),
        "active_learning_days_score": round(_clamp(active_learning_days / window_days * 100.0), 2),
        "completion_rate_score": (
            neutral_score if completion_rate is None else round(_clamp(completion_rate), 2)
        ),
        "weak_topic_attempts_score": round(
            neutral_score if not has_weak_topics
            else _normalize_to_target(weak_topic_attempts, target_weak_topic_attempts),
            2,
        ),
    }
    effort_score = _clamp(round(statistics.mean(components.values()), 2))
    return effort_score, components


def compute_consistency_score(
    *,
    active_learning_days: int,
    window_days: int,
    session_spacing_score: float,
    completion_rate: float | None,
    score_stability_score: float,
    abandonment_free_rate: float | None,
    neutral_score: float,
) -> tuple[float, dict[str, float]]:
    components = {
        "active_days_score": round(_clamp(active_learning_days / window_days * 100.0), 2),
        "session_spacing_score": round(_clamp(session_spacing_score), 2),
        "completion_rate_score": (
            neutral_score if completion_rate is None else round(_clamp(completion_rate), 2)
        ),
        "score_stability_score": round(_clamp(score_stability_score), 2),
        "low_abandonment_score": (
            neutral_score if abandonment_free_rate is None else round(_clamp(abandonment_free_rate), 2)
        ),
    }
    consistency_score = _clamp(round(statistics.mean(components.values()), 2))
    return consistency_score, components


def compute_improvement_score(
    *,
    accuracy_change: float | None,
    improving_topic_count: int,
    declining_topic_count: int,
    repeated_mistake_correction_rate: float | None,
    neutral_score: float,
) -> tuple[float, dict[str, float]]:
    # recent_accuracy_change_score maps a +/- percentage-point change onto
    # 0-100 around a neutral midpoint of 50 (no change = 50, +50pts = 100,
    # -50pts = 0). topic_improvement_score only counts topics with a clear
    # improving/declining direction — "stable" ones don't move the ratio
    # either way.
    recent_accuracy_change_score = (
        neutral_score if accuracy_change is None else round(_clamp(50.0 + accuracy_change), 2)
    )
    total_directional = improving_topic_count + declining_topic_count
    topic_improvement_score = (
        neutral_score if total_directional == 0
        else round(_clamp(improving_topic_count / total_directional * 100.0), 2)
    )
    repeated_mistake_correction_score = (
        neutral_score if repeated_mistake_correction_rate is None
        else round(_clamp(repeated_mistake_correction_rate), 2)
    )
    components = {
        "recent_accuracy_change_score": recent_accuracy_change_score,
        "topic_improvement_score": topic_improvement_score,
        "repeated_mistake_correction_score": repeated_mistake_correction_score,
    }
    improvement_score = _clamp(round(statistics.mean(components.values()), 2))
    return improvement_score, components


def compute_growth_score(
    effort_score: float,
    consistency_score: float,
    improvement_score: float,
    mastery_score: float,
    *,
    effort_weight: float,
    consistency_weight: float,
    improvement_weight: float,
    mastery_weight: float,
) -> float:
    # mastery_score only carries a small (default 0.15) weight here on
    # purpose — this metric is meant to reward the process (effort, showing
    # up, improving), not let raw current performance dominate it.
    weighted_total = (
        _clamp(improvement_score) * improvement_weight
        + _clamp(consistency_score) * consistency_weight
        + _clamp(effort_score) * effort_weight
        + _clamp(mastery_score) * mastery_weight
    )
    return _clamp(round(weighted_total, 2))


def classify_growth_level(
    score: float,
    *,
    growing_threshold: float,
    strong_threshold: float,
    exceptional_threshold: float,
) -> str:
    if score < growing_threshold:
        return "starting"
    if score < strong_threshold:
        return "growing"
    if score < exceptional_threshold:
        return "strong_growth"
    return "exceptional_growth"


def build_growth_analytics(
    *,
    total_attempts_in_window: int,
    min_attempts_in_window: int,
    completed_quiz_count: int,
    attempted_question_count: int,
    active_learning_days: int,
    completion_rate: float | None,
    weak_topic_attempts: int,
    has_weak_topics: bool,
    active_dates: list[date],
    abandonment_free_rate: float | None,
    session_accuracies: list[float],
    consistency_min_sessions: int,
    accuracy_change: float | None,
    improving_topic_count: int,
    declining_topic_count: int,
    repeated_mistake_correction_rate: float | None,
    mastery_score: float,
    window_days: int,
    target_quizzes: int,
    target_questions: int,
    target_weak_topic_attempts: int,
    neutral_score: float,
    effort_weight: float,
    consistency_weight: float,
    improvement_weight: float,
    mastery_weight: float,
    growing_threshold: float,
    strong_threshold: float,
    exceptional_threshold: float,
) -> dict:
    # Gated on activity within the rolling window (not all-time history),
    # since effort/consistency are both scoped to that same window — with no
    # recent activity, growth can't be judged honestly no matter how much
    # history exists further back.
    if total_attempts_in_window < min_attempts_in_window:
        return {
            "effort_score": None,
            "consistency_score": None,
            "improvement_score": None,
            "mastery_score": None,
            "growth_score": None,
            "growth_level": "insufficient_data",
            "components": None,
        }

    session_spacing_score = compute_session_spacing_score(active_dates, neutral_score=neutral_score)
    score_stability_score = _score_stability(
        session_accuracies, min_sessions=consistency_min_sessions, neutral_score=neutral_score,
    )

    effort_score, effort_components = compute_effort_score(
        completed_quiz_count=completed_quiz_count,
        attempted_question_count=attempted_question_count,
        active_learning_days=active_learning_days,
        completion_rate=completion_rate,
        weak_topic_attempts=weak_topic_attempts,
        has_weak_topics=has_weak_topics,
        window_days=window_days,
        target_quizzes=target_quizzes,
        target_questions=target_questions,
        target_weak_topic_attempts=target_weak_topic_attempts,
        neutral_score=neutral_score,
    )
    consistency_score, consistency_components = compute_consistency_score(
        active_learning_days=active_learning_days,
        window_days=window_days,
        session_spacing_score=session_spacing_score,
        completion_rate=completion_rate,
        score_stability_score=score_stability_score,
        abandonment_free_rate=abandonment_free_rate,
        neutral_score=neutral_score,
    )
    improvement_score, improvement_components = compute_improvement_score(
        accuracy_change=accuracy_change,
        improving_topic_count=improving_topic_count,
        declining_topic_count=declining_topic_count,
        repeated_mistake_correction_rate=repeated_mistake_correction_rate,
        neutral_score=neutral_score,
    )

    mastery_score_clamped = _clamp(mastery_score)
    growth_score = compute_growth_score(
        effort_score, consistency_score, improvement_score, mastery_score_clamped,
        effort_weight=effort_weight,
        consistency_weight=consistency_weight,
        improvement_weight=improvement_weight,
        mastery_weight=mastery_weight,
    )
    growth_level = classify_growth_level(
        growth_score,
        growing_threshold=growing_threshold,
        strong_threshold=strong_threshold,
        exceptional_threshold=exceptional_threshold,
    )

    return {
        "effort_score": effort_score,
        "consistency_score": consistency_score,
        "improvement_score": improvement_score,
        "mastery_score": mastery_score_clamped,
        "growth_score": growth_score,
        "growth_level": growth_level,
        "components": {
            "effort": effort_components,
            "consistency": consistency_components,
            "improvement": improvement_components,
        },
    }
