"""
services/growth_service.py
─────────────────────────────
A dedicated, self-contained home for the growth-analytics formula: effort,
consistency, improvement, and the combined growth score/level. Every
function here is PURE — no DB access, no async — mirroring
mastery_service.py's separation of formula from data-gathering, so the
formula itself can be unit-tested completely independently of the database
(see tests/test_growth_service.py). analytics_service.get_user_analytics()
gathers the raw inputs (30-day-windowed counts, per-session accuracies,
topic trend directions, etc.) and hands them to build_growth_analytics()
here. Nothing in this module writes to the database — growth is computed
fresh on every request, never persisted (see get_user_analytics()'s growth
section for why).

── Why this design avoids ranking students by raw score ──────────────────
The research goal behind this endpoint is to describe a student's
TRAJECTORY (are they putting in effort, showing up consistently, actually
improving?) rather than just their current performance level. Concretely:

- effort_score and consistency_score deliberately ignore accuracy/mastery
  entirely — a student who shows up regularly and genuinely attempts
  questions scores well here even if they're not yet accurate.
- Every count-based component (quizzes, questions, weak-topic attempts) is
  normalized against a configurable TARGET and capped at 100 — volume alone
  can't produce an unbounded score, and a student who blows past the target
  gets full marks, not a higher score than someone else who also hit it.
- mastery_score (the one place raw performance enters the formula) carries
  only a 0.15 weight in the final growth_score — high current mastery can
  contribute at most 15 of the 100 points, so it can't dominate a metric
  that's meant to reward process over outcome.
- Every raw count that could be gamed by rapidly creating empty sessions
  (completed_quiz_count, active_learning_days, session spacing) is derived
  ONLY from sessions with at least one graded attempt — an empty session
  with no answered questions contributes to nothing here except pulling
  completion_rate/abandonment down.

Growth formula (each of the 4 top-level inputs clamped to 0-100 before
weighting, and the final score clamped again after):

    growth_score =
        improvement_score * ANALYTICS_GROWTH_IMPROVEMENT_WEIGHT +
        consistency_score * ANALYTICS_GROWTH_CONSISTENCY_WEIGHT +
        effort_score      * ANALYTICS_GROWTH_EFFORT_WEIGHT +
        mastery_score     * ANALYTICS_GROWTH_MASTERY_WEIGHT

All weights/thresholds/targets are read from app.core.config.settings by
the caller and passed in explicitly — nothing here hardcodes them.
"""
import statistics
from datetime import date

# Reused for score_stability_score (consistency) — the exact same "100 minus
# stdev of per-session accuracies" idea already used for mastery's own
# consistency component, just fed this scope's (30-day-windowed) sessions.
from app.services.mastery_service import compute_consistency_score as _score_stability


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _normalize_to_target(raw_count: int, target: int) -> float:
    """
    raw_count/target*100, clamped at 100 — e.g. 15 quizzes against a target
    of 20 scores 75.0; exceeding the target caps at 100 rather than
    overshooting, so extra volume beyond "full effort" doesn't inflate the
    score further.
    """
    if target <= 0:
        return 0.0
    return _clamp(raw_count / target * 100.0)


def compute_rate_score(numerator: int, denominator: int) -> float | None:
    """
    A simple numerator/denominator*100 rate (used for completion_rate and
    the abandonment-derived "low abandonment" score) — None when there's no
    denominator to compute a rate from at all (caller substitutes a neutral
    score in that case).
    """
    if denominator == 0:
        return None
    return _clamp(numerator / denominator * 100.0)


def compute_session_spacing_score(active_dates: list[date], *, neutral_score: float) -> float:
    """
    100 minus the coefficient-of-variation-style spread of the gaps (in
    days) between consecutive active-learning dates — evenly spaced study
    days score near 100, bursty/erratic spacing scores lower.

    `active_dates` must contain only dates with MEANINGFUL activity (at
    least one graded attempt) — see get_user_analytics()'s growth section —
    so this can't be inflated by rapidly created, unanswered sessions.
    Needs at least 3 distinct active dates (2 gaps) to say anything about
    regularity; below that, `neutral_score` is used instead.
    """
    if len(active_dates) < 3:
        return neutral_score
    ordered = sorted(set(active_dates))
    if len(ordered) < 3:
        return neutral_score
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:])]
    mean_gap = statistics.mean(gaps)
    if mean_gap == 0:
        return 100.0  # multiple active days landing on the same calendar date, every time
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
    """
    Equal-weighted average of 5 components, each independently normalized
    to 0-100 — see module docstring for why counts are target-normalized
    rather than unbounded.

    `weak_topic_attempts_score` is neutral (not zero) when the student
    currently has NO weak topics at all — having nothing weak to practice
    isn't a lack of effort.
    """
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
    """Equal-weighted average of 5 components, each already 0-100."""
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
    """
    Equal-weighted average of 3 components:
    - recent_accuracy_change_score: a +/-percentage-point change mapped onto
      0-100 around a neutral midpoint of 50 (a 0-point change scores
      exactly neutral; +50 points scores 100; -50 points scores 0).
    - topic_improvement_score: the share of topics with a definitive
      direction ("improving" vs "declining") that are improving — topics
      that are merely "stable" or "insufficient_data" don't count toward
      either side of this ratio.
    - repeated_mistake_correction_score: directly the repeated-question
      mistake_correction_rate, when any repeated-question data exists.
    """
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
    """
    "starting" | "growing" | "strong_growth" | "exceptional_growth" —
    half-open intervals, same style as mastery_service.classify_mastery_level().
    """
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
    # effort inputs
    completed_quiz_count: int,
    attempted_question_count: int,
    active_learning_days: int,
    completion_rate: float | None,
    weak_topic_attempts: int,
    has_weak_topics: bool,
    # consistency inputs
    active_dates: list[date],
    abandonment_free_rate: float | None,
    session_accuracies: list[float],
    consistency_min_sessions: int,
    # improvement inputs
    accuracy_change: float | None,
    improving_topic_count: int,
    declining_topic_count: int,
    repeated_mistake_correction_rate: float | None,
    # mastery input (already computed elsewhere — see mastery_service.py)
    mastery_score: float,
    # config
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
    """
    Orchestrates the full growth calculation from already-gathered
    primitives — see analytics_service.get_user_analytics()'s growth
    section for where each input comes from. Purely a computation: never
    touches the DB, never persists anything.

    Gated on `total_attempts_in_window` (graded attempts within the rolling
    window — see ANALYTICS_GROWTH_WINDOW_DAYS) rather than all-time history,
    since effort_score and consistency_score are both scoped to that same
    window: with no recent meaningful activity, neither can be computed
    honestly, so the whole growth object is reported as insufficient data
    regardless of how much all-time history exists.
    """
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
