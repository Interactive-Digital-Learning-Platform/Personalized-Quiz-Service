# Continuous Evidence-Weighted Mastery System — pure calculation functions only (no DB access); orchestration lives in difficulty_service.py. Distinct from the unrelated analytics-only app/services/mastery_service.py. All scores are 0-100 and clamped; tunable weights/thresholds live in app/core/config.py's ADAPTIVE_MASTERY_* settings.
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.core.config import settings
from app.services.scoring_service import compute_performance_trend

if TYPE_CHECKING:
    from app.models.lesson_mastery import LessonMastery


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# Canonical definition, moved here so this module has no dependency on difficulty_service.py, which re-imports this name so difficulty_service.DIFFICULTY_BASE_SCORES keeps working.
DIFFICULTY_BASE_SCORES = {"easy": 33.0, "medium": 66.0, "hard": 100.0}


DIFFICULTY_EVIDENCE_MULTIPLIERS: dict[str, float] = {
    "easy": settings.ADAPTIVE_MASTERY_EASY_DIFFICULTY_MULTIPLIER,
    "medium": settings.ADAPTIVE_MASTERY_MEDIUM_DIFFICULTY_MULTIPLIER,
    "hard": settings.ADAPTIVE_MASTERY_HARD_DIFFICULTY_MULTIPLIER,
}

FLUENCY_BASELINE_SECONDS: dict[str, float] = {
    "easy": settings.ADAPTIVE_MASTERY_FLUENCY_EASY_BASELINE_SECONDS,
    "medium": settings.ADAPTIVE_MASTERY_FLUENCY_MEDIUM_BASELINE_SECONDS,
    "hard": settings.ADAPTIVE_MASTERY_FLUENCY_HARD_BASELINE_SECONDS,
}

# Challenge-zone mixed-difficulty distribution for automatic quiz generation, keyed by which side of the two promotion thresholds the student's mastery_score falls on (reuses the promotion thresholds as band boundaries).
CHALLENGE_ZONE_DISTRIBUTION: dict[str, dict[str, float]] = {
    "easy_range": {"easy": 0.70, "medium": 0.25, "hard": 0.05},
    "medium_range": {"easy": 0.25, "medium": 0.55, "hard": 0.20},
    "hard_range": {"easy": 0.10, "medium": 0.35, "hard": 0.55},
}

WEAK_LESSON_TARGET_PERCENTAGE = 0.50
MODERATE_LESSON_TARGET_PERCENTAGE = 0.30
STRONG_LESSON_TARGET_PERCENTAGE = 0.20


# ── Quiz evidence ────────────────────────────────────────────────────────

def calculate_completion_evidence(
    *, ended_by: str, answered_count: int, intended_count: int
) -> float:
    # Normal submissions get full completion evidence; a timeout is scored by how much of the quiz was actually answered rather than zero.
    if ended_by != "timeout":
        return 100.0
    if intended_count <= 0:
        return 0.0
    return _clamp(answered_count / intended_count * 100.0)


def _retention_weight_for_gap(days_since_last_seen: float | None) -> float:
    b = settings
    if days_since_last_seen is None:
        # No timestamp data available for this repeat — fall back to a short-to-moderate gap rather than guessing at either extreme.
        return b.ADAPTIVE_MASTERY_RETENTION_1_TO_3_DAYS_WEIGHT
    if days_since_last_seen < 1:
        return b.ADAPTIVE_MASTERY_RETENTION_UNDER_1_DAY_WEIGHT
    if days_since_last_seen <= 3:
        return b.ADAPTIVE_MASTERY_RETENTION_1_TO_3_DAYS_WEIGHT
    if days_since_last_seen <= 14:
        return b.ADAPTIVE_MASTERY_RETENTION_4_TO_14_DAYS_WEIGHT
    return b.ADAPTIVE_MASTERY_RETENTION_OVER_14_DAYS_WEIGHT


def calculate_retention_evidence(
    repeated_attempts: Sequence[tuple[bool, float | None]],
) -> float:
    # repeated_attempts: (was_correct_this_time, days_since_previously_seen) for repeat questions; longer gaps answered correctly count as stronger retention evidence, and no repeated-question data yet falls back to a neutral-positive score (evidence absence, not poor retention).
    if not repeated_attempts:
        return settings.ADAPTIVE_MASTERY_RETENTION_FALLBACK_SCORE

    weighted_sum = 0.0
    total_weight = 0.0
    for was_correct, days_since in repeated_attempts:
        weight = _retention_weight_for_gap(days_since)
        weighted_sum += (100.0 if was_correct else 0.0) * weight
        total_weight += weight

    if total_weight == 0:
        return settings.ADAPTIVE_MASTERY_RETENTION_FALLBACK_SCORE
    return _clamp(weighted_sum / total_weight)


def calculate_quiz_evidence(
    *,
    accuracy: float,
    difficulty: str,
    retention_evidence: float,
    completion_evidence: float,
) -> float:
    # The single-quiz evidence score: 75% correctness + 10% difficulty + 10% retention + 5% completion, with the difficulty multiplier applied to the correctness term so a harder question counts slightly more without outweighing correctness itself.
    multiplier = DIFFICULTY_EVIDENCE_MULTIPLIERS.get(difficulty, 1.0)
    accuracy_component = _clamp(accuracy * multiplier)
    difficulty_component = DIFFICULTY_BASE_SCORES.get(difficulty, DIFFICULTY_BASE_SCORES["medium"])

    evidence = (
        accuracy_component * settings.ADAPTIVE_MASTERY_ACCURACY_WEIGHT
        + difficulty_component * settings.ADAPTIVE_MASTERY_DIFFICULTY_WEIGHT
        + retention_evidence * settings.ADAPTIVE_MASTERY_RETENTION_WEIGHT
        + completion_evidence * settings.ADAPTIVE_MASTERY_COMPLETION_WEIGHT
    )
    return _clamp(evidence)


# Intentionally the same formula as calculate_quiz_evidence — lesson vs subject only differ in which numbers are passed in, so this is a thin alias.
calculate_lesson_evidence = calculate_quiz_evidence


# ── Gradual update ───────────────────────────────────────────────────────

def calculate_new_evidence_weight(evidence_count: int) -> float:
    # How much weight this quiz's evidence gets in the blend; below LOW_EVIDENCE_THRESHOLD new evidence counts for more, linearly tapering down to the standard NEW_EVIDENCE_WEIGHT by the threshold.
    b = settings
    if evidence_count >= b.ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD:
        return b.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT

    ratio = evidence_count / b.ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD
    return b.ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT - ratio * (
        b.ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT - b.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT
    )


def update_mastery_score(old_mastery: float, quiz_evidence: float, evidence_count_before: int) -> float:
    # new_mastery = old_mastery * old_weight + quiz_evidence * new_weight, clamped 0-100; evidence_count_before is the count *before* this quiz so the first quiz gets the boosted new-evidence weight.
    new_weight = calculate_new_evidence_weight(evidence_count_before)
    old_weight = 1.0 - new_weight
    return _clamp(old_mastery * old_weight + quiz_evidence * new_weight)


# ── Confidence ───────────────────────────────────────────────────────────

def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return y1
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def calculate_confidence(evidence_count: int) -> float:
    # System's confidence in its own mastery estimate (not the student's) — piecewise-linear through the LOW/MODERATE/HIGH evidence breakpoints, flat beyond HIGH_EVIDENCE.
    b = settings
    if evidence_count <= 0:
        return 0.0
    if evidence_count >= b.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_EVIDENCE:
        return b.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE
    if evidence_count <= b.ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE:
        return _clamp(_lerp(
            evidence_count, 0, 0.0,
            b.ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE, b.ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE,
        ))
    if evidence_count <= b.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE:
        return _clamp(_lerp(
            evidence_count,
            b.ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE, b.ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE,
            b.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE, b.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE,
        ))
    return _clamp(_lerp(
        evidence_count,
        b.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE, b.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE,
        b.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_EVIDENCE, b.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE,
    ))


# ── Fluency ──────────────────────────────────────────────────────────────

def calculate_fluency_for_question(*, difficulty: str, response_time: float, correct: bool) -> float:
    # Fluency is about response speed, not knowledge — mastery never looks at response_time; a fast-but-wrong answer is capped low so guessing quickly can't read as fluent.
    b = settings
    baseline = FLUENCY_BASELINE_SECONDS.get(difficulty, FLUENCY_BASELINE_SECONDS["medium"])
    safe_response_time = max(response_time, 0.01)
    ratio = min(baseline / safe_response_time, b.ADAPTIVE_MASTERY_FLUENCY_RATIO_CAP)
    raw = ratio * 50.0

    if not correct:
        return min(raw, b.ADAPTIVE_MASTERY_FLUENCY_INCORRECT_CAP)
    return _clamp(raw, b.ADAPTIVE_MASTERY_FLUENCY_MIN_SCORE, b.ADAPTIVE_MASTERY_FLUENCY_MAX_SCORE)


def calculate_fluency(question_fluencies: list[float]) -> float:
    if not question_fluencies:
        return 50.0
    return _clamp(sum(question_fluencies) / len(question_fluencies))


# ── Recency weighting (historical/trend analysis) ───────────────────────

def get_recency_weight(timestamp: datetime, *, now: datetime | None = None) -> float:
    b = settings
    now = now or datetime.now(UTC)
    days = (now - timestamp).total_seconds() / 86400.0
    if days <= 7:
        return b.ADAPTIVE_MASTERY_RECENCY_0_TO_7_DAYS
    if days <= 14:
        return b.ADAPTIVE_MASTERY_RECENCY_8_TO_14_DAYS
    if days <= 30:
        return b.ADAPTIVE_MASTERY_RECENCY_15_TO_30_DAYS
    if days <= 60:
        return b.ADAPTIVE_MASTERY_RECENCY_31_TO_60_DAYS
    return b.ADAPTIVE_MASTERY_RECENCY_OVER_60_DAYS


# Alias matching the suggested-architecture naming from the spec.
calculate_recency_weight = get_recency_weight


# ── Trend detection ──────────────────────────────────────────────────────

@dataclass
class TrendResult:
    recent_accuracy: float | None
    previous_accuracy: float | None
    trend_score: float | None
    trend_label: str


def calculate_trend(
    session_history: list[tuple[int, int, int, datetime]],
) -> TrendResult:
    # session_history: (session_id, correct_count, total_count, completed_at) for recent qualifying quizzes in one subject; compares recent WINDOW_SIZE quizzes vs. the WINDOW_SIZE before, returning "insufficient_data" instead of a weekly fallback if there isn't enough history, reusing compute_performance_trend() for the actual comparison.
    window = settings.ADAPTIVE_MASTERY_TREND_WINDOW_SIZE
    if len(session_history) < 2 * window:
        return TrendResult(None, None, None, "insufficient_data")

    session_stats = {sid: {"correct": correct, "total": total} for sid, correct, total, _ in session_history}
    session_completed_at = {sid: completed_at for sid, _, _, completed_at in session_history}

    result = compute_performance_trend(
        session_stats,
        session_completed_at,
        window_size=window,
        min_attempts_per_period=1,
        change_threshold=settings.ADAPTIVE_MASTERY_TREND_IMPROVING_THRESHOLD,
        now=datetime.now(UTC),
    )
    return TrendResult(
        recent_accuracy=result["current_period_accuracy"],
        previous_accuracy=result["previous_period_accuracy"],
        trend_score=result["accuracy_change"],
        trend_label=result["trend"],
    )


# ── Lesson -> subject roll-up ────────────────────────────────────────────

def rollup_lesson_mastery_to_subject(
    lesson_rows: Sequence[LessonMastery],
    *,
    all_curriculum_lessons: list[str] | None = None,
    untested_lesson_default: float = 50.0,
) -> float | None:
    # Evidence-weighted average of a subject's lessons' mastery_score using log1p(evidence_count) as the weight so one heavily-practiced lesson can't dominate; returns None if no lesson has evidence and all_curriculum_lessons is None. `all_curriculum_lessons` makes this coverage-aware by folding in untested curriculum lessons at `untested_lesson_default`, so grinding one lesson can't drive up the whole subject average unchecked.
    weighted_sum = 0.0
    weight_total = 0.0
    tested_lesson_names: set[str] = set()
    for row in lesson_rows:
        if row.evidence_count <= 0:
            continue
        weight = math.log1p(row.evidence_count)
        weighted_sum += row.mastery_score * weight
        weight_total += weight
        tested_lesson_names.add(row.lesson)

    if all_curriculum_lessons is not None:
        untested_weight = settings.ADAPTIVE_MASTERY_UNTESTED_LESSON_WEIGHT
        for lesson_name in all_curriculum_lessons:
            if lesson_name in tested_lesson_names:
                continue
            weighted_sum += untested_lesson_default * untested_weight
            weight_total += untested_weight

    if weight_total == 0:
        return None
    return _clamp(weighted_sum / weight_total)


# ── Difficulty transitions (hysteresis + minimum evidence) ──────────────

def _emergency_demotion_applies(mastery_score: float, evidence_count: int, confidence_score: float) -> bool:
    b = settings
    return (
        mastery_score < b.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_THRESHOLD
        and evidence_count >= b.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_EVIDENCE_COUNT
        and confidence_score >= b.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_CONFIDENCE
    )


def determine_difficulty_transition(
    *,
    current_difficulty: str,
    mastery_score: float,
    evidence_count: int,
    confidence_score: float,
    qualifying_completions_at_current_tier: int,
    recent_weak_results: int,
) -> str:
    # Deterministic, hysteresis-based difficulty transition — separate promote/demote thresholds per tier stop mastery near a boundary from bouncing back and forth; promotion needs minimum evidence + qualifying completions, demotion normally needs >=2 weak results, but an "emergency demotion" bypasses that when mastery has fallen far enough to trust the number.
    b = settings

    if current_difficulty == "easy":
        if (
            mastery_score >= b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_THRESHOLD
            and evidence_count >= b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_EVIDENCE_COUNT
            and qualifying_completions_at_current_tier >= b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_QUALIFYING_COMPLETIONS
        ):
            return "medium"
        return "easy"

    if current_difficulty == "medium":
        if (
            mastery_score >= b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_THRESHOLD
            and evidence_count >= b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_EVIDENCE_COUNT
            and qualifying_completions_at_current_tier >= b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_QUALIFYING_COMPLETIONS
        ):
            return "hard"
        if (
            mastery_score <= b.ADAPTIVE_MASTERY_MEDIUM_TO_EASY_THRESHOLD
            and recent_weak_results >= b.ADAPTIVE_MASTERY_DEMOTION_MIN_WEAK_RESULTS
        ):
            return "easy"
        if _emergency_demotion_applies(mastery_score, evidence_count, confidence_score):
            return "easy"
        return "medium"

    # current_difficulty == "hard"
    if (
        mastery_score <= b.ADAPTIVE_MASTERY_HARD_TO_MEDIUM_THRESHOLD
        and recent_weak_results >= b.ADAPTIVE_MASTERY_DEMOTION_MIN_WEAK_RESULTS
    ):
        return "medium"
    if _emergency_demotion_applies(mastery_score, evidence_count, confidence_score):
        return "medium"
    return "hard"


# ── Adaptive quiz generation profile ─────────────────────────────────────

@dataclass
class AdaptiveGenerationProfile:
    difficulty_distribution: dict[str, float] = field(default_factory=dict)
    lesson_targeting: dict[str, float] = field(default_factory=dict)


def _mastery_band(mastery_score: float) -> str:
    b = settings
    if mastery_score < b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_THRESHOLD:
        return "easy_range"
    if mastery_score < b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_THRESHOLD:
        return "medium_range"
    return "hard_range"


def get_adaptive_generation_profile(mastery_score: float) -> AdaptiveGenerationProfile:
    # Challenge-zone mixed-difficulty distribution and weak/moderate/strong lesson targeting split for automatic quiz generation — purely descriptive percentages; converting to actual per-question picks happens in quiz generation orchestration, not here.
    band = _mastery_band(mastery_score)
    return AdaptiveGenerationProfile(
        difficulty_distribution=dict(CHALLENGE_ZONE_DISTRIBUTION[band]),
        lesson_targeting={
            "weak": WEAK_LESSON_TARGET_PERCENTAGE,
            "moderate": MODERATE_LESSON_TARGET_PERCENTAGE,
            "strong": STRONG_LESSON_TARGET_PERCENTAGE,
        },
    )


def describe_adaptive_context(mastery_score: float, confidence_score: float, trend_label: str) -> str:
    # Short, plain-language summary of a student's standing for the Groq prompt — describes bands only, never raw internal scores/ids.
    band = _mastery_band(mastery_score).replace("_range", "")
    if confidence_score >= settings.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE:
        confidence_word = "high"
    elif confidence_score >= settings.ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE:
        confidence_word = "moderate"
    else:
        confidence_word = "still building"
    trend_phrase = {
        "improving": "recently improving",
        "declining": "recently struggling a bit more",
        "stable": "performing steadily",
        "insufficient_data": "still early in their practice history",
    }.get(trend_label, "still early in their practice history")
    return f"{band} mastery level, {confidence_word} confidence in that estimate, {trend_phrase}."


def blend_lesson_weakness_scores(
    cewm_scores: dict[str, float], bkt_scores: dict[str, float], *, bkt_weight: float,
) -> dict[str, float]:
    # Combines CEWM's mastery_score (0-100) with BKT's p_know (0-1) into one 0-100 "how well known" score per lesson; bkt_weight of 0.0 reproduces cewm_scores exactly, 1.0 uses BKT alone, and a lesson present in only one source uses that value directly.
    blended: dict[str, float] = {}
    for lesson in cewm_scores.keys() | bkt_scores.keys():
        cewm_norm = cewm_scores[lesson] / 100.0 if lesson in cewm_scores else None
        bkt_p_know = bkt_scores.get(lesson)

        if cewm_norm is not None and bkt_p_know is not None:
            combined = cewm_norm * (1 - bkt_weight) + bkt_p_know * bkt_weight
        elif cewm_norm is not None:
            combined = cewm_norm
        else:
            combined = bkt_p_know

        blended[lesson] = combined * 100.0
    return blended


def select_preferred_lessons(lesson_mastery_scores: dict[str, float], *, max_lessons: int = 3) -> list[str]:
    # Picks the weakest lessons (lowest mastery_score) to nudge automatic quiz generation toward, approximating the weak/moderate/strong split without forcing an exact per-question quota; returns [] (graceful no-op) if there's no lesson data yet.
    if not lesson_mastery_scores:
        return []
    weakest_first = sorted(lesson_mastery_scores.items(), key=lambda item: item[1])
    half = max(1, math.ceil(len(weakest_first) * WEAK_LESSON_TARGET_PERCENTAGE))
    return [lesson for lesson, _ in weakest_first[: min(half, max_lessons)]]


def allocate_question_counts(distribution: dict[str, float], total: int) -> dict[str, int]:
    # Converts a {key: percentage} distribution into integer counts summing to exactly `total`, using the largest-remainder method so rounding never drops or invents a question; keys with a zero count are omitted.
    if total <= 0 or not distribution:
        return {}

    raw_shares = {key: pct * total for key, pct in distribution.items()}
    counts = {key: int(share) for key, share in raw_shares.items()}
    remainder = total - sum(counts.values())

    if remainder > 0:
        by_fractional_part = sorted(
            distribution.keys(), key=lambda k: raw_shares[k] - counts[k], reverse=True
        )
        for key in by_fractional_part[:remainder]:
            counts[key] += 1

    return {key: count for key, count in counts.items() if count > 0}
