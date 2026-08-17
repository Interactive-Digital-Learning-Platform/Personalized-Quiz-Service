"""Continuous Evidence-Weighted Mastery System — pure calculation functions.

This module holds only pure, deterministic math: no DB access. The
orchestration (reading/writing LessonMastery/SubjectMastery rows, querying
quiz history) lives in difficulty_service.py, which calls into these
functions. Kept as a separate module — rather than folded into
difficulty_service.py directly — purely to keep that file from growing
unwieldy; it is NOT the same thing as, and does not replace,
app/services/mastery_service.py, which is an unrelated analytics-only
scoring system (accuracy/recent-performance/difficulty/retention/consistency
-> mastery_score shown in GET /analytics/me) that never drives difficulty
and is left untouched by this feature.

All scores are 0-100 and clamped. See app/core/config.py's
ADAPTIVE_MASTERY_* settings for every tunable weight/threshold used below —
nothing here should hardcode a magic number that belongs there.
"""
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


# Moved here (from difficulty_service.py) as the canonical definition so
# this module has no dependency on difficulty_service.py, which itself
# depends on this module for orchestration — difficulty_service.py
# re-imports this name so `difficulty_service.DIFFICULTY_BASE_SCORES`
# (used by app/services/analytics/mastery_service.py) keeps working.
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

# Challenge-zone mixed-difficulty distribution for automatic quiz
# generation, keyed by which side of the two promotion thresholds the
# student's current subject mastery_score falls on. Deliberately reuses
# EASY_TO_MEDIUM_THRESHOLD/MEDIUM_TO_HARD_THRESHOLD as the band boundaries
# instead of inventing separate ones, so the "band" a student is in always
# matches the difficulty the promotion logic itself would consider them
# closest to.
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
    # Normal submissions always get full completion evidence. A timeout
    # is scored by how much of the quiz was actually answered rather than
    # zero, since a student who answered 8/10 before time ran out gave
    # real, usable evidence for the other 8.
    if ended_by != "timeout":
        return 100.0
    if intended_count <= 0:
        return 0.0
    return _clamp(answered_count / intended_count * 100.0)


def _retention_weight_for_gap(days_since_last_seen: float | None) -> float:
    b = settings
    if days_since_last_seen is None:
        # No timestamp data available for this repeat — documented
        # fallback: treat it as a short-to-moderate gap rather than
        # guessing at either extreme.
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
    """repeated_attempts: (was_correct_this_time, days_since_previously_seen)
    for each question in this quiz that's a repeat (by fingerprint) of one
    the student has answered before. Longer gaps answered correctly count
    as stronger evidence of real retention, not just short-term recall.

    Falls back to a neutral-positive score when there's no repeated-question
    data yet (e.g. a brand new student, or a quiz made entirely of
    never-before-seen questions) — this is evidence *absence*, not evidence
    of poor retention, so it must not read as a low score.
    """
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
    """The single-quiz evidence score fed into the gradual mastery update.
    75% correctness + 10% difficulty + 10% retention + 5% completion.

    The difficulty multiplier (easy=0.95/medium=1.00/hard=1.05) is applied
    to the correctness term specifically — not as a separate additive
    term — so that correctness on a harder question counts slightly more
    without ever outweighing correctness itself: at +/-5% it can move the
    75%-weighted term by at most +/-3.75 points out of 100.
    """
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


# calculate_lesson_evidence is intentionally the same formula as
# calculate_quiz_evidence — lesson-level and subject-level evidence differ
# only in *which* accuracy/retention/completion numbers are passed in, not
# in the math itself, so this is a thin alias rather than a near-duplicate
# implementation.
calculate_lesson_evidence = calculate_quiz_evidence


# ── Gradual update ───────────────────────────────────────────────────────

def calculate_new_evidence_weight(evidence_count: int) -> float:
    """How much weight this quiz's evidence gets in the blend. Below
    LOW_EVIDENCE_THRESHOLD, new evidence counts for more (up to
    LOW_EVIDENCE_MAX_NEW_WEIGHT at evidence_count=0) since a new student has
    no track record yet worth protecting with the standard 75/25 split.
    Linearly tapers down to the standard NEW_EVIDENCE_WEIGHT by the time
    evidence_count reaches the threshold.
    """
    b = settings
    if evidence_count >= b.ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD:
        return b.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT

    ratio = evidence_count / b.ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD
    return b.ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT - ratio * (
        b.ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT - b.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT
    )


def update_mastery_score(old_mastery: float, quiz_evidence: float, evidence_count_before: int) -> float:
    """new_mastery = old_mastery * old_weight + quiz_evidence * new_weight,
    clamped 0-100. evidence_count_before is the count *before* this quiz is
    added, so the very first quiz already gets the boosted new-evidence
    weight rather than the standard one.
    """
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
    """System's confidence in its own mastery estimate — NOT the student's
    confidence. Piecewise-linear through (0, 0) -> (LOW_EVIDENCE, LOW_SCORE)
    -> (MODERATE_EVIDENCE, MODERATE_SCORE) -> (HIGH_EVIDENCE, HIGH_SCORE),
    flat at HIGH_SCORE beyond that.
    """
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
    """Fluency is about response speed, not knowledge — a correct-but-slow
    answer still counts fully toward mastery (mastery never looks at
    response_time at all); fluency is a separate, purely informational
    score. A fast-but-wrong answer is capped low so guessing quickly can
    never read as fluent.
    """
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
    """session_history: (session_id, correct_count, total_count,
    completed_at) for a student's recent *qualifying* quizzes in one
    subject, most-recent-first order not required (sorted internally).
    Recent WINDOW_SIZE quizzes vs. the WINDOW_SIZE before them; if there
    isn't enough history for two full windows, returns "insufficient_data"
    rather than falling back to a coarser (e.g. weekly) comparison, per
    spec. Reuses compute_performance_trend() (scoring_service.py) for the
    actual weighted-accuracy comparison rather than re-implementing it —
    that function already handles this generically for the analytics
    trend feature; this only adds the stricter insufficient-data gate the
    adaptive engine wants instead of that feature's weekly fallback.
    """
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

def rollup_lesson_mastery_to_subject(lesson_rows: Sequence[LessonMastery]) -> float | None:
    """Evidence-weighted average of a subject's lessons' mastery_score,
    using log1p(evidence_count) as the weight so one heavily-practiced
    lesson can't linearly dominate the subject average the way a raw
    evidence_count weighting would. Returns None (graceful fallback --
    caller should keep whatever subject-level value it already has) if no
    lesson has any evidence yet.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for row in lesson_rows:
        if row.evidence_count <= 0:
            continue
        weight = math.log1p(row.evidence_count)
        weighted_sum += row.mastery_score * weight
        weight_total += weight

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
    """Deterministic, hysteresis-based difficulty transition. Promote and
    demote thresholds are deliberately different per tier (see
    ADAPTIVE_MASTERY_*_THRESHOLD) so mastery hovering near one boundary
    doesn't bounce the difficulty back and forth quiz to quiz. Promotion
    additionally requires minimum evidence + qualifying-completion counts;
    demotion normally requires >=2 weak/declining results, but an
    "emergency demotion" bypasses that when mastery has fallen far enough
    (with enough evidence/confidence to trust the number) that leaving a
    struggling student on a too-hard tier would do more harm than an
    extra, unearned drop.
    """
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
    """Challenge-zone mixed-difficulty distribution and weak/moderate/strong
    lesson targeting split for automatic ("no explicit override") quiz
    generation. Purely descriptive percentages — converting these into
    actual per-question difficulty/lesson picks happens where question
    counts and the subject's LessonMastery rows are available (quiz
    generation orchestration), not here.
    """
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
    """Short, plain-language summary of a student's standing for the Groq
    prompt (see generate_questions()'s adaptive_context param) -- describes
    bands only, never raw internal scores/ids, per the "don't expose
    unnecessary internal DB details to the LLM" requirement.
    """
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


def select_preferred_lessons(lesson_mastery_scores: dict[str, float], *, max_lessons: int = 3) -> list[str]:
    """Picks the weakest lessons (lowest mastery_score) to nudge automatic
    quiz generation toward, approximating the weak/moderate/strong lesson
    targeting split without needing to force an exact per-question quota
    onto the AI's per-question-random-lesson generation -- see
    generate_questions()'s preferred_lessons param. Returns [] if there's
    no lesson data yet (a brand-new subject), which is a graceful no-op,
    not a penalty.
    """
    if not lesson_mastery_scores:
        return []
    weakest_first = sorted(lesson_mastery_scores.items(), key=lambda item: item[1])
    half = max(1, math.ceil(len(weakest_first) * WEAK_LESSON_TARGET_PERCENTAGE))
    return [lesson for lesson, _ in weakest_first[: min(half, max_lessons)]]


def allocate_question_counts(distribution: dict[str, float], total: int) -> dict[str, int]:
    """Converts a {key: percentage} distribution into integer counts that
    sum to exactly `total`, using the largest-remainder method (floor each
    share, then hand out the leftover questions one at a time to whichever
    keys had the largest fractional remainder) so rounding never drops or
    invents a question. Keys with a zero resulting count are omitted.
    """
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
