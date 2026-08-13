"""
tests/test_difficulty_mastery_engine.py
──────────────────────────────────────────
Unit tests for the pure Continuous Evidence-Weighted Mastery System
calculation functions in app/services/difficulty_mastery_engine.py — no DB,
no async, just the math. Integration-level behavior (DB orchestration,
idempotency, concurrency) is covered separately in
tests/test_difficulty_service_adaptive.py.
"""
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.services import difficulty_mastery_engine as engine


# ─────────────────────────────────────────────────────────────────────────────
# calculate_quiz_evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_quiz_evidence_exact_formula_medium_difficulty():
    # accuracy=80, medium multiplier=1.0, retention=70 (fallback), completion=100
    # -> 80*0.75 + 66*0.10 + 70*0.10 + 100*0.05 = 60 + 6.6 + 7 + 5 = 78.6
    evidence = engine.calculate_quiz_evidence(
        accuracy=80.0, difficulty="medium", retention_evidence=70.0, completion_evidence=100.0,
    )
    assert evidence == 78.6


def test_quiz_evidence_harder_difficulty_scores_slightly_higher_for_same_accuracy():
    easy = engine.calculate_quiz_evidence(accuracy=80.0, difficulty="easy", retention_evidence=70.0, completion_evidence=100.0)
    medium = engine.calculate_quiz_evidence(accuracy=80.0, difficulty="medium", retention_evidence=70.0, completion_evidence=100.0)
    hard = engine.calculate_quiz_evidence(accuracy=80.0, difficulty="hard", retention_evidence=70.0, completion_evidence=100.0)
    assert easy < medium < hard


def test_quiz_evidence_difficulty_multiplier_does_not_overpower_correctness():
    # A low-accuracy hard-difficulty quiz must still score well below a
    # high-accuracy easy-difficulty quiz -- the +/-5% multiplier only ever
    # nudges the 75%-weighted accuracy term, never dominates it.
    low_accuracy_hard = engine.calculate_quiz_evidence(accuracy=20.0, difficulty="hard", retention_evidence=70.0, completion_evidence=100.0)
    high_accuracy_easy = engine.calculate_quiz_evidence(accuracy=95.0, difficulty="easy", retention_evidence=70.0, completion_evidence=100.0)
    assert low_accuracy_hard < high_accuracy_easy


def test_quiz_evidence_clamped_to_0_100():
    assert engine.calculate_quiz_evidence(accuracy=0.0, difficulty="easy", retention_evidence=0.0, completion_evidence=0.0) >= 0.0
    assert engine.calculate_quiz_evidence(accuracy=100.0, difficulty="hard", retention_evidence=100.0, completion_evidence=100.0) <= 100.0


def test_calculate_lesson_evidence_is_the_same_formula_as_quiz_evidence():
    assert engine.calculate_lesson_evidence is engine.calculate_quiz_evidence


# ─────────────────────────────────────────────────────────────────────────────
# calculate_completion_evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_completion_evidence_normal_submission_is_always_100():
    assert engine.calculate_completion_evidence(ended_by="submitted", answered_count=1, intended_count=10) == 100.0


def test_completion_evidence_timeout_uses_answered_ratio_not_zero():
    result = engine.calculate_completion_evidence(ended_by="timeout", answered_count=6, intended_count=10)
    assert result == 60.0
    assert result > 0.0


def test_completion_evidence_timeout_zero_intended_count_does_not_crash():
    assert engine.calculate_completion_evidence(ended_by="timeout", answered_count=0, intended_count=0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_retention_evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_retention_evidence_no_repeats_falls_back_to_neutral_score():
    assert engine.calculate_retention_evidence([]) == settings.ADAPTIVE_MASTERY_RETENTION_FALLBACK_SCORE


def test_retention_evidence_all_correct_repeats_scores_100():
    assert engine.calculate_retention_evidence([(True, 20.0), (True, 5.0)]) == 100.0


def test_retention_evidence_all_incorrect_repeats_scores_0():
    assert engine.calculate_retention_evidence([(False, 20.0), (False, 2.0)]) == 0.0


def test_retention_evidence_longer_gap_correct_answer_weighs_more_than_short_gap():
    # One correct answer after a long gap should score higher than the same
    # single correct answer after a very short gap, given a wrong short-gap
    # answer alongside it in each case -- retention over time matters more.
    short_gap_mix = engine.calculate_retention_evidence([(True, 0.5), (False, 0.5)])
    long_gap_mix = engine.calculate_retention_evidence([(True, 20.0), (False, 20.0)])
    assert long_gap_mix == 50.0 and short_gap_mix == 50.0  # symmetric weights within a band cancel out
    # But moving the CORRECT answer to the longer-gap gap shifts the score up.
    correct_is_long_gap = engine.calculate_retention_evidence([(True, 20.0), (False, 0.5)])
    correct_is_short_gap = engine.calculate_retention_evidence([(True, 0.5), (False, 20.0)])
    assert correct_is_long_gap > correct_is_short_gap


def test_retention_evidence_missing_timestamp_uses_documented_fallback_band():
    # None (no timestamp data) should behave like the 1-3 day band.
    with_none = engine.calculate_retention_evidence([(True, None)])
    with_1_to_3 = engine.calculate_retention_evidence([(True, 2.0)])
    assert with_none == with_1_to_3 == 100.0  # both single all-correct -> 100 regardless of weight


# ─────────────────────────────────────────────────────────────────────────────
# gradual update (calculate_new_evidence_weight / update_mastery_score)
# ─────────────────────────────────────────────────────────────────────────────

def test_new_evidence_weight_at_zero_evidence_is_the_boosted_maximum():
    assert engine.calculate_new_evidence_weight(0) == settings.ADAPTIVE_MASTERY_LOW_EVIDENCE_MAX_NEW_WEIGHT


def test_new_evidence_weight_at_or_above_threshold_is_the_standard_weight():
    assert engine.calculate_new_evidence_weight(settings.ADAPTIVE_MASTERY_LOW_EVIDENCE_THRESHOLD) == settings.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT
    assert engine.calculate_new_evidence_weight(999) == settings.ADAPTIVE_MASTERY_NEW_EVIDENCE_WEIGHT


def test_update_mastery_score_blends_old_and_new():
    # evidence_count_before >= LOW_EVIDENCE_THRESHOLD -> standard 75/25 blend.
    result = engine.update_mastery_score(old_mastery=50.0, quiz_evidence=90.0, evidence_count_before=50)
    assert result == 50.0 * 0.75 + 90.0 * 0.25


def test_update_mastery_score_clamped_to_0_100():
    assert engine.update_mastery_score(old_mastery=98.0, quiz_evidence=100.0, evidence_count_before=50) <= 100.0
    assert engine.update_mastery_score(old_mastery=2.0, quiz_evidence=0.0, evidence_count_before=50) >= 0.0


def test_one_bad_quiz_does_not_swing_an_established_mastery_score_drastically():
    # A single 0% quiz from an established mastery of 80 should NOT crater
    # the score to near-zero -- one bad quiz must not swing everything.
    new_score = engine.update_mastery_score(old_mastery=80.0, quiz_evidence=0.0, evidence_count_before=50)
    assert new_score > 50.0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_zero_evidence_is_zero():
    assert engine.calculate_confidence(0) == 0.0


def test_confidence_at_breakpoints_matches_configured_scores():
    assert engine.calculate_confidence(settings.ADAPTIVE_MASTERY_CONFIDENCE_LOW_EVIDENCE) == settings.ADAPTIVE_MASTERY_CONFIDENCE_LOW_SCORE
    assert engine.calculate_confidence(settings.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_EVIDENCE) == settings.ADAPTIVE_MASTERY_CONFIDENCE_MODERATE_SCORE
    assert engine.calculate_confidence(settings.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_EVIDENCE) == settings.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE


def test_confidence_beyond_high_evidence_stays_capped():
    assert engine.calculate_confidence(10_000) == settings.ADAPTIVE_MASTERY_CONFIDENCE_HIGH_SCORE


def test_confidence_is_monotonically_increasing_with_evidence():
    values = [engine.calculate_confidence(n) for n in (0, 3, 5, 10, 15, 22, 30, 50)]
    assert values == sorted(values)


# ─────────────────────────────────────────────────────────────────────────────
# fluency
# ─────────────────────────────────────────────────────────────────────────────

def test_fluency_correct_and_fast_scores_high():
    score = engine.calculate_fluency_for_question(difficulty="medium", response_time=5.0, correct=True)
    assert score >= 90.0


def test_fluency_correct_but_slow_still_scores_above_the_incorrect_cap():
    # Correct-but-slow must still count toward mastery (mastery formula
    # never reads response_time at all) and fluency itself shouldn't be
    # punished as harshly as an outright wrong answer.
    score = engine.calculate_fluency_for_question(difficulty="medium", response_time=100.0, correct=True)
    assert score >= settings.ADAPTIVE_MASTERY_FLUENCY_MIN_SCORE


def test_fluency_fast_but_incorrect_is_capped_low():
    score = engine.calculate_fluency_for_question(difficulty="medium", response_time=1.0, correct=False)
    assert score <= settings.ADAPTIVE_MASTERY_FLUENCY_INCORRECT_CAP


def test_fluency_aggregate_empty_defaults_neutral():
    assert engine.calculate_fluency([]) == 50.0


def test_fluency_aggregate_averages_question_scores():
    assert engine.calculate_fluency([20.0, 100.0]) == 60.0


# ─────────────────────────────────────────────────────────────────────────────
# recency weighting
# ─────────────────────────────────────────────────────────────────────────────

def test_recency_weight_bands():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    assert engine.get_recency_weight(now - timedelta(days=1), now=now) == settings.ADAPTIVE_MASTERY_RECENCY_0_TO_7_DAYS
    assert engine.get_recency_weight(now - timedelta(days=10), now=now) == settings.ADAPTIVE_MASTERY_RECENCY_8_TO_14_DAYS
    assert engine.get_recency_weight(now - timedelta(days=20), now=now) == settings.ADAPTIVE_MASTERY_RECENCY_15_TO_30_DAYS
    assert engine.get_recency_weight(now - timedelta(days=45), now=now) == settings.ADAPTIVE_MASTERY_RECENCY_31_TO_60_DAYS
    assert engine.get_recency_weight(now - timedelta(days=90), now=now) == settings.ADAPTIVE_MASTERY_RECENCY_OVER_60_DAYS


def test_calculate_recency_weight_is_an_alias_of_get_recency_weight():
    assert engine.calculate_recency_weight is engine.get_recency_weight


# ─────────────────────────────────────────────────────────────────────────────
# trend detection
# ─────────────────────────────────────────────────────────────────────────────

def _history(*accuracies: float, start: datetime) -> list[tuple[int, int, int, datetime]]:
    # (session_id, correct, total, completed_at) — 10 questions each, most
    # recent last for readability; calculate_trend sorts internally.
    return [
        (i, round(acc / 10), 10, start + timedelta(days=i))
        for i, acc in enumerate(accuracies)
    ]


def test_trend_insufficient_data_below_two_windows():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = _history(80, 80, 80, 80, start=start)  # 4 < 2*3
    result = engine.calculate_trend(history)
    assert result.trend_label == "insufficient_data"
    assert result.trend_score is None


def test_trend_improving_with_enough_history():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # previous 3 at 50%, recent 3 at 90% -> clearly improving.
    history = _history(50, 50, 50, 90, 90, 90, start=start)
    result = engine.calculate_trend(history)
    assert result.trend_label == "improving"
    assert result.trend_score >= settings.ADAPTIVE_MASTERY_TREND_IMPROVING_THRESHOLD


def test_trend_declining_with_enough_history():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = _history(90, 90, 90, 50, 50, 50, start=start)
    result = engine.calculate_trend(history)
    assert result.trend_label == "declining"


def test_trend_stable_with_enough_history():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = _history(80, 80, 80, 80, 80, 80, start=start)
    result = engine.calculate_trend(history)
    assert result.trend_label == "stable"


# ─────────────────────────────────────────────────────────────────────────────
# lesson -> subject roll-up
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLessonMastery:
    def __init__(self, mastery_score: float, evidence_count: int):
        self.mastery_score = mastery_score
        self.evidence_count = evidence_count


def test_rollup_empty_lessons_returns_none():
    assert engine.rollup_lesson_mastery_to_subject([]) is None


def test_rollup_ignores_lessons_with_zero_evidence():
    rows = [_FakeLessonMastery(90.0, 0), _FakeLessonMastery(60.0, 10)]
    assert engine.rollup_lesson_mastery_to_subject(rows) == 60.0


def test_rollup_high_evidence_lesson_does_not_linearly_dominate():
    # A lesson with 10x the evidence of another should NOT get 10x the
    # weight (log-capped), so the low-evidence lesson still meaningfully
    # pulls the average away from the high-evidence lesson's own score.
    rows = [_FakeLessonMastery(90.0, 100), _FakeLessonMastery(10.0, 5)]
    result = engine.rollup_lesson_mastery_to_subject(rows)
    linear_weighted_average = (90.0 * 100 + 10.0 * 5) / (100 + 5)  # ~85.7 if uncapped
    assert result < linear_weighted_average


# ─────────────────────────────────────────────────────────────────────────────
# difficulty transitions (hysteresis + minimum evidence)
# ─────────────────────────────────────────────────────────────────────────────

def test_promotion_blocked_when_only_mastery_threshold_met():
    result = engine.determine_difficulty_transition(
        current_difficulty="easy", mastery_score=70.0, evidence_count=0, confidence_score=0.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=0,
    )
    assert result == "easy"


def test_promotion_blocked_when_only_evidence_count_met():
    result = engine.determine_difficulty_transition(
        current_difficulty="easy", mastery_score=50.0, evidence_count=100, confidence_score=100.0,
        qualifying_completions_at_current_tier=100, recent_weak_results=0,
    )
    assert result == "easy"


def test_promotion_succeeds_when_all_gates_met():
    result = engine.determine_difficulty_transition(
        current_difficulty="easy", mastery_score=65.0, evidence_count=15, confidence_score=50.0,
        qualifying_completions_at_current_tier=2, recent_weak_results=0,
    )
    assert result == "medium"


def test_medium_to_hard_promotion_uses_its_own_higher_thresholds():
    # Would satisfy easy->medium's gates but not medium->hard's stricter ones.
    result = engine.determine_difficulty_transition(
        current_difficulty="medium", mastery_score=70.0, evidence_count=15, confidence_score=50.0,
        qualifying_completions_at_current_tier=2, recent_weak_results=0,
    )
    assert result == "medium"


def test_demotion_requires_both_mastery_crossing_and_two_weak_results():
    # Mastery below threshold but only 1 weak result -> no demotion yet.
    result = engine.determine_difficulty_transition(
        current_difficulty="medium", mastery_score=35.0, evidence_count=10, confidence_score=10.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=1,
    )
    assert result == "medium"


def test_demotion_succeeds_with_mastery_and_two_weak_results():
    result = engine.determine_difficulty_transition(
        current_difficulty="medium", mastery_score=35.0, evidence_count=10, confidence_score=10.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=2,
    )
    assert result == "easy"


def test_emergency_demotion_bypasses_weak_result_count_when_mastery_collapses():
    result = engine.determine_difficulty_transition(
        current_difficulty="medium",
        mastery_score=settings.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_THRESHOLD - 1.0,
        evidence_count=settings.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_EVIDENCE_COUNT,
        confidence_score=settings.ADAPTIVE_MASTERY_EMERGENCY_DEMOTION_MIN_CONFIDENCE,
        qualifying_completions_at_current_tier=0, recent_weak_results=0,
    )
    assert result == "easy"


def test_emergency_demotion_requires_sufficient_evidence_and_confidence():
    # Mastery is low enough, but evidence/confidence too low to trust it.
    result = engine.determine_difficulty_transition(
        current_difficulty="medium", mastery_score=10.0, evidence_count=1, confidence_score=0.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=0,
    )
    assert result == "medium"


def test_easy_tier_never_demotes_further():
    result = engine.determine_difficulty_transition(
        current_difficulty="easy", mastery_score=0.0, evidence_count=100, confidence_score=100.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=100,
    )
    assert result == "easy"


def test_hard_tier_never_promotes_further():
    result = engine.determine_difficulty_transition(
        current_difficulty="hard", mastery_score=100.0, evidence_count=1000, confidence_score=100.0,
        qualifying_completions_at_current_tier=1000, recent_weak_results=0,
    )
    assert result == "hard"


def test_hard_to_medium_demotion():
    result = engine.determine_difficulty_transition(
        current_difficulty="hard", mastery_score=50.0, evidence_count=30, confidence_score=80.0,
        qualifying_completions_at_current_tier=0, recent_weak_results=2,
    )
    assert result == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# challenge-zone generation profile
# ─────────────────────────────────────────────────────────────────────────────

def test_generation_profile_low_mastery_favors_easy():
    profile = engine.get_adaptive_generation_profile(30.0)
    assert profile.difficulty_distribution["easy"] > profile.difficulty_distribution["hard"]


def test_generation_profile_high_mastery_favors_hard():
    profile = engine.get_adaptive_generation_profile(90.0)
    assert profile.difficulty_distribution["hard"] > profile.difficulty_distribution["easy"]


def test_generation_profile_lesson_targeting_sums_to_one():
    profile = engine.get_adaptive_generation_profile(50.0)
    assert round(sum(profile.lesson_targeting.values()), 6) == 1.0


def test_allocate_question_counts_sums_to_total_with_no_rounding_loss():
    distribution = {"easy": 0.70, "medium": 0.25, "hard": 0.05}
    for total in (1, 2, 3, 5, 7, 10, 13, 20):
        counts = engine.allocate_question_counts(distribution, total)
        assert sum(counts.values()) == total


def test_allocate_question_counts_omits_zero_count_tiers():
    counts = engine.allocate_question_counts({"easy": 0.70, "medium": 0.25, "hard": 0.05}, 1)
    assert "hard" not in counts or counts.get("hard") == 0
    assert sum(counts.values()) == 1


def test_allocate_question_counts_handles_zero_total():
    assert engine.allocate_question_counts({"easy": 1.0}, 0) == {}


def test_select_preferred_lessons_empty_when_no_data():
    assert engine.select_preferred_lessons({}) == []


def test_select_preferred_lessons_picks_weakest():
    scores = {"Algebra": 90.0, "Geometry": 20.0, "Trig": 55.0, "Calculus": 95.0}
    preferred = engine.select_preferred_lessons(scores, max_lessons=1)
    assert preferred == ["Geometry"]


def test_select_preferred_lessons_respects_max_lessons_cap():
    scores = {f"Lesson{i}": float(i) for i in range(10)}
    preferred = engine.select_preferred_lessons(scores, max_lessons=2)
    assert len(preferred) <= 2
