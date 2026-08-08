"""
tests/test_mastery_service.py
────────────────────────────────
Unit tests for app/services/mastery_service.py's formula, independent of
the database — every function under test is pure (no DB access, no async),
so these run as plain synchronous tests with no fixtures at all.

See tests/test_analytics_mastery.py for integration tests exercising the
same formula wired up end-to-end through GET /analytics/me.
"""
from app.services import mastery_service


# ─────────────────────────────────────────────────────────────────────────────
# compute_mastery_score
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = dict(
    accuracy_weight=0.35,
    recent_performance_weight=0.25,
    difficulty_weight=0.15,
    retention_weight=0.15,
    consistency_weight=0.10,
)


def test_compute_mastery_score_weighted_formula():
    # NOTE: the task's own example (accuracy=60, recent=70, difficulty=50,
    # retention=75, consistency=55 -> mastery_score=64.5) doesn't actually
    # match its own stated weights/formula: 60*.35 + 70*.25 + 50*.15 +
    # 75*.15 + 55*.10 = 62.75, not 64.5. This test uses the formula exactly
    # as specified in prose and asserts the arithmetically correct result,
    # rather than reproducing the example's own inconsistent arithmetic.
    score = mastery_service.compute_mastery_score(60, 70, 50, 75, 55, **DEFAULT_WEIGHTS)
    assert score == 62.75


def test_compute_mastery_score_all_100_is_100():
    score = mastery_service.compute_mastery_score(100, 100, 100, 100, 100, **DEFAULT_WEIGHTS)
    assert score == 100.0


def test_compute_mastery_score_all_zero_is_zero():
    score = mastery_service.compute_mastery_score(0, 0, 0, 0, 0, **DEFAULT_WEIGHTS)
    assert score == 0.0


def test_compute_mastery_score_clamps_out_of_range_components():
    # A component above 100 or below 0 must not be able to drag the result
    # outside 0-100, or skew it beyond what an honest 0-100 input could.
    over = mastery_service.compute_mastery_score(150, 70, 50, 75, 55, **DEFAULT_WEIGHTS)
    within_range = mastery_service.compute_mastery_score(100, 70, 50, 75, 55, **DEFAULT_WEIGHTS)
    assert over == within_range  # 150 clamps down to 100, same as passing 100 directly

    under = mastery_service.compute_mastery_score(-50, 70, 50, 75, 55, **DEFAULT_WEIGHTS)
    at_zero = mastery_service.compute_mastery_score(0, 70, 50, 75, 55, **DEFAULT_WEIGHTS)
    assert under == at_zero


# ─────────────────────────────────────────────────────────────────────────────
# classify_mastery_level
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = dict(developing_threshold=40.0, proficient_threshold=70.0, advanced_threshold=85.0)


def test_classify_mastery_level_boundaries():
    assert mastery_service.classify_mastery_level(0.0, **THRESHOLDS) == "beginner"
    assert mastery_service.classify_mastery_level(39.99, **THRESHOLDS) == "beginner"
    assert mastery_service.classify_mastery_level(40.0, **THRESHOLDS) == "developing"
    assert mastery_service.classify_mastery_level(69.99, **THRESHOLDS) == "developing"
    assert mastery_service.classify_mastery_level(70.0, **THRESHOLDS) == "proficient"
    assert mastery_service.classify_mastery_level(84.99, **THRESHOLDS) == "proficient"
    assert mastery_service.classify_mastery_level(85.0, **THRESHOLDS) == "advanced"
    assert mastery_service.classify_mastery_level(100.0, **THRESHOLDS) == "advanced"


# ─────────────────────────────────────────────────────────────────────────────
# compute_difficulty_score
# ─────────────────────────────────────────────────────────────────────────────

BASE_SCORES = {"easy": 33.0, "medium": 66.0, "hard": 100.0}


def test_compute_difficulty_score_averages_base_and_performance():
    # hard base (100) + 80% performance at hard -> (100 + 80) / 2 = 90.0
    score = mastery_service.compute_difficulty_score(
        "hard", 80.0, base_scores=BASE_SCORES, neutral_score=50.0,
    )
    assert score == 90.0


def test_compute_difficulty_score_uses_neutral_when_no_performance_data():
    # easy base (33) + neutral (50) -> 41.5, when there's no accuracy yet at
    # the current difficulty (e.g. just promoted/demoted).
    score = mastery_service.compute_difficulty_score(
        "easy", None, base_scores=BASE_SCORES, neutral_score=50.0,
    )
    assert score == 41.5


# ─────────────────────────────────────────────────────────────────────────────
# compute_retention_score
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_retention_score_neutral_when_no_repeated_data():
    score = mastery_service.compute_retention_score(0, 0, 0.0, neutral_score=50.0)
    assert score == 50.0


def test_compute_retention_score_uses_real_rate_when_data_exists():
    # 3 corrected out of 4 total mistake-repeat events -> 75% rate.
    score = mastery_service.compute_retention_score(3, 1, 75.0, neutral_score=50.0)
    assert score == 75.0


def test_compute_retention_score_zero_rate_is_not_confused_with_no_data():
    # Real data exists (2 repeated-mistake events), but correction rate is
    # genuinely 0% — must NOT be swapped for the neutral fallback.
    score = mastery_service.compute_retention_score(0, 2, 0.0, neutral_score=50.0)
    assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_consistency_score
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_consistency_score_neutral_below_min_sessions():
    score = mastery_service.compute_consistency_score([80.0, 90.0], min_sessions=3, neutral_score=50.0)
    assert score == 50.0


def test_compute_consistency_score_perfectly_consistent_scores():
    score = mastery_service.compute_consistency_score(
        [80.0, 80.0, 80.0, 80.0], min_sessions=3, neutral_score=50.0,
    )
    assert score == 100.0  # zero variation -> 100 - 0 = 100


def test_compute_consistency_score_variable_scores_scored_lower():
    consistent = mastery_service.compute_consistency_score(
        [80.0, 80.0, 80.0], min_sessions=3, neutral_score=50.0,
    )
    variable = mastery_service.compute_consistency_score(
        [20.0, 80.0, 40.0], min_sessions=3, neutral_score=50.0,
    )
    assert variable < consistent


# ─────────────────────────────────────────────────────────────────────────────
# build_mastery_analytics (full orchestration)
# ─────────────────────────────────────────────────────────────────────────────

FULL_KWARGS = dict(
    min_attempts=5,
    base_difficulty_scores=BASE_SCORES,
    neutral_score=50.0,
    consistency_min_sessions=3,
    **DEFAULT_WEIGHTS,
    **THRESHOLDS,
)


def test_build_mastery_analytics_insufficient_data_below_min_attempts():
    result = mastery_service.build_mastery_analytics(
        total_attempted=4,
        accuracy=90.0,
        recent_performance_score=90.0,
        current_difficulty="easy",
        accuracy_at_current_difficulty=90.0,
        corrected_previous_mistakes=1,
        repeated_same_mistakes=0,
        mistake_correction_rate=100.0,
        session_accuracies=[90.0, 90.0, 90.0],
        **FULL_KWARGS,
    )
    assert result == {
        "mastery_score": None,
        "mastery_level": "insufficient_data",
        "mastery_components": None,
    }


def test_build_mastery_analytics_full_computation():
    result = mastery_service.build_mastery_analytics(
        total_attempted=10,
        accuracy=60.0,
        recent_performance_score=70.0,
        current_difficulty="easy",   # base 33, performance-at-difficulty 67 -> difficulty_score 50.0
        accuracy_at_current_difficulty=67.0,
        corrected_previous_mistakes=3,
        repeated_same_mistakes=1,
        mistake_correction_rate=75.0,
        session_accuracies=[70.0, 80.0, 60.0],  # some variation, not zero
        **FULL_KWARGS,
    )
    assert result["mastery_score"] is not None
    assert result["mastery_level"] in {"beginner", "developing", "proficient", "advanced"}
    components = result["mastery_components"]
    assert components["accuracy_score"] == 60.0
    assert components["recent_performance_score"] == 70.0
    assert components["difficulty_score"] == 50.0
    assert components["retention_score"] == 75.0
    # Re-derive the expected final score directly from the formula to keep
    # this test honest about what it's actually verifying.
    expected = mastery_service.compute_mastery_score(
        components["accuracy_score"],
        components["recent_performance_score"],
        components["difficulty_score"],
        components["retention_score"],
        components["consistency_score"],
        **DEFAULT_WEIGHTS,
    )
    assert result["mastery_score"] == expected
