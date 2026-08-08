"""
tests/test_growth_service.py
────────────────────────────────
Unit tests for app/services/growth_service.py's formula, independent of the
database — every function under test is pure (no DB access, no async), so
these run as plain synchronous tests with no fixtures at all.

See tests/test_analytics_growth.py for integration tests exercising the same
formula wired up end-to-end through GET /analytics/me.
"""
from datetime import date

from app.services import growth_service

DEFAULT_WEIGHTS = dict(
    effort_weight=0.20,
    consistency_weight=0.25,
    improvement_weight=0.40,
    mastery_weight=0.15,
)
DEFAULT_LEVEL_THRESHOLDS = dict(
    growing_threshold=40.0, strong_threshold=70.0, exceptional_threshold=85.0,
)
NEUTRAL = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_to_target / compute_rate_score
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_to_target_below_at_and_above():
    assert growth_service._normalize_to_target(10, 20) == 50.0
    assert growth_service._normalize_to_target(20, 20) == 100.0
    # Exceeding the target caps at 100 rather than overshooting.
    assert growth_service._normalize_to_target(40, 20) == 100.0


def test_compute_rate_score_zero_denominator_is_none():
    assert growth_service.compute_rate_score(0, 0) is None


def test_compute_rate_score_normal_case():
    assert growth_service.compute_rate_score(3, 4) == 75.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_session_spacing_score
# ─────────────────────────────────────────────────────────────────────────────

def test_session_spacing_neutral_below_three_dates():
    score = growth_service.compute_session_spacing_score(
        [date(2026, 1, 1), date(2026, 1, 3)], neutral_score=NEUTRAL,
    )
    assert score == NEUTRAL


def test_session_spacing_perfectly_regular_scores_100():
    dates = [date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 5), date(2026, 1, 7)]
    score = growth_service.compute_session_spacing_score(dates, neutral_score=NEUTRAL)
    assert score == 100.0


def test_session_spacing_irregular_scores_lower_than_regular():
    regular = growth_service.compute_session_spacing_score(
        [date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 5), date(2026, 1, 7)],
        neutral_score=NEUTRAL,
    )
    irregular = growth_service.compute_session_spacing_score(
        [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 15), date(2026, 1, 16)],
        neutral_score=NEUTRAL,
    )
    assert irregular < regular


# ─────────────────────────────────────────────────────────────────────────────
# compute_effort_score
# ─────────────────────────────────────────────────────────────────────────────

def test_effort_score_full_marks_at_or_above_targets():
    score, components = growth_service.compute_effort_score(
        completed_quiz_count=20, attempted_question_count=150, active_learning_days=30,
        completion_rate=100.0, weak_topic_attempts=10, has_weak_topics=True,
        window_days=30, target_quizzes=20, target_questions=150,
        target_weak_topic_attempts=10, neutral_score=NEUTRAL,
    )
    assert score == 100.0
    assert all(v == 100.0 for v in components.values())


def test_effort_score_weak_topic_component_neutral_when_no_weak_topics():
    _, components = growth_service.compute_effort_score(
        completed_quiz_count=5, attempted_question_count=50, active_learning_days=10,
        completion_rate=80.0, weak_topic_attempts=0, has_weak_topics=False,
        window_days=30, target_quizzes=20, target_questions=150,
        target_weak_topic_attempts=10, neutral_score=NEUTRAL,
    )
    assert components["weak_topic_attempts_score"] == NEUTRAL


def test_effort_score_not_inflated_by_empty_sessions_low_completion_rate():
    """
    Anti-gaming: rapidly creating 10 empty sessions alongside 1 real
    completed quiz drags completion_rate down to ~9% — effort_score must
    reflect that, not read as if 11 sessions of "activity" happened.
    """
    completion_rate = growth_service.compute_rate_score(1, 11)  # 1 completed of 11 total
    score, components = growth_service.compute_effort_score(
        completed_quiz_count=1, attempted_question_count=3, active_learning_days=1,
        completion_rate=completion_rate, weak_topic_attempts=0, has_weak_topics=False,
        window_days=30, target_quizzes=20, target_questions=150,
        target_weak_topic_attempts=10, neutral_score=NEUTRAL,
    )
    assert components["completion_rate_score"] < 10.0
    assert score < 30.0  # low across the board, not propped up by raw session count


# ─────────────────────────────────────────────────────────────────────────────
# compute_consistency_score
# ─────────────────────────────────────────────────────────────────────────────

def test_consistency_score_uses_neutral_for_missing_rates():
    # active_learning_days=0 legitimately scores 0 (there's no "unknown" case
    # for a plain count) — only completion_rate/abandonment_free_rate, which
    # are None here (no sessions to compute a rate from), fall back to neutral.
    score, components = growth_service.compute_consistency_score(
        active_learning_days=0, window_days=30, session_spacing_score=NEUTRAL,
        completion_rate=None, score_stability_score=NEUTRAL, abandonment_free_rate=None,
        neutral_score=NEUTRAL,
    )
    assert components["active_days_score"] == 0.0
    assert components["completion_rate_score"] == NEUTRAL
    assert components["low_abandonment_score"] == NEUTRAL
    # (0 + 50 + 50 + 50 + 50) / 5 = 40.0
    assert score == 40.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_improvement_score
# ─────────────────────────────────────────────────────────────────────────────

def test_improvement_score_accuracy_change_mapping():
    # A 0-point change maps to the neutral midpoint (50).
    _, flat = growth_service.compute_improvement_score(
        accuracy_change=0.0, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, neutral_score=NEUTRAL,
    )
    assert flat["recent_accuracy_change_score"] == 50.0

    # +50 points of improvement maps to the maximum (100), clamped.
    _, up = growth_service.compute_improvement_score(
        accuracy_change=50.0, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, neutral_score=NEUTRAL,
    )
    assert up["recent_accuracy_change_score"] == 100.0

    # -70 points would go below 0 without clamping.
    _, down = growth_service.compute_improvement_score(
        accuracy_change=-70.0, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, neutral_score=NEUTRAL,
    )
    assert down["recent_accuracy_change_score"] == 0.0


def test_improvement_score_topic_ratio_ignores_stable_topics():
    _, components = growth_service.compute_improvement_score(
        accuracy_change=None, improving_topic_count=3, declining_topic_count=1,
        repeated_mistake_correction_rate=None, neutral_score=NEUTRAL,
    )
    assert components["topic_improvement_score"] == 75.0  # 3 / (3+1) * 100


def test_improvement_score_neutral_when_no_directional_topics():
    _, components = growth_service.compute_improvement_score(
        accuracy_change=None, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, neutral_score=NEUTRAL,
    )
    assert components["topic_improvement_score"] == NEUTRAL
    assert components["recent_accuracy_change_score"] == NEUTRAL
    assert components["repeated_mistake_correction_score"] == NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# compute_growth_score / classify_growth_level
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_growth_score_weighted_formula():
    score = growth_service.compute_growth_score(
        effort_score=72, consistency_score=65, improvement_score=80, mastery_score=55,
        **DEFAULT_WEIGHTS,
    )
    # 80*.40 + 65*.25 + 72*.20 + 55*.15 = 32 + 16.25 + 14.4 + 8.25 = 70.9
    assert score == 70.9


def test_compute_growth_score_clamps_out_of_range_component():
    over = growth_service.compute_growth_score(
        effort_score=150, consistency_score=65, improvement_score=80, mastery_score=55,
        **DEFAULT_WEIGHTS,
    )
    capped = growth_service.compute_growth_score(
        effort_score=100, consistency_score=65, improvement_score=80, mastery_score=55,
        **DEFAULT_WEIGHTS,
    )
    assert over == capped


def test_classify_growth_level_boundaries():
    assert growth_service.classify_growth_level(0.0, **DEFAULT_LEVEL_THRESHOLDS) == "starting"
    assert growth_service.classify_growth_level(39.99, **DEFAULT_LEVEL_THRESHOLDS) == "starting"
    assert growth_service.classify_growth_level(40.0, **DEFAULT_LEVEL_THRESHOLDS) == "growing"
    assert growth_service.classify_growth_level(69.99, **DEFAULT_LEVEL_THRESHOLDS) == "growing"
    assert growth_service.classify_growth_level(70.0, **DEFAULT_LEVEL_THRESHOLDS) == "strong_growth"
    assert growth_service.classify_growth_level(84.99, **DEFAULT_LEVEL_THRESHOLDS) == "strong_growth"
    assert growth_service.classify_growth_level(85.0, **DEFAULT_LEVEL_THRESHOLDS) == "exceptional_growth"
    assert growth_service.classify_growth_level(100.0, **DEFAULT_LEVEL_THRESHOLDS) == "exceptional_growth"


# ─────────────────────────────────────────────────────────────────────────────
# build_growth_analytics (full orchestration)
# ─────────────────────────────────────────────────────────────────────────────

FULL_KWARGS = dict(
    min_attempts_in_window=3,
    window_days=30,
    target_quizzes=20,
    target_questions=150,
    target_weak_topic_attempts=10,
    neutral_score=NEUTRAL,
    **DEFAULT_WEIGHTS,
    **DEFAULT_LEVEL_THRESHOLDS,
)


def test_build_growth_analytics_insufficient_data_below_min_attempts():
    result = growth_service.build_growth_analytics(
        total_attempts_in_window=2,
        completed_quiz_count=0, attempted_question_count=2, active_learning_days=1,
        completion_rate=None, weak_topic_attempts=0, has_weak_topics=False,
        active_dates=[date(2026, 1, 1)], abandonment_free_rate=None,
        session_accuracies=[], consistency_min_sessions=3,
        accuracy_change=None, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, mastery_score=50.0,
        **FULL_KWARGS,
    )
    assert result == {
        "effort_score": None,
        "consistency_score": None,
        "improvement_score": None,
        "mastery_score": None,
        "growth_score": None,
        "growth_level": "insufficient_data",
        "components": None,
    }


def test_build_growth_analytics_zero_meaningful_activity_is_insufficient():
    """
    Anti-gaming, end to end: many rapidly created EMPTY sessions (zero
    graded attempts) must never produce a real growth score — the gate is
    on meaningful attempts, which is exactly zero here regardless of how
    many sessions exist.
    """
    result = growth_service.build_growth_analytics(
        total_attempts_in_window=0,
        completed_quiz_count=0, attempted_question_count=0, active_learning_days=0,
        completion_rate=growth_service.compute_rate_score(0, 15),  # 15 empty sessions, 0 completed
        weak_topic_attempts=0, has_weak_topics=False,
        active_dates=[], abandonment_free_rate=growth_service.compute_rate_score(0, 15),
        session_accuracies=[], consistency_min_sessions=3,
        accuracy_change=None, improving_topic_count=0, declining_topic_count=0,
        repeated_mistake_correction_rate=None, mastery_score=50.0,
        **FULL_KWARGS,
    )
    assert result["growth_level"] == "insufficient_data"
    assert result["growth_score"] is None


def test_build_growth_analytics_full_computation_is_internally_consistent():
    result = growth_service.build_growth_analytics(
        total_attempts_in_window=40,
        completed_quiz_count=8, attempted_question_count=40, active_learning_days=12,
        completion_rate=80.0, weak_topic_attempts=4, has_weak_topics=True,
        active_dates=[date(2026, 1, d) for d in (1, 4, 7, 10, 13, 16)],
        abandonment_free_rate=90.0,
        session_accuracies=[60.0, 70.0, 65.0, 75.0, 68.0],
        consistency_min_sessions=3,
        accuracy_change=10.0, improving_topic_count=2, declining_topic_count=1,
        repeated_mistake_correction_rate=66.0, mastery_score=55.0,
        **FULL_KWARGS,
    )
    assert result["growth_level"] != "insufficient_data"
    assert result["components"] is not None

    # The response's own growth_score must be reproducible from its own
    # effort/consistency/improvement/mastery scores — the breakdown is a
    # faithful explanation, not decoration.
    expected = growth_service.compute_growth_score(
        result["effort_score"], result["consistency_score"],
        result["improvement_score"], result["mastery_score"],
        **DEFAULT_WEIGHTS,
    )
    assert result["growth_score"] == expected
    assert result["growth_level"] == growth_service.classify_growth_level(
        result["growth_score"], **DEFAULT_LEVEL_THRESHOLDS,
    )
