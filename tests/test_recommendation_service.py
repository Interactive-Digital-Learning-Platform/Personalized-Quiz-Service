"""
tests/test_recommendation_service.py
────────────────────────────────────────
Unit tests for app/services/recommendation_service.py — independent of the
database. generate_recommendations() only depends on the SHAPE of the
analytics dict that analytics_service.get_user_analytics() assembles, so
these tests build minimal dicts by hand rather than touching a real DB.

See tests/test_analytics_recommendations.py for integration tests exercising
this wired up end-to-end through GET /analytics/me.
"""
from datetime import datetime, timezone

from app.core.config import settings
from app.services import recommendation_service

NOW = datetime.now(timezone.utc)


def _topic(
    topic="Algebra", status="strong", accuracy=90.0, total_attempted=10,
    trend="stable", repeated_same_mistakes=0, behavior="fast_and_accurate",
    mastery_score=80.0, last_attempted_at=None,
) -> dict:
    return {
        "topic": topic,
        "status": status,
        "accuracy": accuracy,
        "total_attempted": total_attempted,
        "last_attempted_at": last_attempted_at or NOW,
        "answering_behavior": behavior,
        "performance_trend": {"trend": trend},
        "repeated_question_analytics": {"repeated_same_mistakes": repeated_same_mistakes},
        "mastery_score": mastery_score,
    }


def _subject(
    subject="Mathematics", accuracy=90.0, trend="stable", accuracy_change=0.0,
    repeated_same_mistakes=0, mastery_score=80.0, mastery_level="advanced",
    current_difficulty="medium", consecutive_strong_quizzes=0,
    quizzes_required_for_promotion=2, next_difficulty="hard", topics=None,
) -> dict:
    return {
        "subject": subject,
        "accuracy": accuracy,
        "performance_trend": {"trend": trend, "accuracy_change": accuracy_change},
        "repeated_question_analytics": {"repeated_same_mistakes": repeated_same_mistakes},
        "mastery_score": mastery_score,
        "mastery_level": mastery_level,
        "current_difficulty": current_difficulty,
        "consecutive_strong_quizzes": consecutive_strong_quizzes,
        "quizzes_required_for_promotion": quizzes_required_for_promotion,
        "next_difficulty": next_difficulty,
        "topics": topics or [],
    }


def _analytics(subjects=None, strong_subjects=None, total_sessions=10, abandoned_sessions=0) -> dict:
    return {
        "subjects": subjects or [],
        "strong_subjects": strong_subjects or [],
        "total_sessions": total_sessions,
        "abandoned_sessions": abandoned_sessions,
    }


def _generate(analytics: dict) -> list[dict]:
    return recommendation_service.generate_recommendations(analytics, settings_obj=settings, now=NOW)


# ─────────────────────────────────────────────────────────────────────────────
# One trigger per recommendation type
# ─────────────────────────────────────────────────────────────────────────────

def test_weak_topic_recommendation_matches_expected_shape():
    topic = _topic(
        topic="Writing Techniques", status="weak", accuracy=30.0, total_attempted=10,
        trend="declining", repeated_same_mistakes=4,
    )
    analytics = _analytics(subjects=[_subject(subject="English", topics=[topic])])

    recs = _generate(analytics)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["priority"] == 1
    assert rec["type"] == "weak_topic"
    assert rec["subject"] == "English"
    assert rec["topic"] == "Writing Techniques"
    assert rec["reason"] == "Accuracy is 30% across 10 attempts"
    assert rec["recommended_action"] == "Complete an easy practice quiz on Writing Techniques"
    assert rec["recommended_difficulty"] == "easy"
    assert rec["supporting_metrics"] == {
        "accuracy": 30.0, "attempts": 10, "trend": "declining", "repeated_mistakes": 4,
    }


def test_declining_subject_recommendation():
    subj = _subject(subject="Science", trend="declining", accuracy_change=-15.3, topics=[])
    recs = _generate(_analytics(subjects=[subj]))
    assert len(recs) == 1
    assert recs[0]["type"] == "declining_subject"
    assert recs[0]["subject"] == "Science"
    assert "15" in recs[0]["reason"]


def test_repeated_mistake_recommendation():
    assert settings.ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_THRESHOLD == 2
    topic = _topic(status="strong", repeated_same_mistakes=3)
    recs = _generate(_analytics(subjects=[_subject(topics=[topic])]))
    assert len(recs) == 1
    assert recs[0]["type"] == "repeated_mistake"


def test_careless_guessing_recommendation():
    topic = _topic(behavior="fast_but_inaccurate")
    recs = _generate(_analytics(subjects=[_subject(topics=[topic])]))
    assert len(recs) == 1
    assert recs[0]["type"] == "careless_guessing"
    assert recs[0]["recommended_difficulty"] is None


def test_slow_response_recommendation():
    for behavior in ("slow_and_accurate", "slow_and_inaccurate"):
        topic = _topic(behavior=behavior)
        recs = _generate(_analytics(subjects=[_subject(topics=[topic])]))
        assert len(recs) == 1
        assert recs[0]["type"] == "slow_response"


def test_incomplete_quiz_recommendation():
    assert settings.ANALYTICS_RECOMMENDATION_ABANDONMENT_THRESHOLD == 20.0
    analytics = _analytics(subjects=[], total_sessions=10, abandoned_sessions=3)  # 30%
    recs = _generate(analytics)
    assert len(recs) == 1
    assert recs[0]["type"] == "incomplete_quiz"
    assert recs[0]["subject"] is None
    assert recs[0]["topic"] is None
    assert recs[0]["reason"] == "3 of 10 quizzes were abandoned before completion"


def test_incomplete_quiz_not_triggered_below_threshold():
    analytics = _analytics(subjects=[], total_sessions=10, abandoned_sessions=1)  # 10% < 20%
    assert _generate(analytics) == []


def test_difficulty_ready_for_promotion():
    subj = _subject(
        current_difficulty="medium", consecutive_strong_quizzes=1,
        quizzes_required_for_promotion=2, next_difficulty="hard", topics=[],
    )
    recs = _generate(_analytics(subjects=[subj]))
    assert len(recs) == 1
    assert recs[0]["type"] == "difficulty_ready_for_promotion"
    assert recs[0]["recommended_difficulty"] == "medium"


def test_no_promotion_recommendation_when_already_at_max_difficulty():
    # Even with a qualifying streak, no promotion is possible from "hard".
    subj = _subject(
        current_difficulty="hard", consecutive_strong_quizzes=1,
        quizzes_required_for_promotion=2, next_difficulty="hard", topics=[],
        mastery_level="developing",  # avoid also matching maintain_strong_subject
    )
    assert _generate(_analytics(subjects=[subj])) == []


def test_maintain_strong_subject_recommendation():
    subj = _subject(subject="History", trend="stable", mastery_level="advanced", topics=[])
    recs = _generate(_analytics(subjects=[subj], strong_subjects=["History"]))
    assert len(recs) == 1
    assert recs[0]["type"] == "maintain_strong_subject"


def test_maintain_strong_subject_not_triggered_for_developing_mastery():
    subj = _subject(subject="History", trend="stable", mastery_level="developing", topics=[])
    assert _generate(_analytics(subjects=[subj], strong_subjects=["History"])) == []


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication, ordering, and the 5-item cap
# ─────────────────────────────────────────────────────────────────────────────

def test_dedup_keeps_only_highest_priority_type_per_topic():
    # This topic qualifies for BOTH weak_topic (priority 1) and
    # repeated_mistake (priority 3) — only one recommendation must survive,
    # and it must be the higher-priority (lower number) type.
    topic = _topic(status="weak", accuracy=25.0, repeated_same_mistakes=3)
    recs = _generate(_analytics(subjects=[_subject(topics=[topic])]))
    assert len(recs) == 1
    assert recs[0]["type"] == "weak_topic"


def test_weak_topic_outranks_maintain_strong_subject():
    weak_topic = _topic(topic="Fractions", status="weak", accuracy=20.0)
    weak_subject = _subject(subject="Mathematics", topics=[weak_topic])
    strong_subject = _subject(
        subject="Art", trend="stable", mastery_level="advanced", topics=[],
    )
    recs = _generate(_analytics(
        subjects=[weak_subject, strong_subject], strong_subjects=["Art"],
    ))
    types_in_order = [r["type"] for r in recs]
    assert types_in_order.index("weak_topic") < types_in_order.index("maintain_strong_subject")
    priorities = {r["type"]: r["priority"] for r in recs}
    assert priorities["weak_topic"] < priorities["maintain_strong_subject"]


def test_capped_at_five_recommendations():
    topics = [
        _topic(topic=f"Topic{i}", status="weak", accuracy=20.0 + i)
        for i in range(8)
    ]
    recs = _generate(_analytics(subjects=[_subject(topics=topics)]))
    assert len(recs) == settings.ANALYTICS_RECOMMENDATION_MAX_COUNT == 5
    assert [r["priority"] for r in recs] == [1, 2, 3, 4, 5]


def test_priorities_are_sequential_starting_at_one():
    topic1 = _topic(topic="A", status="weak", accuracy=20.0)
    topic2 = _topic(topic="B", status="weak", accuracy=25.0)
    recs = _generate(_analytics(subjects=[_subject(topics=[topic1, topic2])]))
    assert [r["priority"] for r in recs] == list(range(1, len(recs) + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Insufficient data -> empty list
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_list_when_nothing_qualifies():
    # An entirely unremarkable subject/topic: strong-but-not-in-strong_subjects,
    # not declining, no repeated mistakes, normal-paced answers, no promotion
    # streak, no abandonment.
    topic = _topic(status="strong", trend="stable", behavior="fast_and_accurate")
    subj = _subject(trend="stable", mastery_level="developing", topics=[topic])
    analytics = _analytics(subjects=[subj], strong_subjects=[], total_sessions=5, abandoned_sessions=0)
    assert _generate(analytics) == []


def test_empty_list_with_no_subjects_at_all():
    assert _generate(_analytics(subjects=[])) == []
