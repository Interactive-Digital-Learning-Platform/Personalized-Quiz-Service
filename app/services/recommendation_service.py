from datetime import datetime, timezone

# Lower = more urgent, and this is the PRIMARY sort key — severity (below)
# only breaks ties within the same type, so a "weak_topic" always outranks a
# "maintain_strong_subject" regardless of either one's severity score.
TYPE_BASE_PRIORITY = {
    "weak_topic": 1,
    "declining_subject": 2,
    "repeated_mistake": 3,
    "careless_guessing": 4,
    "slow_response": 5,
    "incomplete_quiz": 6,
    "difficulty_ready_for_promotion": 7,
    "maintain_strong_subject": 8,
}

_SLOW_BEHAVIORS = {"slow_and_accurate", "slow_and_inaccurate"}
_STRONG_MASTERY_LEVELS = {"proficient", "advanced"}


def _severity_score(
    *,
    accuracy: float | None,
    attempts: int,
    repeated_mistakes: int,
    mastery_score: float | None,
    last_attempted_at: datetime | None,
    now: datetime,
    accuracy_weight: float,
    attempts_weight: float,
    attempts_cap: int,
    repeated_mistakes_weight: float,
    mastery_weight: float,
    recency_weight: float,
    recency_half_life_days: float,
) -> float:
    # Higher = more urgent — only used to order candidates that already
    # share the same type (TYPE_BASE_PRIORITY decides which types win).
    score = 0.0
    if accuracy is not None:
        score += (100.0 - accuracy) * accuracy_weight
    score += min(attempts, attempts_cap) * attempts_weight
    score += repeated_mistakes * repeated_mistakes_weight
    if mastery_score is not None:
        score += (100.0 - mastery_score) * mastery_weight
    if last_attempted_at is not None:
        days_ago = max((now - last_attempted_at).total_seconds() / 86400.0, 0.0)
        # Exponential decay — something struggled with yesterday is more
        # actionable today than something struggled with two months ago.
        recency_factor = 0.5 ** (days_ago / recency_half_life_days)
        score += recency_factor * 100.0 * recency_weight
    return score


def _subject_topic_totals(subject_data: dict) -> tuple[int, datetime | None]:
    # Only used as tie-breaker inputs for subject-level candidates. Since
    # subjects[].topics is capped for display, a subject with more topics
    # than that cap gets slightly undercounted here — fine, since this only
    # affects ordering among same-type candidates, never whether one triggers.
    topics = subject_data["topics"]
    if not topics:
        return 0, None
    total_attempted = sum(t["total_attempted"] for t in topics)
    last_attempted_at = max(t["last_attempted_at"] for t in topics)
    return total_attempted, last_attempted_at


def _build_candidates(analytics: dict, *, settings_obj) -> list[dict]:
    candidates: list[dict] = []

    for subject_data in analytics["subjects"]:
        subject = subject_data["subject"]

        for topic_data in subject_data["topics"]:
            topic = topic_data["topic"]
            trend = topic_data["performance_trend"]["trend"]
            repeated = topic_data["repeated_question_analytics"]
            repeated_mistakes = repeated["repeated_same_mistakes"]
            accuracy = topic_data["accuracy"]
            attempts = topic_data["total_attempted"]
            mastery_score = topic_data["mastery_score"]
            last_attempted_at = topic_data["last_attempted_at"]
            behavior = topic_data["answering_behavior"]

            common = {
                "subject": subject, "topic": topic,
                "_accuracy": accuracy, "_attempts": attempts,
                "_repeated_mistakes": repeated_mistakes, "_mastery_score": mastery_score,
                "_last_attempted_at": last_attempted_at,
            }

            if topic_data["status"] == "weak":
                candidates.append({
                    **common,
                    "type": "weak_topic",
                    "reason": f"Accuracy is {accuracy:.0f}% across {attempts} attempts",
                    "recommended_action": f"Complete an easy practice quiz on {topic}",
                    "recommended_difficulty": "easy",
                    "supporting_metrics": {
                        "accuracy": accuracy, "attempts": attempts,
                        "trend": trend, "repeated_mistakes": repeated_mistakes,
                    },
                })

            if repeated_mistakes >= settings_obj.ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_THRESHOLD:
                candidates.append({
                    **common,
                    "type": "repeated_mistake",
                    "reason": (
                        f"The same mistake was repeated {repeated_mistakes} times "
                        f"without correction in {topic}"
                    ),
                    "recommended_action": f"Review and correct past mistakes in {topic}",
                    "recommended_difficulty": "easy",
                    "supporting_metrics": {
                        "accuracy": accuracy, "attempts": attempts,
                        "trend": trend, "repeated_mistakes": repeated_mistakes,
                    },
                })

            if behavior == "fast_but_inaccurate":
                candidates.append({
                    **common,
                    "type": "careless_guessing",
                    "reason": f"Answers in {topic} are fast but often incorrect",
                    "recommended_action": f"Slow down and re-read each question carefully in {topic}",
                    "recommended_difficulty": None,
                    "supporting_metrics": {
                        "accuracy": accuracy, "attempts": attempts,
                        "trend": trend, "repeated_mistakes": repeated_mistakes,
                    },
                })

            if behavior in _SLOW_BEHAVIORS:
                candidates.append({
                    **common,
                    "type": "slow_response",
                    "reason": f"Response times in {topic} are slower than the student's usual pace",
                    "recommended_action": f"Practice {topic} questions under light time pressure",
                    "recommended_difficulty": None,
                    "supporting_metrics": {
                        "accuracy": accuracy, "attempts": attempts,
                        "trend": trend, "repeated_mistakes": repeated_mistakes,
                    },
                })

        subject_trend = subject_data["performance_trend"]
        subject_repeated = subject_data["repeated_question_analytics"]
        subject_attempts, subject_last_attempted_at = _subject_topic_totals(subject_data)
        subject_common = {
            "subject": subject, "topic": None,
            "_accuracy": subject_data["accuracy"], "_attempts": subject_attempts,
            "_repeated_mistakes": subject_repeated["repeated_same_mistakes"],
            "_mastery_score": subject_data["mastery_score"],
            "_last_attempted_at": subject_last_attempted_at,
        }

        if subject_trend["trend"] == "declining":
            candidates.append({
                **subject_common,
                "type": "declining_subject",
                "reason": (
                    f"Accuracy in {subject} dropped {abs(subject_trend['accuracy_change']):.0f} "
                    "points recently"
                ),
                "recommended_action": f"Review recent mistakes in {subject} and retake a quiz",
                "recommended_difficulty": subject_data["current_difficulty"],
                "supporting_metrics": {
                    "accuracy": subject_data["accuracy"], "attempts": subject_attempts,
                    "trend": "declining", "repeated_mistakes": subject_repeated["repeated_same_mistakes"],
                },
            })

        # Only fires when the existing SubjectMastery streak already says
        # the student is one strong quiz from promotion — never invents its
        # own readiness criteria separate from the real adaptive-difficulty engine.
        if (
            subject_data["consecutive_strong_quizzes"] >= 1
            and subject_data["consecutive_strong_quizzes"] == subject_data["quizzes_required_for_promotion"] - 1
            and subject_data["next_difficulty"] != subject_data["current_difficulty"]
        ):
            candidates.append({
                **subject_common,
                "type": "difficulty_ready_for_promotion",
                "reason": (
                    f"One more strong quiz in {subject} will advance the student to "
                    f"{subject_data['next_difficulty']} difficulty"
                ),
                "recommended_action": f"Take one more {subject_data['current_difficulty']} quiz in {subject}",
                "recommended_difficulty": subject_data["current_difficulty"],
                "supporting_metrics": {
                    "accuracy": subject_data["accuracy"], "attempts": subject_attempts,
                    "trend": subject_trend["trend"],
                    "repeated_mistakes": subject_repeated["repeated_same_mistakes"],
                },
            })

        if (
            subject in analytics["strong_subjects"]
            and subject_trend["trend"] != "declining"
            and subject_data["mastery_level"] in _STRONG_MASTERY_LEVELS
        ):
            candidates.append({
                **subject_common,
                "type": "maintain_strong_subject",
                "reason": f"Consistently strong performance in {subject}",
                "recommended_action": f"Keep up periodic review in {subject} to maintain mastery",
                "recommended_difficulty": subject_data["current_difficulty"],
                "supporting_metrics": {
                    "accuracy": subject_data["accuracy"], "attempts": subject_attempts,
                    "trend": subject_trend["trend"],
                    "repeated_mistakes": subject_repeated["repeated_same_mistakes"],
                },
            })

    total_sessions = analytics["total_sessions"]
    abandoned_sessions = analytics["abandoned_sessions"]
    if total_sessions > 0:
        abandonment_rate = abandoned_sessions / total_sessions * 100.0
        if abandonment_rate >= settings_obj.ANALYTICS_RECOMMENDATION_ABANDONMENT_THRESHOLD:
            candidates.append({
                "subject": None, "topic": None,
                "_accuracy": None, "_attempts": total_sessions,
                "_repeated_mistakes": 0, "_mastery_score": None, "_last_attempted_at": None,
                "type": "incomplete_quiz",
                "reason": f"{abandoned_sessions} of {total_sessions} quizzes were abandoned before completion",
                "recommended_action": "Finish quizzes once started instead of leaving them incomplete",
                "recommended_difficulty": None,
                "supporting_metrics": {
                    "accuracy": None, "attempts": total_sessions,
                    "trend": None, "repeated_mistakes": 0,
                },
            })

    return candidates


def generate_recommendations(
    analytics: dict,
    *,
    settings_obj=None,
    now: datetime | None = None,
) -> list[dict]:
    # Builds the final, capped, deduplicated, priority-ordered list from an
    # already-assembled analytics dict — no DB access, no Groq, fully
    # deterministic. Returns [] when nothing triggers, which naturally
    # happens whenever there's too little data (every trigger already
    # requires its own minimum, e.g. "weak" status needs enough attempts).
    if settings_obj is None:
        from app.core.config import settings as settings_obj
    if now is None:
        now = datetime.now(timezone.utc)

    candidates = _build_candidates(analytics, settings_obj=settings_obj)
    if not candidates:
        return []

    for candidate in candidates:
        severity = _severity_score(
            accuracy=candidate["_accuracy"],
            attempts=candidate["_attempts"],
            repeated_mistakes=candidate["_repeated_mistakes"],
            mastery_score=candidate["_mastery_score"],
            last_attempted_at=candidate["_last_attempted_at"],
            now=now,
            accuracy_weight=settings_obj.ANALYTICS_RECOMMENDATION_ACCURACY_WEIGHT,
            attempts_weight=settings_obj.ANALYTICS_RECOMMENDATION_ATTEMPTS_WEIGHT,
            attempts_cap=settings_obj.ANALYTICS_RECOMMENDATION_ATTEMPTS_CAP,
            repeated_mistakes_weight=settings_obj.ANALYTICS_RECOMMENDATION_REPEATED_MISTAKES_WEIGHT,
            mastery_weight=settings_obj.ANALYTICS_RECOMMENDATION_MASTERY_WEIGHT,
            recency_weight=settings_obj.ANALYTICS_RECOMMENDATION_RECENCY_WEIGHT,
            recency_half_life_days=settings_obj.ANALYTICS_RECOMMENDATION_RECENCY_HALF_LIFE_DAYS,
        )
        candidate["_sort_key"] = (TYPE_BASE_PRIORITY[candidate["type"]], -severity)

    # Keep only the single best candidate per (subject, topic) — no
    # duplicate recommendations for the same subject/topic pair.
    best_by_key: dict[tuple[str | None, str | None], dict] = {}
    for candidate in candidates:
        key = (candidate["subject"], candidate["topic"])
        existing = best_by_key.get(key)
        if existing is None or candidate["_sort_key"] < existing["_sort_key"]:
            best_by_key[key] = candidate

    ranked = sorted(best_by_key.values(), key=lambda c: c["_sort_key"])
    ranked = ranked[: settings_obj.ANALYTICS_RECOMMENDATION_MAX_COUNT]

    return [
        {
            "priority": i,
            "type": candidate["type"],
            "subject": candidate["subject"],
            "topic": candidate["topic"],
            "reason": candidate["reason"],
            "recommended_action": candidate["recommended_action"],
            "recommended_difficulty": candidate["recommended_difficulty"],
            "supporting_metrics": candidate["supporting_metrics"],
        }
        for i, candidate in enumerate(ranked, start=1)
    ]
