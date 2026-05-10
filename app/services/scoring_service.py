"""
services/scoring_service.py
────────────────────────────
Pure calculation functions for quiz scoring.

These are intentionally stateless pure functions — they take numbers and return
numbers. No DB access, no Groq calls. This makes them trivially unit-testable.

Called by quiz_service.py after grading individual attempts.
"""


def calculate_score(correct_count: int, total_questions: int) -> float:
    """
    Returns the raw score (number of correct answers).
    Kept as a dedicated function for clarity and future extensibility
    (e.g. weighted scoring per difficulty).
    """
    return float(correct_count)


def calculate_accuracy(correct_count: int, total_questions: int) -> float:
    """
    Returns accuracy as a percentage (0.0 – 100.0).

    Example: 7 correct out of 10 → 70.0
    """
    if total_questions == 0:
        return 0.0
    return round((correct_count / total_questions) * 100.0, 2)


def calculate_avg_response_time(timings: list[float]) -> float:
    """
    Returns the average response time in seconds.

    `timings` is a list of per-question response times sent by the frontend.
    """
    if not timings:
        return 0.0
    return round(sum(timings) / len(timings), 3)


def identify_weak_topic(
    question_topics: dict[int, str | None],
    wrong_question_ids: list[int],
) -> str | None:
    """
    Given a mapping of question_id → topic/lesson, and the list of incorrectly
    answered question IDs, find the topic with the most wrong answers.

    Returns the weak topic name, or None if there's insufficient data.

    Args:
        question_topics: {question_id: topic_string}
        wrong_question_ids: list of question IDs answered incorrectly

    Example:
        question_topics = {1: "Algebra", 2: "Algebra", 3: "Geometry"}
        wrong_question_ids = [1, 2]
        → returns "Algebra"
    """
    if not wrong_question_ids:
        return None

    topic_counts: dict[str, int] = {}
    for qid in wrong_question_ids:
        topic = question_topics.get(qid)
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    if not topic_counts:
        return None

    # Return the topic with the highest error count
    return max(topic_counts, key=lambda t: topic_counts[t])


def compute_performance_band(accuracy: float) -> str:
    """
    Maps an accuracy percentage to a human-readable performance band.
    Used for display purposes in the frontend.

    Returns: "Excellent" | "Good" | "Fair" | "Needs Improvement"
    """
    if accuracy >= 85.0:
        return "Excellent"
    elif accuracy >= 70.0:
        return "Good"
    elif accuracy >= 50.0:
        return "Fair"
    else:
        return "Needs Improvement"
