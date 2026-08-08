"""
services/analytics/repeated_mistake_service.py
──────────────────────────────────────────────────
RepeatedMistakeAnalyticsService — groups graded attempts by
Question.question_fingerprint (the stable identifier for "the same or an
equivalent question" — see app/services/question_fingerprint.py) and
derives repeated_question_analytics for the overall/subject/topic scopes,
delegating the actual correction/repeat counting to scoring_service.

Also derives each topic's most-recently-attempted difficulty from the same
chronologically-ordered rows — used by MasteryScoreService, since topics
have no dedicated adaptive-difficulty table of their own to read from (see
that service's docstring for why LessonMastery isn't used instead).
"""
from app.services.analytics.types import RepeatedAttemptRow
from app.services.scoring_service import aggregate_repeated_question_stats, compute_repeated_question_group_stats


class RepeatedMistakeAnalyticsService:
    def __init__(self, repeated_attempt_rows: list[RepeatedAttemptRow]):
        fingerprint_groups: dict[str, list[RepeatedAttemptRow]] = {}
        # Rows arrive already in chronological order (see queries.
        # fetch_repeated_attempt_rows) — overwriting this dict while
        # iterating leaves each (subject, topic) key pointing at its MOST
        # RECENTLY attempted difficulty.
        self.most_recent_topic_difficulty: dict[tuple[str, str], str] = {}
        for row in repeated_attempt_rows:
            fingerprint_groups.setdefault(row.fingerprint, []).append(row)
            self.most_recent_topic_difficulty[(row.subject, row.topic)] = row.difficulty

        self._overall_group_stats: list[dict] = []
        self._subject_group_stats: dict[str, list[dict]] = {}
        self._topic_group_stats: dict[tuple[str, str], list[dict]] = {}

        for group_rows in fingerprint_groups.values():
            if len(group_rows) < 2:
                continue  # "Ignore questions attempted only once."
            stats = compute_repeated_question_group_stats([row.correct for row in group_rows])
            self._overall_group_stats.append(stats)
            # A fingerprint is derived from (subject, lesson, text), so
            # every row in this group shares the same subject/topic — safe
            # to read from just the first row.
            first = group_rows[0]
            self._subject_group_stats.setdefault(first.subject, []).append(stats)
            self._topic_group_stats.setdefault((first.subject, first.topic), []).append(stats)

    @property
    def overall(self) -> dict:
        return aggregate_repeated_question_stats(self._overall_group_stats)

    def for_subject(self, subject: str) -> dict:
        return aggregate_repeated_question_stats(self._subject_group_stats.get(subject, []))

    def for_topic(self, subject: str, topic: str) -> dict:
        return aggregate_repeated_question_stats(self._topic_group_stats.get((subject, topic), []))
