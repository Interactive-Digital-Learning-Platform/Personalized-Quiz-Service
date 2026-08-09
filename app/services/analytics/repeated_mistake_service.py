from app.services.analytics.types import RepeatedAttemptRow
from app.services.scoring_service import aggregate_repeated_question_stats, compute_repeated_question_group_stats


class RepeatedMistakeAnalyticsService:
    # Groups graded attempts by Question.question_fingerprint ("the same or
    # an equivalent question") and derives repeated_question_analytics for
    # each scope. Also tracks each topic's most-recently-attempted
    # difficulty from the same rows, for MasteryScoreService — topics have
    # no dedicated adaptive-difficulty table of their own.

    def __init__(self, repeated_attempt_rows: list[RepeatedAttemptRow]):
        fingerprint_groups: dict[str, list[RepeatedAttemptRow]] = {}
        # Rows arrive in chronological order already, so overwriting this
        # dict while iterating naturally leaves each key pointing at its
        # most recently attempted difficulty.
        self.most_recent_topic_difficulty: dict[tuple[str, str], str] = {}
        for row in repeated_attempt_rows:
            fingerprint_groups.setdefault(row.fingerprint, []).append(row)
            self.most_recent_topic_difficulty[(row.subject, row.topic)] = row.difficulty

        self._overall_group_stats: list[dict] = []
        self._subject_group_stats: dict[str, list[dict]] = {}
        self._topic_group_stats: dict[tuple[str, str], list[dict]] = {}

        for group_rows in fingerprint_groups.values():
            if len(group_rows) < 2:
                continue  # a question attempted only once has nothing to "repeat"
            stats = compute_repeated_question_group_stats([row.correct for row in group_rows])
            self._overall_group_stats.append(stats)
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
