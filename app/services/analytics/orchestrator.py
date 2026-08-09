from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.analytics import queries
from app.services.analytics.growth_service import GrowthAnalyticsService
from app.services.analytics.mastery_service import MasteryScoreService
from app.services.analytics.recommendation_service import RecommendationService
from app.services.analytics.repeated_mistake_service import RepeatedMistakeAnalyticsService
from app.services.analytics.subject_service import SubjectAnalyticsService, build_difficulty_performance_by_subject
from app.services.analytics.summary_service import AnalyticsSummaryService
from app.services.analytics.timing import timed_phase
from app.services.analytics.topic_service import TopicAnalyticsService
from app.services.analytics.trend_service import TrendAnalyticsService
from app.services.analytics.types import ResponseTimeRow
from app.services.quiz_service import get_or_create_user


def _group_response_time_rows(
    rows: list[ResponseTimeRow],
) -> tuple[dict[str, list[ResponseTimeRow]], dict[tuple[str, str], list[ResponseTimeRow]]]:
    by_subject: dict[str, list[ResponseTimeRow]] = {}
    by_topic: dict[tuple[str, str], list[ResponseTimeRow]] = {}
    for row in rows:
        by_subject.setdefault(row.subject, []).append(row)
        by_topic.setdefault((row.subject, row.topic), []).append(row)
    return by_subject, by_topic


class AnalyticsOrchestrationService:
    # Builds the full GET /analytics/me response for one user: runs every
    # query in queries.py exactly once (18 bounded, user-scoped SQL
    # statements total — fixed no matter how many subjects/topics/sessions
    # the user has, since nothing here loops a query per subject or topic),
    # then runs the analytics services in dependency order over that data,
    # entirely in Python from here on.

    def __init__(self, db: AsyncSession, clerk_id: str):
        self._db = db
        self._clerk_id = clerk_id

    async def build(self) -> dict:
        db = self._db

        with timed_phase("resolve_user"):
            user = await get_or_create_user(db, self._clerk_id)

        with timed_phase("queries"):
            analytics_rows = await queries.fetch_analytics_rows(db, user.id)
            mastery_by_subject = await queries.fetch_subject_mastery_by_subject(db, user.id)
            session_stats = await queries.fetch_session_completion_stats(db, user.id)
            graded_totals = await queries.fetch_graded_totals(db, user.id)
            topic_rows = await queries.fetch_topic_rows(db, user.id)
            response_time_rows = await queries.fetch_response_time_rows(db, user.id)
            session_completed_at = await queries.fetch_session_completed_at(db, user.id)
            trend_attempt_rows = await queries.fetch_trend_attempt_rows(db, user.id)
            repeated_attempt_rows = await queries.fetch_repeated_attempt_rows(db, user.id)
            difficulty_attempt_rows = await queries.fetch_difficulty_attempt_rows(db, user.id)
            difficulty_session_rows = await queries.fetch_difficulty_session_rows(db, user.id)
            topic_difficulty_rows = await queries.fetch_topic_difficulty_rows(db, user.id)

            growth_window_start = datetime.now(timezone.utc) - timedelta(days=settings.ANALYTICS_GROWTH_WINDOW_DAYS)
            growth_session_rows = await queries.fetch_growth_session_rows(db, user.id, growth_window_start)
            growth_attempt_rows = await queries.fetch_growth_attempt_rows(db, user.id, growth_window_start)

        with timed_phase("topics"):
            topics_by_subject = TopicAnalyticsService(topic_rows).build()

        with timed_phase("trend"):
            trend_service = TrendAnalyticsService(trend_attempt_rows, session_completed_at)

        with timed_phase("repeated_mistakes"):
            repeated_service = RepeatedMistakeAnalyticsService(repeated_attempt_rows)

        with timed_phase("summary"):
            summary_service = AnalyticsSummaryService(session_stats, graded_totals, response_time_rows)
            overall_median_response_time = summary_service.overall_median_response_time
            _, response_time_by_topic = _group_response_time_rows(response_time_rows)
            TopicAnalyticsService.attach_response_time(
                topics_by_subject, response_time_by_topic, overall_median_response_time,
            )
            TopicAnalyticsService.attach_trend(topics_by_subject, trend_service)
            TopicAnalyticsService.attach_repeated_mistakes(topics_by_subject, repeated_service)

        with timed_phase("mastery"):
            mastery_service = MasteryScoreService(topic_difficulty_rows)
            TopicAnalyticsService.attach_mastery(topics_by_subject, mastery_service, trend_service, repeated_service)
            TopicAnalyticsService.sort_topics(topics_by_subject)

        with timed_phase("subjects"):
            response_time_by_subject, _ = _group_response_time_rows(response_time_rows)
            difficulty_performance_by_subject = build_difficulty_performance_by_subject(
                difficulty_attempt_rows, difficulty_session_rows,
            )
            subject_service = SubjectAnalyticsService(
                analytics_rows=analytics_rows,
                mastery_by_subject=mastery_by_subject,
                topics_by_subject=topics_by_subject,
                subject_response_time_rows=response_time_by_subject,
                overall_median_response_time=overall_median_response_time,
                trend_service=trend_service,
                repeated_mistake_service=repeated_service,
                mastery_service=mastery_service,
                difficulty_performance_by_subject=difficulty_performance_by_subject,
                max_topics=settings.ANALYTICS_MAX_TOPICS_PER_SUBJECT,
            )
            subjects_data, strong_subjects, weak_subjects = subject_service.build()

        overall_performance_trend = trend_service.trend_for(trend_service.overall_session_stats)
        overall_repeated_question_analytics = repeated_service.overall

        result = {
            **summary_service.build(),
            "subjects": subjects_data,
            "strong_subjects": strong_subjects,
            "weak_subjects": weak_subjects,
            "performance_trend": overall_performance_trend,
            "repeated_question_analytics": overall_repeated_question_analytics,
        }

        with timed_phase("growth"):
            growth_analytics_service = GrowthAnalyticsService(
                growth_session_rows=growth_session_rows,
                growth_attempt_rows=growth_attempt_rows,
                topics_by_subject=topics_by_subject,
                overall_performance_trend=overall_performance_trend,
                overall_repeated_question_analytics=overall_repeated_question_analytics,
                subjects_data=subjects_data,
            )
            result["growth"] = growth_analytics_service.build()

        with timed_phase("recommendations"):
            result["recommendations"] = RecommendationService(result).build()

        return result


async def get_user_analytics(db: AsyncSession, clerk_id: str) -> dict:
    return await AnalyticsOrchestrationService(db, clerk_id).build()
