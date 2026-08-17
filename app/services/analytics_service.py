import logging

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import Analytics
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.services.analytics.orchestrator import (
    get_user_analytics as _get_user_analytics,
)
from app.services.analytics.queries import valid_response_time_case
from app.services.quiz_service import get_or_create_user
from app.services.scoring_service import identify_weak_topic

logger = logging.getLogger(__name__)


async def update_analytics_after_submission(
    db: AsyncSession,
    clerk_id: str,
    session_id: int,
) -> None:
    # Recomputes the rolling (user, subject) Analytics row from ALL of that
    # user's QuestionAttempt rows in this subject (not just this session), so
    # it reflects lifetime performance. Called after every quiz submission —
    # and also after a session delete, to keep this row from going stale
    # (see the DELETE /quiz/sessions/{id} route).
    user = await get_or_create_user(db, clerk_id)

    session_result = await db.execute(
        select(QuizSession).where(QuizSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()
    if session is None:
        logger.warning("update_analytics: session %d not found", session_id)
        return

    subject = session.subject

    agg_stmt = (
        select(
            func.count(QuestionAttempt.id).label("total"),
            func.sum(
                cast(QuestionAttempt.correct, Integer)
            ).label("correct_sum"),
            func.avg(valid_response_time_case()).label("avg_time"),
        )
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(
            QuizSession.user_id == user.id,
            QuizSession.subject == subject,
            QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
    )
    agg_result = await db.execute(agg_stmt)
    row = agg_result.one()

    total: int = row.total or 0
    correct_sum: int = row.correct_sum or 0
    avg_time: float = float(row.avg_time or 0.0)
    accuracy = (correct_sum / total * 100.0) if total > 0 else 0.0

    wrong_stmt = (
        select(QuestionAttempt.question_id)
        .where(
            QuestionAttempt.session_id == session_id,
            QuestionAttempt.correct == False, 
        )
    )
    wrong_result = await db.execute(wrong_stmt)
    wrong_ids = [r[0] for r in wrong_result.all()]

    question_topics: dict[int, str | None] = {}
    if wrong_ids:
        q_stmt = select(Question.id, Question.lesson).where(Question.id.in_(wrong_ids))
        q_result = await db.execute(q_stmt)
        question_topics = {r[0]: r[1] for r in q_result.all()}

    weak_topic = identify_weak_topic(question_topics, wrong_ids)

    existing_stmt = select(Analytics).where(
        Analytics.user_id == user.id,
        Analytics.subject == subject,
    )
    existing_result = await db.execute(existing_stmt)
    analytics_row: Analytics | None = existing_result.scalar_one_or_none()

    if total == 0:
        # Every session that ever fed this row has since been deleted (this
        # only happens via the delete-triggered recompute — a submission
        # always adds at least one attempt). Drop the row instead of leaving
        # a stale "0% accuracy" card for a subject with no sessions left.
        if analytics_row:
            await db.delete(analytics_row)
            logger.info("Removed analytics for user=%d, subject=%s: no sessions remain", user.id, subject)
        await db.commit()
        return

    if analytics_row:
        analytics_row.accuracy = round(accuracy, 2)
        analytics_row.avg_response_time = round(avg_time, 3)
        analytics_row.weak_topic = weak_topic
        logger.info(
            "Updated analytics for user=%d, subject=%s: accuracy=%.1f%%",
            user.id, subject, accuracy,
        )
    else:
        analytics_row = Analytics(
            user_id=user.id,
            subject=subject,
            accuracy=round(accuracy, 2),
            avg_response_time=round(avg_time, 3),
            weak_topic=weak_topic,
        )
        db.add(analytics_row)
        logger.info(
            "Created analytics for user=%d, subject=%s: accuracy=%.1f%%",
            user.id, subject, accuracy,
        )

    await db.commit()


async def get_user_analytics(db: AsyncSession, clerk_id: str) -> dict:
    # Thin delegator to AnalyticsOrchestrationService (app/services/analytics/) —
    # kept here so the routes don't need to import the orchestrator directly.
    return await _get_user_analytics(db, clerk_id)
