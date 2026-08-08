"""
services/analytics_service.py
──────────────────────────────
Business logic for computing and updating user analytics.

`update_analytics_after_submission()` is called after every quiz submission
(`POST /quiz/submit`) to maintain the `analytics` table as an up-to-date
aggregate of the user's performance.

`get_user_analytics()` — called by `GET /analytics/me` and
`GET /analytics/feedback` — is now a thin delegator to
AnalyticsOrchestrationService (see app/services/analytics/orchestrator.py).
The full ~1200-line implementation that used to live in this function was
split into a dedicated `app/services/analytics/` package (typed query
layer + 8 single-responsibility calculation services + one orchestrator)
so this endpoint's logic stays maintainable as more analytics features are
added — see that package's __init__.py for the full architecture and
orchestrator.py for the exact query count.
"""
import logging

from sqlalchemy import Integer, cast, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import Analytics
from app.models.quiz_session import QuizSession, QuestionAttempt
from app.models.question import Question
from app.services.analytics.orchestrator import get_user_analytics as _get_user_analytics
from app.services.analytics.queries import valid_response_time_case
from app.services.quiz_service import get_or_create_user
from app.services.scoring_service import identify_weak_topic

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UPSERT ANALYTICS AFTER SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

async def update_analytics_after_submission(
    db: AsyncSession,
    clerk_id: str,
    session_id: int,
) -> None:
    """
    Recalculate and upsert the Analytics row for (user, subject) after a quiz.

    This implements a simple rolling average:
    - We aggregate ALL QuestionAttempt rows for the user in this subject,
      not just the current session, so the analytics reflect lifetime performance.

    Called by the quiz submission route — fire and forget from the caller's
    perspective (errors are logged but don't break the response).
    """
    user = await get_or_create_user(db, clerk_id)

    # Find the session to get the subject
    session_result = await db.execute(
        select(QuizSession).where(QuizSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()
    if session is None:
        logger.warning("update_analytics: session %d not found", session_id)
        return

    subject = session.subject

    # ── Aggregate attempts for this user × subject ─────────────────────────────
    # Join QuestionAttempt → QuizSession to filter by user and subject
    agg_stmt = (
        select(
            func.count(QuestionAttempt.id).label("total"),
            func.sum(
                cast(QuestionAttempt.correct, Integer)
            ).label("correct_sum"),
            # CASE-wrapped so an invalid response time is excluded from this
            # average without shrinking `total`/`correct_sum` above — those
            # are accuracy counts and must stay response-time-agnostic.
            func.avg(valid_response_time_case()).label("avg_time"),
        )
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(
            QuizSession.user_id == user.id,
            QuizSession.subject == subject,
            QuestionAttempt.correct.is_not(None),
        )
    )
    agg_result = await db.execute(agg_stmt)
    row = agg_result.one()

    total: int = row.total or 0
    correct_sum: int = row.correct_sum or 0
    avg_time: float = float(row.avg_time or 0.0)
    accuracy = (correct_sum / total * 100.0) if total > 0 else 0.0

    # ── Identify weak topic ────────────────────────────────────────────────────
    # Find wrong attempts in this session → map to question lessons
    wrong_stmt = (
        select(QuestionAttempt.question_id)
        .where(
            QuestionAttempt.session_id == session_id,
            QuestionAttempt.correct == False,  # noqa: E712
        )
    )
    wrong_result = await db.execute(wrong_stmt)
    wrong_ids = [r[0] for r in wrong_result.all()]

    # Build question_id → lesson mapping for wrong questions
    question_topics: dict[int, str | None] = {}
    if wrong_ids:
        q_stmt = select(Question.id, Question.lesson).where(Question.id.in_(wrong_ids))
        q_result = await db.execute(q_stmt)
        question_topics = {r[0]: r[1] for r in q_result.all()}

    weak_topic = identify_weak_topic(question_topics, wrong_ids)

    # ── Upsert Analytics row ───────────────────────────────────────────────────
    existing_stmt = select(Analytics).where(
        Analytics.user_id == user.id,
        Analytics.subject == subject,
    )
    existing_result = await db.execute(existing_stmt)
    analytics_row: Analytics | None = existing_result.scalar_one_or_none()

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


# ─────────────────────────────────────────────────────────────────────────────
# FETCH USER ANALYTICS PROFILE
# ─────────────────────────────────────────────────────────────────────────────

async def get_user_analytics(db: AsyncSession, clerk_id: str) -> dict:
    """
    Build the full analytics profile for `GET /analytics/me`.

    Delegates entirely to AnalyticsOrchestrationService — see
    app/services/analytics/orchestrator.py for the query plan and
    app/services/analytics/__init__.py for the full architecture. Kept as a
    module-level function here (rather than requiring every caller to
    import the orchestrator directly) so existing callers — the
    GET /analytics/me and GET /analytics/feedback routes — don't need to
    change.
    """
    return await _get_user_analytics(db, clerk_id)
