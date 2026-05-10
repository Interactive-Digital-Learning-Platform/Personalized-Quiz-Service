"""
services/analytics_service.py
──────────────────────────────
Business logic for computing and updating user analytics.

Called after every quiz submission (`POST /quiz/submit`) to maintain the
`analytics` table as an up-to-date aggregate of the user's performance.

Also called by `GET /analytics/me` and `GET /analytics/feedback` to provide
the user's learning profile.
"""
import logging

from sqlalchemy import Integer, cast, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import Analytics
from app.models.quiz_session import QuizSession, QuestionAttempt
from app.models.question import Question
from app.models.user import User
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
            func.avg(QuestionAttempt.response_time).label("avg_time"),
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

    Returns:
        {
            overall_accuracy: float,
            overall_avg_response_time: float,
            total_sessions: int,
            subjects: list[{subject, accuracy, avg_response_time, weak_topic}],
            strong_subjects: list[str],
            weak_subjects: list[str],
        }
    """
    user = await get_or_create_user(db, clerk_id)

    # ── Fetch per-subject analytics rows ──────────────────────────────────────
    analytics_stmt = select(Analytics).where(Analytics.user_id == user.id)
    analytics_result = await db.execute(analytics_stmt)
    rows: list[Analytics] = list(analytics_result.scalars().all())

    # ── Count total quiz sessions ──────────────────────────────────────────────
    session_count_stmt = select(func.count(QuizSession.id)).where(
        QuizSession.user_id == user.id
    )
    total_sessions: int = (await db.execute(session_count_stmt)).scalar_one()

    # ── Compute overall averages ───────────────────────────────────────────────
    if rows:
        overall_accuracy = round(sum(r.accuracy for r in rows) / len(rows), 2)
        overall_avg_response_time = round(
            sum(r.avg_response_time for r in rows) / len(rows), 3
        )
    else:
        overall_accuracy = 0.0
        overall_avg_response_time = 0.0

    # ── Sort subjects by accuracy ──────────────────────────────────────────────
    sorted_subjects = sorted(rows, key=lambda r: r.accuracy, reverse=True)
    midpoint = len(sorted_subjects) // 2

    strong_subjects = [r.subject for r in sorted_subjects[:midpoint]] if sorted_subjects else []
    weak_subjects = [r.subject for r in sorted_subjects[midpoint:]] if sorted_subjects else []

    subjects_data = [
        {
            "subject": r.subject,
            "accuracy": r.accuracy,
            "avg_response_time": r.avg_response_time,
            "weak_topic": r.weak_topic,
        }
        for r in rows
    ]

    return {
        "overall_accuracy": overall_accuracy,
        "overall_avg_response_time": overall_avg_response_time,
        "total_sessions": total_sessions,
        "subjects": subjects_data,
        "strong_subjects": strong_subjects,
        "weak_subjects": weak_subjects,
    }
