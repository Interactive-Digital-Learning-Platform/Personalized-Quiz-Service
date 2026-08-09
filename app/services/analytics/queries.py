from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, and_, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.analytics import Analytics
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion, QuizProgressSnapshot
from app.models.subject_mastery import SubjectMastery
from app.services.analytics.types import (
    DifficultyAttemptRow,
    DifficultySessionRow,
    GradedTotals,
    GrowthAttemptRow,
    GrowthSessionRow,
    RepeatedAttemptRow,
    ResponseTimeRow,
    SessionCompletionStats,
    TopicDifficultyRow,
    TopicRow,
    TrendAttemptRow,
)

# Every query below filters QuizSession.deleted_at IS NULL — this endpoint is
# a dashboard of the user's CURRENT sessions, so a soft-deleted one (see
# DELETE /quiz/sessions/{id}) must disappear from every number here, not just
# the sessions list. That's deliberately different from quiz_service.
# _get_recent_lessons() and the SubjectMastery/LessonMastery tables, which
# keep reading through deleted sessions since those drive quiz generation
# and must never lose history just because the user tidied up their list.

UNKNOWN_TOPIC = "Unknown Topic"


def as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _topic_expr():
    return func.coalesce(func.nullif(func.trim(Question.lesson), ""), UNKNOWN_TOPIC)


def valid_response_time_case():
    # NULLs out an invalid response_time so avg()/sum() over it skips those
    # rows without shrinking any other aggregate sharing the same WHERE
    # clause. Never touches the raw stored value.
    return case(
        (
            and_(
                QuestionAttempt.response_time.is_not(None),
                QuestionAttempt.response_time >= 0,
                QuestionAttempt.response_time <= settings.ANALYTICS_MAX_VALID_RESPONSE_TIME_SECONDS,
            ),
            QuestionAttempt.response_time,
        ),
        else_=None,
    )


async def fetch_analytics_rows(db: AsyncSession, user_id: int) -> list[Analytics]:
    stmt = select(Analytics).where(Analytics.user_id == user_id)
    return list((await db.execute(stmt)).scalars().all())


async def fetch_subject_mastery_by_subject(db: AsyncSession, user_id: int) -> dict[str, SubjectMastery]:
    stmt = select(SubjectMastery).where(SubjectMastery.user_id == user_id)
    rows = list((await db.execute(stmt)).scalars().all())
    return {m.subject: m for m in rows}


async def fetch_session_completion_stats(db: AsyncSession, user_id: int) -> SessionCompletionStats:
    total_sessions: int = int(
        (await db.execute(
            select(func.count(QuizSession.id)).where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None))
        )).scalar_one() or 0
    )

    completion_stmt = (
        select(
            func.count(QuizCompletion.id).label("completed_total"),
            func.sum(case((QuizCompletion.ended_by == "timeout", 1), else_=0)).label("timed_out_total"),
            func.avg(QuizCompletion.total_time).label("avg_duration"),
        )
        .join(QuizSession, QuizSession.id == QuizCompletion.session_id)
        .where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None))
    )
    completion_row = (await db.execute(completion_stmt)).one()
    # Postgres can return SUM()/CASE aggregates as Decimal rather than plain
    # int (SQLite doesn't do this) — explicit int() here rather than `or 0`
    # alone, which only masks the type when the result happens to be zero.
    completed_sessions = int(completion_row.completed_total or 0)
    timed_out_sessions = int(completion_row.timed_out_total or 0)
    average_session_duration_seconds = round(float(completion_row.avg_duration or 0.0), 2)

    average_questions_per_session = round(
        float((await db.execute(
            select(func.avg(QuizSession.question_count)).where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None))
        )).scalar_one() or 0.0),
        2,
    )

    # Abandoned = incomplete session inactive longer than the threshold.
    # Loads one row per incomplete session, then does the "older than N
    # hours" check in Python with explicit UTC normalization — SQLite drops
    # tzinfo on read, so comparing against a tz-aware cutoff is simpler done
    # here than as raw SQL across two dialects.
    last_snapshot_subq = (
        select(
            QuizProgressSnapshot.session_id.label("session_id"),
            func.max(QuizProgressSnapshot.saved_at).label("last_saved_at"),
        )
        .group_by(QuizProgressSnapshot.session_id)
        .subquery()
    )
    incomplete_activity_stmt = (
        select(
            QuizSession.created_at,
            last_snapshot_subq.c.last_saved_at,
        )
        .select_from(QuizSession)
        .outerjoin(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .outerjoin(last_snapshot_subq, last_snapshot_subq.c.session_id == QuizSession.id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuizCompletion.id.is_(None),
        )
    )
    incomplete_rows = (await db.execute(incomplete_activity_stmt)).all()

    incomplete_sessions = len(incomplete_rows)
    abandoned_cutoff = datetime.now(timezone.utc) - timedelta(
        hours=settings.ANALYTICS_ABANDONED_AFTER_HOURS
    )
    abandoned_sessions = sum(
        1
        for row in incomplete_rows
        if as_utc(row.last_saved_at or row.created_at) < abandoned_cutoff
    )

    completion_rate = (
        round(completed_sessions / total_sessions * 100.0, 2) if total_sessions > 0 else 0.0
    )
    timeout_rate = (
        round(timed_out_sessions / total_sessions * 100.0, 2) if total_sessions > 0 else 0.0
    )

    return SessionCompletionStats(
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        timed_out_sessions=timed_out_sessions,
        incomplete_sessions=incomplete_sessions,
        abandoned_sessions=abandoned_sessions,
        average_session_duration_seconds=average_session_duration_seconds,
        average_questions_per_session=average_questions_per_session,
        completion_rate=completion_rate,
        timeout_rate=timeout_rate,
    )


async def fetch_graded_totals(db: AsyncSession, user_id: int) -> GradedTotals:
    # Weighted, attempt-level totals — deliberately not an average of each
    # subject's own accuracy, which would let a 2-question subject count as
    # much as a 50-question one.
    graded_stmt = (
        select(
            func.count(QuestionAttempt.id).label("graded_total"),
            func.sum(cast(QuestionAttempt.correct, Integer)).label("correct_sum"),
        )
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
    )
    graded_row = (await db.execute(graded_stmt)).one()
    graded_total = int(graded_row.graded_total or 0)
    total_correct_answers = int(graded_row.correct_sum or 0)
    total_incorrect_answers = graded_total - total_correct_answers

    # Unanswered = session.question_count minus recorded attempts per
    # session, via subquery — still a single aggregate query.
    attempt_counts_subq = (
        select(
            QuestionAttempt.session_id.label("session_id"),
            func.count(QuestionAttempt.id).label("attempt_count"),
        )
        .group_by(QuestionAttempt.session_id)
        .subquery()
    )
    missing_expr = QuizSession.question_count - func.coalesce(attempt_counts_subq.c.attempt_count, 0)
    unanswered_stmt = (
        select(func.coalesce(func.sum(case((missing_expr > 0, missing_expr), else_=0)), 0))
        .select_from(QuizSession)
        .outerjoin(attempt_counts_subq, attempt_counts_subq.c.session_id == QuizSession.id)
        .where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None))
    )
    total_unanswered_questions = int((await db.execute(unanswered_stmt)).scalar_one() or 0)

    total_questions_attempted = (
        total_correct_answers + total_incorrect_answers + total_unanswered_questions
    )
    overall_accuracy = (
        round(total_correct_answers / total_questions_attempted * 100.0, 2)
        if total_questions_attempted > 0
        else 0.0
    )

    return GradedTotals(
        total_correct_answers=total_correct_answers,
        total_incorrect_answers=total_incorrect_answers,
        total_unanswered_questions=total_unanswered_questions,
        total_questions_attempted=total_questions_attempted,
        overall_accuracy=overall_accuracy,
    )


async def fetch_topic_rows(db: AsyncSession, user_id: int) -> list[TopicRow]:
    # Grouped by Question.subject/lesson, not QuizSession.lesson — one
    # session can span multiple lessons since each question gets its own
    # random lesson at generation time.
    topic_expr = _topic_expr()
    stmt = (
        select(
            Question.subject.label("subject"),
            topic_expr.label("topic"),
            func.count(QuestionAttempt.id).label("total_attempted"),
            func.sum(cast(QuestionAttempt.correct, Integer)).label("total_correct"),
            func.max(
                func.coalesce(QuizCompletion.completed_at, QuizSession.created_at)
            ).label("last_attempted_at"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .outerjoin(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
        .group_by(Question.subject, topic_expr)
    )
    rows = (await db.execute(stmt)).all()
    return [
        TopicRow(
            subject=r.subject,
            topic=r.topic,
            total_attempted=int(r.total_attempted or 0),
            total_correct=int(r.total_correct or 0),
            last_attempted_at=as_utc(r.last_attempted_at),
        )
        for r in rows
    ]


async def fetch_response_time_rows(db: AsyncSession, user_id: int) -> list[ResponseTimeRow]:
    # Every valid response time this user has, unaggregated — median/stddev
    # can't be recombined from grouped sub-aggregates, so one Python-side
    # pass over this list computes every scope (overall/subject/topic).
    topic_expr = _topic_expr()
    stmt = (
        select(
            QuizSession.subject.label("subject"),
            topic_expr.label("topic"),
            QuestionAttempt.correct.label("correct"),
            QuestionAttempt.response_time.label("response_time"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
            QuestionAttempt.response_time.is_not(None),
            QuestionAttempt.response_time >= 0,
            QuestionAttempt.response_time <= settings.ANALYTICS_MAX_VALID_RESPONSE_TIME_SECONDS,
        )
    )
    rows = (await db.execute(stmt)).all()
    return [
        ResponseTimeRow(subject=r.subject, topic=r.topic, correct=bool(r.correct), response_time=r.response_time)
        for r in rows
    ]


async def fetch_session_completed_at(db: AsyncSession, user_id: int) -> dict[int, datetime]:
    stmt = (
        select(
            QuizSession.id.label("session_id"),
            QuizCompletion.completed_at.label("completed_at"),
        )
        .select_from(QuizSession)
        .join(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None))
    )
    rows = (await db.execute(stmt)).all()
    return {row.session_id: as_utc(row.completed_at) for row in rows}


async def fetch_trend_attempt_rows(db: AsyncSession, user_id: int) -> list[TrendAttemptRow]:
    topic_expr = _topic_expr()
    stmt = (
        select(
            QuestionAttempt.session_id.label("session_id"),
            QuizSession.subject.label("subject"),
            topic_expr.label("topic"),
            QuestionAttempt.correct.label("correct"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .join(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
    )
    rows = (await db.execute(stmt)).all()
    return [
        TrendAttemptRow(session_id=r.session_id, subject=r.subject, topic=r.topic, correct=bool(r.correct))
        for r in rows
    ]


async def fetch_repeated_attempt_rows(db: AsyncSession, user_id: int) -> list[RepeatedAttemptRow]:
    # Ordered by (session.created_at, attempt.id) since QuestionAttempt has
    # no timestamp of its own — session creation time plus insertion-order
    # id is the best available chronological proxy.
    topic_expr = _topic_expr()
    stmt = (
        select(
            QuestionAttempt.id.label("attempt_id"),
            QuestionAttempt.correct.label("correct"),
            Question.question_fingerprint.label("fingerprint"),
            Question.subject.label("subject"),
            topic_expr.label("topic"),
            Question.difficulty.label("difficulty"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
        .order_by(QuizSession.created_at.asc(), QuestionAttempt.id.asc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        RepeatedAttemptRow(
            attempt_id=r.attempt_id, correct=bool(r.correct), fingerprint=r.fingerprint,
            subject=r.subject, topic=r.topic, difficulty=r.difficulty,
        )
        for r in rows
    ]


async def fetch_difficulty_attempt_rows(db: AsyncSession, user_id: int) -> list[DifficultyAttemptRow]:
    # Grouped by each question's own difficulty (not the session's), so a
    # session mixing difficulty levels is still split correctly.
    stmt = (
        select(
            QuizSession.subject.label("subject"),
            Question.difficulty.label("difficulty"),
            func.count(QuestionAttempt.id).label("total_attempted"),
            func.sum(cast(QuestionAttempt.correct, Integer)).label("total_correct"),
            func.avg(valid_response_time_case()).label("avg_response_time"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
        .group_by(QuizSession.subject, Question.difficulty)
    )
    rows = (await db.execute(stmt)).all()
    return [
        DifficultyAttemptRow(
            subject=r.subject, difficulty=r.difficulty,
            total_attempted=int(r.total_attempted or 0), total_correct=int(r.total_correct or 0),
            avg_response_time=round(float(r.avg_response_time or 0.0), 3),
        )
        for r in rows
    ]


async def fetch_difficulty_session_rows(db: AsyncSession, user_id: int) -> list[DifficultySessionRow]:
    stmt = (
        select(
            QuizSession.subject.label("subject"),
            Question.difficulty.label("difficulty"),
            func.count(func.distinct(QuizSession.id)).label("completed_sessions"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .join(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
        .group_by(QuizSession.subject, Question.difficulty)
    )
    rows = (await db.execute(stmt)).all()
    return [
        DifficultySessionRow(
            subject=r.subject, difficulty=r.difficulty, completed_sessions=int(r.completed_sessions or 0),
        )
        for r in rows
    ]


async def fetch_topic_difficulty_rows(db: AsyncSession, user_id: int) -> list[TopicDifficultyRow]:
    # Not sourced from LessonMastery — its `lesson` column is the raw string,
    # which isn't guaranteed to match the normalized topic label used here.
    topic_expr = _topic_expr()
    stmt = (
        select(
            QuizSession.subject.label("subject"),
            topic_expr.label("topic"),
            Question.difficulty.label("difficulty"),
            func.count(QuestionAttempt.id).label("total_attempted"),
            func.sum(cast(QuestionAttempt.correct, Integer)).label("total_correct"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
        )
        .group_by(QuizSession.subject, topic_expr, Question.difficulty)
    )
    rows = (await db.execute(stmt)).all()
    return [
        TopicDifficultyRow(
            subject=r.subject, topic=r.topic, difficulty=r.difficulty,
            total_attempted=int(r.total_attempted or 0), total_correct=int(r.total_correct or 0),
        )
        for r in rows
    ]


async def fetch_growth_session_rows(
    db: AsyncSession, user_id: int, window_start: datetime,
) -> list[GrowthSessionRow]:
    last_snapshot_subq = (
        select(
            QuizProgressSnapshot.session_id.label("session_id"),
            func.max(QuizProgressSnapshot.saved_at).label("last_saved_at"),
        )
        .group_by(QuizProgressSnapshot.session_id)
        .subquery()
    )
    stmt = (
        select(
            QuizSession.id.label("session_id"),
            QuizSession.created_at.label("created_at"),
            QuizCompletion.id.label("completion_id"),
            last_snapshot_subq.c.last_saved_at.label("last_saved_at"),
        )
        .select_from(QuizSession)
        .outerjoin(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .outerjoin(last_snapshot_subq, last_snapshot_subq.c.session_id == QuizSession.id)
        .where(QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None), QuizSession.created_at >= window_start)
    )
    rows = (await db.execute(stmt)).all()
    return [
        GrowthSessionRow(
            session_id=r.session_id, created_at=r.created_at,
            completion_id=r.completion_id, last_saved_at=r.last_saved_at,
        )
        for r in rows
    ]


async def fetch_growth_attempt_rows(
    db: AsyncSession, user_id: int, window_start: datetime,
) -> list[GrowthAttemptRow]:
    # The only source for effort's question-count/active-days/spacing
    # components, so a rapidly-created unanswered session contributes to
    # none of them — it can only hurt completion_rate/abandonment instead.
    topic_expr = _topic_expr()
    stmt = (
        select(
            QuestionAttempt.session_id.label("session_id"),
            QuestionAttempt.correct.label("correct"),
            QuizSession.subject.label("subject"),
            topic_expr.label("topic"),
            QuizSession.created_at.label("session_created_at"),
        )
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id, QuizSession.deleted_at.is_(None),
            QuestionAttempt.correct.is_not(None),
            QuizSession.created_at >= window_start,
        )
    )
    rows = (await db.execute(stmt)).all()
    return [
        GrowthAttemptRow(
            session_id=r.session_id, correct=bool(r.correct), subject=r.subject, topic=r.topic,
            session_created_at=r.session_created_at,
        )
        for r in rows
    ]
