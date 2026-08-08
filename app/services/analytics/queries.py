"""
services/analytics/queries.py
────────────────────────────────
ALL database access for GET /analytics/me, in one place.

Every function here is a single, bounded, aggregate SQL statement scoped to
one user via `user_id` — never a loop issuing one query per subject/topic/
session (see orchestrator.py's docstring for the full query count and why
it stays fixed no matter how much data the user has). Nothing in this
module performs business-rule calculations (rates, thresholds, formulas) —
it only shapes raw query results into the typed dataclasses in types.py;
the 8 orchestration services derive everything else from those.

This is a line-for-line move of queries that already existed in
app/services/analytics_service.py before this refactor — no query was
added, removed, or changed.
"""
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

# Normalized label used whenever a question's stored `lesson` is missing/blank.
UNKNOWN_TOPIC = "Unknown Topic"


def as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC (SQLite drops tzinfo on read; every
    timestamp this app writes — server_default=func.now() — is UTC anyway)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _topic_expr():
    """Question.lesson, normalized: blank/whitespace-only collapses to
    UNKNOWN_TOPIC. Shared by every query below that groups by topic, so a
    topic label is computed identically everywhere it appears."""
    return func.coalesce(func.nullif(func.trim(Question.lesson), ""), UNKNOWN_TOPIC)


def valid_response_time_case():
    """
    SQL CASE expression yielding QuestionAttempt.response_time when it's
    valid (non-negative, not absurdly large) and NULL otherwise — so
    func.avg()/func.sum() over it silently ignore invalid rows without
    shrinking a query's other aggregates that share the same WHERE clause.
    Raw QuestionAttempt rows are never modified — this only affects what a
    given aggregate function reads from at calculation time.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 1-2. Analytics + SubjectMastery rows (per subject)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_analytics_rows(db: AsyncSession, user_id: int) -> list[Analytics]:
    stmt = select(Analytics).where(Analytics.user_id == user_id)
    return list((await db.execute(stmt)).scalars().all())


async def fetch_subject_mastery_by_subject(db: AsyncSession, user_id: int) -> dict[str, SubjectMastery]:
    """Read-only — GET /analytics/me never writes to SubjectMastery; only
    difficulty_service.update_subject_mastery_after_submission() does."""
    stmt = select(SubjectMastery).where(SubjectMastery.user_id == user_id)
    rows = list((await db.execute(stmt)).scalars().all())
    return {m.subject: m for m in rows}


# ─────────────────────────────────────────────────────────────────────────────
# 3-6. Session completion stats (total / completed / timed-out / avg duration /
#      avg questions per session / incomplete+abandoned)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_session_completion_stats(db: AsyncSession, user_id: int) -> SessionCompletionStats:
    """
    Combines what used to be 4 separate query round-trips in the
    pre-refactor function (total session count, completion/timeout/duration
    aggregate, average questions per session, incomplete-session activity)
    — still exactly 4 queries, just gathered behind one typed return value.

    See AnalyticsSummaryService for the abandonment/completion-rate
    classification rules this feeds into.
    """
    total_sessions: int = int(
        (await db.execute(
            select(func.count(QuizSession.id)).where(QuizSession.user_id == user_id)
        )).scalar_one() or 0
    )

    completion_stmt = (
        select(
            func.count(QuizCompletion.id).label("completed_total"),
            func.sum(case((QuizCompletion.ended_by == "timeout", 1), else_=0)).label("timed_out_total"),
            func.avg(QuizCompletion.total_time).label("avg_duration"),
        )
        .join(QuizSession, QuizSession.id == QuizCompletion.session_id)
        .where(QuizSession.user_id == user_id)
    )
    completion_row = (await db.execute(completion_stmt)).one()
    # NOTE: Postgres can return SUM()/CASE-aggregate results as
    # decimal.Decimal rather than a plain int (asyncpg maps numeric-typed
    # aggregates that way), which breaks later float arithmetic — SQLite
    # doesn't have this issue, so this only shows up against the real DB.
    # Explicitly convert to `int` right here rather than relying on `or 0`
    # alone, which only masks the type when the result happens to be zero.
    completed_sessions = int(completion_row.completed_total or 0)
    timed_out_sessions = int(completion_row.timed_out_total or 0)
    average_session_duration_seconds = round(float(completion_row.avg_duration or 0.0), 2)

    average_questions_per_session = round(
        float((await db.execute(
            select(func.avg(QuizSession.question_count)).where(QuizSession.user_id == user_id)
        )).scalar_one() or 0.0),
        2,
    )

    # Abandoned: incomplete sessions inactive longer than the threshold.
    # Loads one row per INCOMPLETE session (bounded by total_sessions, not by
    # attempt volume), then does the "older than N hours" comparison in
    # Python with explicit UTC normalization — SQLite (used in tests/local
    # dev) drops tzinfo on read, and comparing that against a tz-aware
    # cutoff safely is simpler and more portable done here than as raw SQL
    # across two dialects.
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
            QuizSession.user_id == user_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# 7-8. Graded totals (correct/incorrect/unanswered) — the source of truth for
#      overall_accuracy
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_graded_totals(db: AsyncSession, user_id: int) -> GradedTotals:
    """
    Weighted, attempt-level totals — deliberately NOT an average of each
    subject's own accuracy (see AnalyticsSummaryService's docstring for why
    that distinction matters).
    """
    graded_stmt = (
        select(
            func.count(QuestionAttempt.id).label("graded_total"),
            func.sum(cast(QuestionAttempt.correct, Integer)).label("correct_sum"),
        )
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .where(
            QuizSession.user_id == user_id,
            QuestionAttempt.correct.is_not(None),
        )
    )
    graded_row = (await db.execute(graded_stmt)).one()
    graded_total = int(graded_row.graded_total or 0)
    total_correct_answers = int(graded_row.correct_sum or 0)
    total_incorrect_answers = graded_total - total_correct_answers

    # Unanswered questions: session.question_count vs recorded attempts.
    # Per-session attempt counts via a subquery, then compare against how
    # many questions that session actually had. Still a single aggregate
    # query — no per-attempt or per-session rows are loaded into Python.
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
        .where(QuizSession.user_id == user_id)
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


# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-topic breakdown
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_topic_rows(db: AsyncSession, user_id: int) -> list[TopicRow]:
    """
    Grouped by (question.subject, question.lesson) — NOT QuizSession.lesson,
    since one session can span multiple lessons (see quiz_service.py's
    random-per-question lesson assignment). `last_attempted_at` prefers each
    attempt's QuizCompletion.completed_at (always present in practice —
    attempts are only ever created inside submit_quiz(), in the same
    transaction as the QuizCompletion row); QuizSession.created_at is a
    defensive fallback only.
    """
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
            QuizSession.user_id == user_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# 10. Response-time raw rows (validity-filtered)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_response_time_rows(db: AsyncSession, user_id: int) -> list[ResponseTimeRow]:
    """
    One bounded fetch of every VALID response time this user has (negative,
    null, and implausibly-large values excluded via the WHERE clause here —
    raw QuestionAttempt rows are never touched). median/stddev genuinely
    can't be recombined from grouped sub-aggregates (a median of a union
    isn't derivable from its parts' medians), so a single Python-side pass
    over this one list computes every scope (overall/subject/topic) — see
    scoring_service.compute_median_and_stddev() for why that's plain Python
    rather than dialect-specific SQL.
    """
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
            QuizSession.user_id == user_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# 11-12. Performance trend raw data
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_session_completed_at(db: AsyncSession, user_id: int) -> dict[int, datetime]:
    stmt = (
        select(
            QuizSession.id.label("session_id"),
            QuizCompletion.completed_at.label("completed_at"),
        )
        .select_from(QuizSession)
        .join(QuizCompletion, QuizCompletion.session_id == QuizSession.id)
        .where(QuizSession.user_id == user_id)
    )
    rows = (await db.execute(stmt)).all()
    return {row.session_id: as_utc(row.completed_at) for row in rows}


async def fetch_trend_attempt_rows(db: AsyncSession, user_id: int) -> list[TrendAttemptRow]:
    """Graded attempts belonging to a COMPLETED session only — trend is
    defined over completed sessions (see scoring_service.compute_performance_trend)."""
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
            QuizSession.user_id == user_id,
            QuestionAttempt.correct.is_not(None),
        )
    )
    rows = (await db.execute(stmt)).all()
    return [
        TrendAttemptRow(session_id=r.session_id, subject=r.subject, topic=r.topic, correct=bool(r.correct))
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 13. Repeated-question raw rows (chronologically ordered)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_repeated_attempt_rows(db: AsyncSession, user_id: int) -> list[RepeatedAttemptRow]:
    """
    Chronological ordering uses (QuizSession.created_at, QuestionAttempt.id)
    — QuestionAttempt has no per-attempt timestamp of its own in this
    schema, so session creation time plus the attempt's own insertion-order
    primary key (a stable, deterministic tiebreak for attempts within the
    same session) is the best available proxy, and is exact for the common
    case of one attempt per session on a given fingerprint.
    """
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
            QuizSession.user_id == user_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# 14-16. Difficulty-level raw rows
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_difficulty_attempt_rows(db: AsyncSession, user_id: int) -> list[DifficultyAttemptRow]:
    """
    Grouped by each QUESTION's own `difficulty` (not QuizSession.difficulty)
    — every question in a session shares the session's difficulty in this
    app's current generation flow, but grouping at the question level
    handles a mixed-difficulty session correctly too, rather than assuming
    session-level uniformity that might not always hold.
    """
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
            QuizSession.user_id == user_id,
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
    """
    Distinct COMPLETED sessions per (subject, difficulty). A session with
    graded attempts at more than one difficulty — if that's ever possible —
    correctly counts toward every difficulty bucket it touches, since this
    is a separate GROUP BY over the same join, not derived from
    fetch_difficulty_attempt_rows().
    """
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
            QuizSession.user_id == user_id,
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
    """
    Per-(subject, topic, difficulty) accuracy — used only to feed topic-
    level mastery's difficulty_score component (see MasteryScoreService).
    Deliberately not sourced from LessonMastery: its `lesson` column stores
    the raw lesson string, which isn't guaranteed to match the normalized
    topic label used throughout this endpoint (blank/whitespace lessons
    collapse to "Unknown Topic" here, but not necessarily in LessonMastery).
    """
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
            QuizSession.user_id == user_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# 17-18. Growth rolling-window raw rows
# ─────────────────────────────────────────────────────────────────────────────

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
        .where(QuizSession.user_id == user_id, QuizSession.created_at >= window_start)
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
    """
    Graded attempts within the window — the ONLY source for effort's
    attempted_question_count/active_learning_days/session-spacing, so a
    rapidly created, unanswered session contributes to none of them (it can
    only ever hurt completion_rate/abandonment, computed from
    fetch_growth_session_rows() instead).
    """
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
            QuizSession.user_id == user_id,
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
