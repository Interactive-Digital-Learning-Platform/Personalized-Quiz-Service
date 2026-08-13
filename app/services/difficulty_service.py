import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.lesson_mastery import LessonMastery
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.models.quiz_tracking import QuizCompletion
from app.models.subject_mastery import SubjectMastery
from app.services import difficulty_mastery_engine as engine

logger = logging.getLogger(__name__)

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DEFAULT_DIFFICULTY = "easy"
# Canonical definition now lives in difficulty_mastery_engine.py (which this
# module depends on); re-exported here since app/services/analytics/
# mastery_service.py reads it as difficulty_service.DIFFICULTY_BASE_SCORES.
DIFFICULTY_BASE_SCORES = engine.DIFFICULTY_BASE_SCORES

# Legacy streak-based thresholds. consecutive_strong/consecutive_weak still
# update from these (kept as secondary/informational evidence per the
# Continuous Evidence-Weighted Mastery System spec) but no longer drive
# mastery.difficulty directly -- determine_difficulty_transition() in
# difficulty_mastery_engine.py does that now.
PROMOTE_ACCURACY_THRESHOLD = 80.0
DEMOTE_ACCURACY_THRESHOLD = 40.0
PROMOTE_STREAK_REQUIRED = 2
DEMOTE_STREAK_REQUIRED = 1


class _MasteryRow(Protocol):
    difficulty: str
    last_accuracy: float
    consecutive_strong: int
    consecutive_weak: int
    mastery_score: float
    fluency_score: float
    confidence_score: float
    evidence_count: int
    recent_accuracy: float | None
    previous_accuracy: float | None
    trend_score: float | None
    trend_label: str
    retention_score: float | None
    last_mastery_update: datetime | None


@dataclass
class GradedAnswer:
    """One graded question from a submission, as much as the mastery engine
    needs -- built once by quiz_service.submit_quiz() (which already grades
    each answer) and passed to both update_mastery_after_submission() and
    update_subject_mastery_after_submission() so neither has to re-derive
    correctness or re-fetch Question rows.
    """
    lesson: str
    difficulty: str
    correct: bool | None
    response_time: float
    fingerprint: str


def _step_difficulty(current: str, direction: int) -> str:
    idx = DIFFICULTY_LEVELS.index(current)
    new_idx = max(0, min(len(DIFFICULTY_LEVELS) - 1, idx + direction))
    return DIFFICULTY_LEVELS[new_idx]


def _clamp_percentage(value: float) -> float:
    return max(0.0, min(100.0, value))


def _update_secondary_streak_counters(mastery: _MasteryRow, accuracy: float) -> None:
    # Same streak definition as before promotion/demotion moved to the new
    # engine -- kept purely as secondary/informational evidence now, per
    # the spec's instruction to preserve it rather than drop it.
    if accuracy >= PROMOTE_ACCURACY_THRESHOLD:
        mastery.consecutive_strong += 1
        mastery.consecutive_weak = 0
    elif accuracy <= DEMOTE_ACCURACY_THRESHOLD:
        mastery.consecutive_weak += 1
        mastery.consecutive_strong = 0
    else:
        mastery.consecutive_strong = 0
        mastery.consecutive_weak = 0
    mastery.last_accuracy = round(accuracy, 2)


def _promotion_requirements(current_difficulty: str) -> tuple[float, int, int] | None:
    """(mastery_threshold, min_evidence_count, min_qualifying_completions)
    for promoting OUT of current_difficulty, or None at the top tier."""
    b = settings
    if current_difficulty == "easy":
        return (
            b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_THRESHOLD,
            b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_EVIDENCE_COUNT,
            b.ADAPTIVE_MASTERY_EASY_TO_MEDIUM_MIN_QUALIFYING_COMPLETIONS,
        )
    if current_difficulty == "medium":
        return (
            b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_THRESHOLD,
            b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_EVIDENCE_COUNT,
            b.ADAPTIVE_MASTERY_MEDIUM_TO_HARD_MIN_QUALIFYING_COMPLETIONS,
        )
    return None


def _demotion_threshold(current_difficulty: str) -> float | None:
    b = settings
    if current_difficulty == "medium":
        return b.ADAPTIVE_MASTERY_MEDIUM_TO_EASY_THRESHOLD
    if current_difficulty == "hard":
        return b.ADAPTIVE_MASTERY_HARD_TO_MEDIUM_THRESHOLD
    return None


def _count_qualifying_and_weak(accuracies: list[float], current_difficulty: str) -> tuple[int, int]:
    requirements = _promotion_requirements(current_difficulty)
    promote_threshold = requirements[0] if requirements else None
    demote_threshold = _demotion_threshold(current_difficulty)
    qualifying = sum(1 for a in accuracies if promote_threshold is not None and a >= promote_threshold)
    weak = sum(1 for a in accuracies if demote_threshold is not None and a <= demote_threshold)
    return qualifying, weak


def _log_mastery_change(
    *, user_id: int, subject: str, lesson: str | None,
    old_mastery: float, new_mastery: float, quiz_evidence: float,
    old_difficulty: str, new_difficulty: str, confidence: float, trend_label: str,
) -> None:
    # Deliberately excludes question text/selected answers -- only
    # aggregate numbers, per the spec's requirement not to log that data.
    scope = f"subject={subject}" if lesson is None else f"subject={subject} lesson={lesson}"
    logger.info(
        "Mastery updated: user=%s %s mastery=%.2f->%.2f quiz_evidence=%.2f "
        "difficulty=%s->%s confidence=%.2f trend=%s",
        user_id, scope, old_mastery, new_mastery, quiz_evidence,
        old_difficulty, new_difficulty, confidence, trend_label,
    )


# ── Row access ────────────────────────────────────────────────────────────

async def _get_lesson_mastery_row(
    db: AsyncSession, user_id: int, subject: str, lesson: str, *, for_update: bool = False
) -> LessonMastery | None:
    stmt = select(LessonMastery).where(
        LessonMastery.user_id == user_id,
        LessonMastery.subject == subject,
        LessonMastery.lesson == lesson,
    )
    if for_update:
        stmt = stmt.with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def _get_all_lesson_mastery_rows(db: AsyncSession, user_id: int, subject: str) -> list[LessonMastery]:
    stmt = select(LessonMastery).where(LessonMastery.user_id == user_id, LessonMastery.subject == subject)
    return list((await db.execute(stmt)).scalars().all())


async def _get_subject_mastery_row(
    db: AsyncSession, user_id: int, subject: str, *, for_update: bool = False
) -> SubjectMastery | None:
    stmt = select(SubjectMastery).where(
        SubjectMastery.user_id == user_id,
        SubjectMastery.subject == subject,
    )
    if for_update:
        stmt = stmt.with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_current_difficulty(db: AsyncSession, user_id: int, subject: str, lesson: str) -> str:
    mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def get_subject_difficulty(db: AsyncSession, user_id: int, subject: str) -> str:
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    return mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY


async def get_subject_mastery_score(db: AsyncSession, user_id: int, subject: str) -> float:
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    return mastery.mastery_score if mastery is not None else 50.0


async def get_subject_adaptive_summary(db: AsyncSession, user_id: int, subject: str) -> dict:
    """mastery_score/confidence_score/trend_label for a subject, for the
    challenge-zone generation profile and the Groq adaptive_context hint --
    a single row fetch shared by both rather than querying twice."""
    mastery = await _get_subject_mastery_row(db, user_id, subject)
    if mastery is None:
        return {"mastery_score": 50.0, "confidence_score": 0.0, "trend_label": "insufficient_data"}
    return {
        "mastery_score": mastery.mastery_score,
        "confidence_score": mastery.confidence_score,
        "trend_label": mastery.trend_label,
    }


async def get_subject_lesson_mastery_scores(db: AsyncSession, user_id: int, subject: str) -> dict[str, float]:
    """lesson -> mastery_score for every lesson with a LessonMastery row in
    this subject, for weak/moderate/strong lesson targeting during
    automatic quiz generation."""
    rows = await _get_all_lesson_mastery_rows(db, user_id, subject)
    return {row.lesson: row.mastery_score for row in rows}


# ── History queries for retention / trend / transition evidence ─────────
# Both of the following deliberately read through soft-deleted sessions
# (no QuizSession.deleted_at filter) -- adaptive difficulty and quiz
# generation are the "generation-critical" side of the existing soft-delete
# split (see QuizSession.deleted_at's docstring / _get_recent_lessons in
# quiz_service.py); only the user-facing analytics dashboard filters
# deleted sessions out.

async def _fetch_prior_fingerprint_timestamps(
    db: AsyncSession, *, user_id: int, session_id: int, session_created_at: datetime, fingerprints: set[str],
) -> dict[str, datetime]:
    if not fingerprints:
        return {}
    from sqlalchemy import func as sa_func

    stmt = (
        select(Question.question_fingerprint, sa_func.max(QuizSession.created_at).label("last_seen_at"))
        .select_from(QuestionAttempt)
        .join(QuizSession, QuizSession.id == QuestionAttempt.session_id)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(
            QuizSession.user_id == user_id,
            QuizSession.id != session_id,
            QuizSession.created_at < session_created_at,
            Question.question_fingerprint.in_(fingerprints),
            QuestionAttempt.correct.is_not(None),
        )
        .group_by(Question.question_fingerprint)
    )
    rows = (await db.execute(stmt)).all()
    return {r.question_fingerprint: r.last_seen_at for r in rows}


@dataclass
class _RecentCompletionRow:
    session_id: int
    correct_count: int
    total_questions: int
    accuracy: float
    created_at: datetime
    lesson_accuracy_breakdown: dict | None


async def _fetch_recent_completions(
    db: AsyncSession, user_id: int, subject: str, limit: int
) -> list[_RecentCompletionRow]:
    stmt = (
        select(
            QuizSession.id, QuizCompletion.correct_count, QuizCompletion.total_questions,
            QuizCompletion.accuracy, QuizSession.created_at, QuizCompletion.lesson_accuracy_breakdown,
        )
        .select_from(QuizCompletion)
        .join(QuizSession, QuizSession.id == QuizCompletion.session_id)
        .where(QuizSession.user_id == user_id, QuizSession.subject == subject)
        .order_by(QuizSession.created_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        _RecentCompletionRow(r.id, r.correct_count, r.total_questions, r.accuracy, r.created_at, r.lesson_accuracy_breakdown)
        for r in rows
    ]


def _extract_subject_history(rows: list[_RecentCompletionRow]) -> list[tuple[int, int, int, datetime]]:
    min_q = settings.ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS
    return [(r.session_id, r.correct_count, r.total_questions, r.created_at) for r in rows if r.total_questions >= min_q]


def _extract_subject_accuracies(rows: list[_RecentCompletionRow]) -> list[float]:
    min_q = settings.ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS
    return [r.accuracy for r in rows if r.total_questions >= min_q]


def _extract_lesson_history(rows: list[_RecentCompletionRow], lesson: str) -> list[tuple[int, int, int, datetime]]:
    min_q = settings.ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS
    result: list[tuple[int, int, int, datetime]] = []
    for r in rows:
        stats = (r.lesson_accuracy_breakdown or {}).get(lesson)
        if stats and stats.get("total", 0) >= min_q:
            result.append((r.session_id, int(stats["correct"]), int(stats["total"]), r.created_at))
    return result


def _extract_lesson_accuracies(rows: list[_RecentCompletionRow], lesson: str) -> list[float]:
    min_q = settings.ADAPTIVE_MASTERY_QUALIFYING_COMPLETION_MIN_QUESTIONS
    result: list[float] = []
    for r in rows:
        stats = (r.lesson_accuracy_breakdown or {}).get(lesson)
        if stats and stats.get("total", 0) >= min_q:
            result.append(float(stats.get("accuracy", 0.0)))
    return result


def _retention_pairs_for(
    answers: list[GradedAnswer], prior_seen: dict[str, datetime], session_created_at: datetime
) -> list[tuple[bool, float]]:
    return [
        (bool(a.correct), (session_created_at - prior_seen[a.fingerprint]).total_seconds() / 86400.0)
        for a in answers
        if a.correct is not None and a.fingerprint in prior_seen
    ]


def _fluency_values_for(answers: list[GradedAnswer]) -> list[float]:
    return [
        engine.calculate_fluency_for_question(difficulty=a.difficulty, response_time=a.response_time, correct=bool(a.correct))
        for a in answers
        if a.correct is not None
    ]


# ── Submission entry points ──────────────────────────────────────────────

async def update_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    lesson_accuracy_breakdown: dict[str, dict],
    session: QuizSession,
    ended_by: str,
    graded_answers: list[GradedAnswer],
) -> None:
    now = datetime.now(timezone.utc)
    completion_evidence = engine.calculate_completion_evidence(
        ended_by=ended_by, answered_count=len(graded_answers), intended_count=session.question_count,
    )
    fingerprints = {a.fingerprint for a in graded_answers if a.fingerprint}
    prior_seen = await _fetch_prior_fingerprint_timestamps(
        db, user_id=user_id, session_id=session.id, session_created_at=session.created_at, fingerprints=fingerprints,
    )
    recent_completions = await _fetch_recent_completions(
        db, user_id, subject, settings.ADAPTIVE_MASTERY_TRANSITION_LOOKBACK_QUIZZES
    )

    for lesson, stats in lesson_accuracy_breakdown.items():
        if lesson == "unknown":
            continue

        accuracy = float(stats.get("accuracy", 0.0))
        lesson_answers = [a for a in graded_answers if a.lesson == lesson]
        # A lesson's questions within one submission are generated at a
        # single difficulty in practice, so the first answer is representative.
        lesson_difficulty = lesson_answers[0].difficulty if lesson_answers else DEFAULT_DIFFICULTY

        retention_evidence = engine.calculate_retention_evidence(
            _retention_pairs_for(lesson_answers, prior_seen, session.created_at)
        )
        fluency_values = _fluency_values_for(lesson_answers)

        mastery = await _get_lesson_mastery_row(db, user_id, subject, lesson, for_update=True)
        if mastery is None:
            mastery = LessonMastery(
                user_id=user_id, subject=subject, lesson=lesson, difficulty=DEFAULT_DIFFICULTY,
                last_accuracy=0.0, consecutive_strong=0, consecutive_weak=0,
                mastery_score=50.0, fluency_score=50.0, confidence_score=0.0, evidence_count=0,
                trend_label="insufficient_data",
            )
            db.add(mastery)

        old_difficulty = mastery.difficulty
        old_mastery_score = mastery.mastery_score

        _update_secondary_streak_counters(mastery, accuracy)

        quiz_evidence = engine.calculate_quiz_evidence(
            accuracy=accuracy, difficulty=lesson_difficulty,
            retention_evidence=retention_evidence, completion_evidence=completion_evidence,
        )
        evidence_count_before = mastery.evidence_count
        mastery.mastery_score = engine.update_mastery_score(mastery.mastery_score, quiz_evidence, evidence_count_before)
        # evidence_count counts QUESTIONS answered, not quizzes submitted --
        # confidence's breakpoints ("0 questions -> 0%, 5 -> low, 15 ->
        # moderate, 30+ -> high") are expressed in questions, and a
        # question-count basis also keeps a 2-question 100% quiz from
        # carrying as much weight as a 20-question one.
        mastery.evidence_count = evidence_count_before + len(lesson_answers)
        mastery.confidence_score = engine.calculate_confidence(mastery.evidence_count)
        mastery.retention_score = retention_evidence
        if fluency_values:
            quiz_fluency = engine.calculate_fluency(fluency_values)
            mastery.fluency_score = engine.update_mastery_score(mastery.fluency_score, quiz_fluency, evidence_count_before)
        mastery.last_mastery_update = now

        trend = engine.calculate_trend(_extract_lesson_history(recent_completions, lesson))
        mastery.recent_accuracy = trend.recent_accuracy
        mastery.previous_accuracy = trend.previous_accuracy
        mastery.trend_score = trend.trend_score
        mastery.trend_label = trend.trend_label

        qualifying, weak = _count_qualifying_and_weak(
            _extract_lesson_accuracies(recent_completions, lesson), mastery.difficulty
        )
        new_difficulty = engine.determine_difficulty_transition(
            current_difficulty=mastery.difficulty,
            mastery_score=mastery.mastery_score,
            evidence_count=mastery.evidence_count,
            confidence_score=mastery.confidence_score,
            qualifying_completions_at_current_tier=qualifying,
            recent_weak_results=weak,
        )
        mastery.difficulty = new_difficulty

        _log_mastery_change(
            user_id=user_id, subject=subject, lesson=lesson,
            old_mastery=old_mastery_score, new_mastery=mastery.mastery_score, quiz_evidence=quiz_evidence,
            old_difficulty=old_difficulty, new_difficulty=new_difficulty,
            confidence=mastery.confidence_score, trend_label=mastery.trend_label,
        )

    await db.commit()


async def update_subject_mastery_after_submission(
    db: AsyncSession,
    user_id: int,
    subject: str,
    accuracy: float,
    session: QuizSession,
    ended_by: str,
    graded_answers: list[GradedAnswer],
) -> None:
    now = datetime.now(timezone.utc)
    completion_evidence = engine.calculate_completion_evidence(
        ended_by=ended_by, answered_count=len(graded_answers), intended_count=session.question_count,
    )
    fingerprints = {a.fingerprint for a in graded_answers if a.fingerprint}
    prior_seen = await _fetch_prior_fingerprint_timestamps(
        db, user_id=user_id, session_id=session.id, session_created_at=session.created_at, fingerprints=fingerprints,
    )
    retention_evidence = engine.calculate_retention_evidence(
        _retention_pairs_for(graded_answers, prior_seen, session.created_at)
    )
    fluency_values = _fluency_values_for(graded_answers)

    mastery = await _get_subject_mastery_row(db, user_id, subject, for_update=True)
    if mastery is None:
        mastery = SubjectMastery(
            user_id=user_id, subject=subject, difficulty=DEFAULT_DIFFICULTY,
            last_accuracy=0.0, consecutive_strong=0, consecutive_weak=0,
            mastery_score=50.0, fluency_score=50.0, confidence_score=0.0, evidence_count=0,
            trend_label="insufficient_data",
        )
        db.add(mastery)

    old_difficulty = mastery.difficulty
    old_mastery_score = mastery.mastery_score

    _update_secondary_streak_counters(mastery, accuracy)

    quiz_evidence = engine.calculate_quiz_evidence(
        accuracy=accuracy, difficulty=mastery.difficulty,
        retention_evidence=retention_evidence, completion_evidence=completion_evidence,
    )
    evidence_count_before = mastery.evidence_count
    direct_mastery_score = engine.update_mastery_score(mastery.mastery_score, quiz_evidence, evidence_count_before)
    # evidence_count counts QUESTIONS answered, not quizzes submitted --
    # see the matching comment in update_mastery_after_submission.
    mastery.evidence_count = evidence_count_before + len(graded_answers)
    mastery.confidence_score = engine.calculate_confidence(mastery.evidence_count)
    mastery.retention_score = retention_evidence
    if fluency_values:
        quiz_fluency = engine.calculate_fluency(fluency_values)
        mastery.fluency_score = engine.update_mastery_score(mastery.fluency_score, quiz_fluency, evidence_count_before)
    mastery.last_mastery_update = now

    # Roll up this subject's lessons (log-capped so one heavily-practiced
    # lesson can't dominate) and blend with the direct quiz-evidence update
    # above -- weighted towards the roll-up so lesson-level mastery carries
    # more say than a bare subject accuracy figure, per spec. Falls back to
    # the direct value alone when there's no lesson data yet (e.g. this
    # submission's own lesson-level update, called separately, hasn't run
    # or every lesson came back "unknown").
    lesson_rows = await _get_all_lesson_mastery_rows(db, user_id, subject)
    rollup = engine.rollup_lesson_mastery_to_subject(lesson_rows)
    if rollup is None:
        mastery.mastery_score = direct_mastery_score
    else:
        mastery.mastery_score = _clamp_percentage(
            rollup * settings.ADAPTIVE_MASTERY_SUBJECT_LESSON_ROLLUP_WEIGHT
            + direct_mastery_score * settings.ADAPTIVE_MASTERY_SUBJECT_DIRECT_EVIDENCE_WEIGHT
        )

    recent_completions = await _fetch_recent_completions(
        db, user_id, subject, settings.ADAPTIVE_MASTERY_TRANSITION_LOOKBACK_QUIZZES
    )
    trend = engine.calculate_trend(_extract_subject_history(recent_completions))
    mastery.recent_accuracy = trend.recent_accuracy
    mastery.previous_accuracy = trend.previous_accuracy
    mastery.trend_score = trend.trend_score
    mastery.trend_label = trend.trend_label

    qualifying, weak = _count_qualifying_and_weak(_extract_subject_accuracies(recent_completions), mastery.difficulty)
    new_difficulty = engine.determine_difficulty_transition(
        current_difficulty=mastery.difficulty,
        mastery_score=mastery.mastery_score,
        evidence_count=mastery.evidence_count,
        confidence_score=mastery.confidence_score,
        qualifying_completions_at_current_tier=qualifying,
        recent_weak_results=weak,
    )
    mastery.difficulty = new_difficulty

    _log_mastery_change(
        user_id=user_id, subject=subject, lesson=None,
        old_mastery=old_mastery_score, new_mastery=mastery.mastery_score, quiz_evidence=quiz_evidence,
        old_difficulty=old_difficulty, new_difficulty=new_difficulty,
        confidence=mastery.confidence_score, trend_label=mastery.trend_label,
    )

    await db.commit()


def describe_subject_mastery(mastery: SubjectMastery | None) -> dict:
    # Read-only projection of a SubjectMastery row into the "how close to
    # promotion/demotion" fields GET /analytics/me shows -- reuses the same
    # helpers the real engine uses (_promotion_requirements,
    # _demotion_threshold) so this can never drift from what actually
    # drives promotion. Never writes to the DB.
    current_difficulty = mastery.difficulty if mastery is not None else DEFAULT_DIFFICULTY
    consecutive_strong = mastery.consecutive_strong if mastery is not None else 0
    consecutive_weak = mastery.consecutive_weak if mastery is not None else 0
    mastery_score = mastery.mastery_score if mastery is not None else 50.0
    evidence_count = mastery.evidence_count if mastery is not None else 0

    next_difficulty = _step_difficulty(current_difficulty, +1)
    at_max_difficulty = next_difficulty == current_difficulty
    requirements = _promotion_requirements(current_difficulty)
    demotion_threshold = _demotion_threshold(current_difficulty)

    if at_max_difficulty or requirements is None:
        promotion_threshold = 100.0
        quizzes_required_for_promotion = 0
        promotion_readiness = 0.0
    else:
        mastery_threshold, min_evidence, min_completions = requirements
        promotion_threshold = mastery_threshold
        quizzes_required_for_promotion = min_completions
        mastery_progress = _clamp_percentage(mastery_score / mastery_threshold * 100.0) if mastery_threshold > 0 else 100.0
        evidence_progress = _clamp_percentage(evidence_count / min_evidence * 100.0) if min_evidence > 0 else 100.0
        promotion_readiness = round(min(mastery_progress, evidence_progress), 2)

    if at_max_difficulty:
        message = f"Already at the highest difficulty ({current_difficulty})."
    elif promotion_readiness >= 100.0:
        message = f"Ready to advance to {next_difficulty} difficulty."
    elif requirements is not None and evidence_count < requirements[1]:
        message = f"Keep practicing {current_difficulty} questions to build enough evidence for {next_difficulty} difficulty."
    else:
        message = f"{round(promotion_readiness)}% of the way to {next_difficulty} difficulty."

    return {
        "current_difficulty": current_difficulty,
        "consecutive_strong_quizzes": consecutive_strong,
        "consecutive_weak_quizzes": consecutive_weak,
        "promotion_threshold": promotion_threshold,
        "demotion_threshold": demotion_threshold if demotion_threshold is not None else DEMOTE_ACCURACY_THRESHOLD,
        "quizzes_required_for_promotion": quizzes_required_for_promotion,
        "promotion_progress_percentage": promotion_readiness,
        "next_difficulty": next_difficulty,
        "difficulty_status_message": message,
        "promotion_readiness": promotion_readiness,
    }
