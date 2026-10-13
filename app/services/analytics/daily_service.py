from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.analytics import queries
from app.services.analytics.types import DailySessionRow
from app.services.quiz_service import get_or_create_user


def _accuracy(correct: int, graded: int) -> float | None:
    # Weighted by question count (same as overall_accuracy / performance_trend); None, not 0, on a day with no
    # graded answers so the client can tell "no activity" apart from "got everything wrong".
    return round(correct / graded * 100.0, 2) if graded > 0 else None


def _avg_response_time(rt_sum: float, rt_count: int) -> float | None:
    return round(rt_sum / rt_count, 3) if rt_count > 0 else None


def progress_score(accuracy: float, baseline_accuracy: float | None, questions_answered: int) -> dict[str, float]:
    # Daily Progress Score, 0-100: a weighted blend of
    #   accuracy     -- that day's accuracy,
    #   improvement  -- 50 + (day accuracy - the student's own accuracy over everything BEFORE that day), clamped to
    #                   0-100, so 50 = "same as usual" and +/-50 points saturates; 50 (neutral) with no prior history,
    #   effort       -- questions answered vs PROGRESS_SCORE_DAILY_QUESTION_TARGET, capped at 100.
    # Comparing against the student's own baseline (not other students) is what makes it an improvement measure.
    improvement = 50.0 if baseline_accuracy is None else min(100.0, max(0.0, 50.0 + accuracy - baseline_accuracy))
    effort = min(100.0, questions_answered / settings.PROGRESS_SCORE_DAILY_QUESTION_TARGET * 100.0)
    score = (
        settings.PROGRESS_SCORE_WEIGHT_ACCURACY * accuracy
        + settings.PROGRESS_SCORE_WEIGHT_IMPROVEMENT * improvement
        + settings.PROGRESS_SCORE_WEIGHT_EFFORT * effort
    )
    return {
        "progress_score": round(min(100.0, max(0.0, score)), 1),
        "score_accuracy": round(accuracy, 1),
        "score_improvement": round(improvement, 1),
        "score_effort": round(effort, 1),
    }


def build_daily_analytics(rows: list[DailySessionRow], *, days: int, tz: ZoneInfo, now: datetime) -> dict:
    # Buckets quizzes by the local calendar date they were completed on in `tz` -- day boundaries must follow the
    # student's clock, not UTC (a 01:00 quiz in Sri Lanka, UTC+5:30, is still "today" there but "yesterday" in
    # UTC). Returns every day in the window oldest-first, including zero-activity days, so charts have no gaps.
    # `rows` may span the student's whole history: days before the window only feed the progress-score baseline.
    end_date: date = now.astimezone(tz).date()
    start_date = end_date - timedelta(days=days - 1)

    buckets: dict[date, dict[str, float]] = defaultdict(
        lambda: {"quizzes": 0, "graded": 0, "correct": 0, "study_time": 0.0, "rt_sum": 0.0, "rt_count": 0}
    )
    for row in rows:
        local_day = row.completed_at.astimezone(tz).date()
        if local_day > end_date:
            continue
        b = buckets[local_day]
        b["quizzes"] += 1
        b["graded"] += row.graded_count
        b["correct"] += row.correct_count
        b["study_time"] += row.study_time
        b["rt_sum"] += row.response_time_sum
        b["rt_count"] += row.response_time_count

    daily = []
    for offset in range(days):
        day = start_date + timedelta(days=offset)
        b = buckets.get(day)
        if b is None or b["graded"] == 0:
            daily.append({
                "date": day, "quizzes_completed": int(b["quizzes"]) if b else 0, "questions_answered": 0,
                "correct_answers": 0, "accuracy": None,
                "study_time_seconds": round(b["study_time"], 2) if b else 0.0, "avg_response_time": None,
                "baseline_accuracy": None, "progress_score": None,
                "score_accuracy": None, "score_improvement": None, "score_effort": None,
            })
            continue

        prior_graded = sum(v["graded"] for d, v in buckets.items() if d < day)
        prior_correct = sum(v["correct"] for d, v in buckets.items() if d < day)
        baseline = _accuracy(int(prior_correct), int(prior_graded))
        accuracy = round(b["correct"] / b["graded"] * 100.0, 2)  # b["graded"] > 0 on this branch

        daily.append({
            "date": day,
            "quizzes_completed": int(b["quizzes"]),
            "questions_answered": int(b["graded"]),
            "correct_answers": int(b["correct"]),
            "accuracy": accuracy,
            "study_time_seconds": round(b["study_time"], 2),
            "avg_response_time": _avg_response_time(b["rt_sum"], int(b["rt_count"])),
            "baseline_accuracy": baseline,
            **progress_score(accuracy, baseline, int(b["graded"])),
        })

    window = [buckets[d["date"]] for d in daily if d["date"] in buckets]
    totals = {key: sum(b[key] for b in window) for key in ("quizzes", "graded", "correct", "study_time", "rt_sum", "rt_count")}
    scored = [d["progress_score"] for d in daily if d["progress_score"] is not None]
    return {
        "days": days,
        "timezone": tz.key,
        "start_date": start_date,
        "end_date": end_date,
        "active_days": sum(1 for d in daily if d["quizzes_completed"] > 0),
        "total_quizzes_completed": int(totals["quizzes"]),
        "total_questions_answered": int(totals["graded"]),
        "total_correct_answers": int(totals["correct"]),
        "accuracy": _accuracy(int(totals["correct"]), int(totals["graded"])),
        "total_study_time_seconds": round(totals["study_time"], 2),
        "avg_response_time": _avg_response_time(totals["rt_sum"], int(totals["rt_count"])),
        # Mean of the active days' scores -- rest days don't drag the week down, they just aren't counted.
        "weekly_progress_score": round(sum(scored) / len(scored), 1) if scored else None,
        "daily": daily,
    }


async def get_daily_analytics(db: AsyncSession, clerk_id: str, *, days: int, tz: ZoneInfo) -> dict:
    user = await get_or_create_user(db, clerk_id)
    # Full history, not just the window: each day's progress score compares it against everything before it.
    rows = await queries.fetch_daily_session_rows(db, user.id)
    return build_daily_analytics(rows, days=days, tz=tz, now=datetime.now(timezone.utc))
