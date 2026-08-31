"""
tests/test_bkt_submission_flow.py
────────────────────────────────────
Integration tests for BKT's wiring into quiz_service.submit_quiz — the guard
that must skip BKT for retake sessions (mirroring the existing CEWM retake
guard), exercised through the real submit_quiz() function rather than by
calling bkt_service directly (test_bkt_service.py covers that layer).
"""
from sqlalchemy import select

from app.models.question import Question
from app.models.quiz_session import QuizSession
from app.models.skill_bkt_state import SkillBKTState
from app.schemas.quiz import AnswerItem, SubmitQuizRequest
from app.services.quiz_service import create_retake_session, get_or_create_user, submit_quiz
from tests.conftest import TEST_CLERK_ID


async def _make_session_with_questions(
    db, user_id: int, subject: str, lesson: str, grade: int | None, count: int = 5,
) -> tuple[QuizSession, list[Question]]:
    session = QuizSession(
        user_id=user_id, subject=subject, lesson=lesson, difficulty="easy",
        question_count=count, grade=grade,
    )
    db.add(session)
    await db.flush()

    questions = []
    for i in range(count):
        q = Question(
            question=f"Q{i}?", options=["A", "B", "C", "D"], correct_answer="A",
            subject=subject, lesson=lesson, difficulty="easy",
        )
        db.add(q)
        questions.append(q)
    await db.flush()
    await db.commit()
    await db.refresh(session)
    return session, questions


async def _get_bkt_state(db, user_id: int, subject: str, lesson: str, grade: int | None) -> SkillBKTState | None:
    grade_filter = SkillBKTState.grade == grade if grade is not None else SkillBKTState.grade.is_(None)
    stmt = select(SkillBKTState).where(
        SkillBKTState.user_id == user_id, SkillBKTState.subject == subject,
        SkillBKTState.lesson == lesson, grade_filter,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def test_submit_quiz_updates_bkt_state(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    session, questions = await _make_session_with_questions(db_session, user.id, "Mathematics", "Fractions", grade=10)

    payload = SubmitQuizRequest(
        session_id=session.id,
        answers=[
            AnswerItem(question_id=q.id, selected_answer="A", response_time=5.0)
            for q in questions
        ],
        ended_by="submitted",
    )
    await submit_quiz(db_session, TEST_CLERK_ID, payload)

    state = await _get_bkt_state(db_session, user.id, "Mathematics", "Fractions", 10)
    assert state is not None
    assert state.opportunities == 5
    assert state.last_correct is True


async def test_submit_quiz_skips_bkt_for_retake_sessions(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    original, questions = await _make_session_with_questions(db_session, user.id, "Science", "Kinematics", grade=11)

    # Complete the original once, same as a normal quiz.
    await submit_quiz(db_session, TEST_CLERK_ID, SubmitQuizRequest(
        session_id=original.id,
        answers=[AnswerItem(question_id=q.id, selected_answer="A", response_time=5.0) for q in questions],
        ended_by="submitted",
    ))
    state_after_first = await _get_bkt_state(db_session, user.id, "Science", "Kinematics", 11)
    assert state_after_first.opportunities == 5

    retake = await create_retake_session(db_session, TEST_CLERK_ID, original.id)
    retake_questions = (
        await db_session.execute(select(Question).where(Question.subject == "Science", Question.lesson == "Kinematics"))
    ).scalars().all()
    await submit_quiz(db_session, TEST_CLERK_ID, SubmitQuizRequest(
        session_id=retake.id,
        answers=[AnswerItem(question_id=q.id, selected_answer="A", response_time=5.0) for q in retake_questions],
        ended_by="submitted",
    ))

    # The user has already seen every correct answer by the time they
    # retake — a retake submission must never move the BKT estimate, same
    # as it must never move adaptive difficulty (see QuizSession.is_retake).
    state_after_retake = await _get_bkt_state(db_session, user.id, "Science", "Kinematics", 11)
    assert state_after_retake.opportunities == 5
