"""
tests/test_quiz_generation_pool.py
─────────────────────────────────────
Covers the question-pool-first generation design in quiz_service.py:

- Quiz generation reads pre-generated questions for a (subject, difficulty,
  grade) bucket first (_fetch_pool_questions), excluding whatever the
  requesting user has already been asked (via QuestionAttempt/QuizSession),
  and only calls Groq synchronously for whatever the pool doesn't cover.
- A bucket below settings.POOL_MIN_SIZE triggers a debounced, fire-and-
  forget background top-up (_kickoff_pool_replenish) -- never blocks the
  request that triggered it.
- For a grade with curriculum data (10/11 -- see app/data/curriculum/), both
  the pool read and the background top-up must only ever surface/generate
  questions pinned to that grade's real syllabus lessons -- a Grade 10
  request must never be served a Grade 11 (or legacy pre-curriculum)
  question, or one tagged with a lesson name that isn't actually in
  grade_10.json.
"""
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.services import curriculum_service, quiz_service
from app.services.quiz_service import _fetch_pool_questions, get_or_create_user
from tests.conftest import TEST_CLERK_ID

GENERATE_URL = "/api/v1/quiz/generate"

# Captured at import time, before the autouse `_no_real_pool_replenish`
# fixture (conftest.py) monkeypatches quiz_service._kickoff_pool_replenish
# to a no-op for every test -- the debounce test below needs the REAL
# implementation, not that no-op.
_real_kickoff_pool_replenish = quiz_service._kickoff_pool_replenish

# _is_near_duplicate compares significant ALPHA words only -- texts
# differing solely by a digit (e.g. "Question 0?" vs "Question 1?") look
# identical to it, since digits aren't alpha and get stripped from the
# comparison. Distinct words per index avoid tripping that dedup logic
# within a single test's own seed/mock data.
_DISTINCT_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]


def _fresh_question(i: int) -> dict:
    word = _DISTINCT_WORDS[i % len(_DISTINCT_WORDS)]
    return {
        "question": f"Fresh AI {word} question?",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
        "explanation": "because",
        "lesson": "Algebra",
    }


async def _mark_seen(db, user_id: int, questions: list[Question]) -> None:
    session = QuizSession(
        user_id=user_id, subject="Placeholder", lesson="Placeholder", difficulty="easy", question_count=len(questions),
    )
    db.add(session)
    await db.flush()
    for question in questions:
        db.add(QuestionAttempt(session_id=session.id, question_id=question.id, selected_answer="A", correct=True))
    await db.commit()


# ── Per-user exclusion ────────────────────────────────────────────────────

async def test_pool_excludes_questions_this_user_has_already_attempted(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    seen_question = Question(
        question="Already seen question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    )
    db_session.add(seen_question)
    await db_session.flush()
    await _mark_seen(db_session, user.id, [seen_question])

    unseen_question = Question(
        question="Never seen question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    )
    db_session.add(unseen_question)
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish"):
        result = await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=5, exclude_ids=set(), exclude_texts=[],
        )

    returned_texts = {q.question for q in result}
    assert "Never seen question?" in returned_texts
    assert "Already seen question?" not in returned_texts


async def test_pool_serves_a_question_to_a_different_user_who_has_not_seen_it(db_session):
    """The pool is a shared bank, not consumed on first use -- the same
    question can go to a different user who hasn't attempted it yet."""
    seen_by_first_user = await get_or_create_user(db_session, TEST_CLERK_ID)
    other_user = await get_or_create_user(db_session, "another-test-user")

    question = Question(
        question="Shared pool question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    )
    db_session.add(question)
    await db_session.flush()
    await _mark_seen(db_session, seen_by_first_user.id, [question])

    with patch.object(quiz_service, "_kickoff_pool_replenish"):
        result = await _fetch_pool_questions(
            db_session, user_id=other_user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=5, exclude_ids=set(), exclude_texts=[],
        )

    assert {q.question for q in result} == {"Shared pool question?"}


# ── Pool-first ordering: Groq is skipped entirely when the pool covers it ──

async def test_pool_fully_covers_request_groq_never_called(client, db_session):
    expected_texts = {f"Pool {word} question?" for word in _DISTINCT_WORDS[:3]}
    for text in expected_texts:
        db_session.add(Question(
            question=text, options=["A", "B", "C", "D"], correct_answer="A",
            subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
        ))
    await db_session.commit()

    with patch(
        "app.services.quiz_service.generate_questions", new=AsyncMock(),
    ) as mock_generate:
        resp = await client.post(
            GENERATE_URL, json={"subject": "Mathematics", "difficulty": "easy", "question_count": 3},
        )

    assert resp.status_code == 201
    mock_generate.assert_not_called()
    body = resp.json()
    assert len(body["questions"]) == 3
    assert {q["question"] for q in body["questions"]} == expected_texts


async def test_pool_partial_coverage_calls_groq_for_shortfall_only(client, db_session):
    db_session.add(Question(
        question="Pool question only one?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    ))
    await db_session.commit()

    async def _side_effect(*, grade, subject, difficulty, question_count, **_kwargs):
        assert question_count == 2  # only the shortfall, not the full 3
        return [_fresh_question(i) for i in range(question_count)]

    with patch(
        "app.services.quiz_service.generate_questions", new=AsyncMock(side_effect=_side_effect),
    ) as mock_generate:
        resp = await client.post(
            GENERATE_URL, json={"subject": "Mathematics", "difficulty": "easy", "question_count": 3},
        )

    assert resp.status_code == 201
    assert mock_generate.call_count == 1
    body = resp.json()
    assert len(body["questions"]) == 3


# ── Debounced background top-up ─────────────────────────────────────────────

async def test_low_bucket_triggers_exactly_one_replenish_call(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # Bucket has fewer rows than settings.POOL_MIN_SIZE.
    assert settings.POOL_MIN_SIZE > 1
    db_session.add(Question(
        question="Only cached question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    ))
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish") as mock_kickoff:
        await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=1, exclude_ids=set(), exclude_texts=[],
        )

    mock_kickoff.assert_called_once_with("Mathematics", "easy", 10)


async def test_healthy_bucket_does_not_trigger_replenish(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    for i in range(settings.POOL_MIN_SIZE):
        db_session.add(Question(
            question=f"Healthy pool question {i}?", options=["A", "B", "C", "D"], correct_answer="A",
            subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
        ))
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish") as mock_kickoff:
        await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=1, exclude_ids=set(), exclude_texts=[],
        )

    mock_kickoff.assert_not_called()


def test_kickoff_pool_replenish_is_debounced():
    """Concurrent/repeated triggers for the same (subject, difficulty, grade)
    bucket must only ever spawn one in-flight background task -- and a
    different grade must NOT be debounced together with it, since Grade 10
    and Grade 11 buckets for the same subject+difficulty are otherwise
    indistinguishable."""
    created_coros = []

    def _fake_create_task(coro):
        coro.close()  # never actually run it in this synchronous test
        created_coros.append(coro)

    quiz_service._replenishing_buckets.clear()
    with patch("asyncio.create_task", _fake_create_task):
        _real_kickoff_pool_replenish("Mathematics", "easy", 10)
        _real_kickoff_pool_replenish("Mathematics", "easy", 10)
        _real_kickoff_pool_replenish("Mathematics", "medium", 10)
        _real_kickoff_pool_replenish("Mathematics", "easy", 11)

    assert len(created_coros) == 3  # (Math,easy,10) once, (Math,medium,10) once, (Math,easy,11) once
    quiz_service._replenishing_buckets.clear()


# ── Curriculum scoping (Grade 10/11 only ever serve their own syllabus) ────

async def test_pool_excludes_off_curriculum_lesson_even_when_subject_and_grade_match(db_session):
    """A pool row tagged with a lesson that isn't actually in
    app/data/curriculum/grade_10.json (e.g. a legacy row from before
    curriculum constraints existed) must never be served for a Grade 10
    request, even though its subject/difficulty/grade all match."""
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    valid_lesson_question = Question(
        question="Valid curriculum lesson question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    )
    off_curriculum_question = Question(
        question="Off-curriculum lesson question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="easy", grade=10,
    )
    db_session.add_all([valid_lesson_question, off_curriculum_question])
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish"):
        result = await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=5, exclude_ids=set(), exclude_texts=[],
        )

    returned_texts = {q.question for q in result}
    assert "Valid curriculum lesson question?" in returned_texts
    assert "Off-curriculum lesson question?" not in returned_texts


async def test_pool_excludes_a_different_grades_question_even_when_subject_matches(db_session):
    """A Grade 11 pool row must never be served for a Grade 10 request, even
    with an identical subject+difficulty and a lesson name that happens to
    be valid for Grade 11's own curriculum."""
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    grade_10_question = Question(
        question="Grade 10 question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Fractions", difficulty="easy", grade=10,
    )
    grade_11_question = Question(
        question="Grade 11 question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Real Numbers", difficulty="easy", grade=11,
    )
    db_session.add_all([grade_10_question, grade_11_question])
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish"):
        result = await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=5, exclude_ids=set(), exclude_texts=[],
        )

    returned_texts = {q.question for q in result}
    assert "Grade 10 question?" in returned_texts
    assert "Grade 11 question?" not in returned_texts


async def test_replenish_pool_task_constrains_new_questions_to_curriculum_lessons(monkeypatch):
    """_replenish_pool_task() must pass the grade's curriculum lesson list
    into generate_questions() the same way the live per-request path
    (_generate_or_cache_tier) already does -- without this, every
    background-generated pool question for Grade 10/11 got an AI-invented
    free-text lesson name instead of one from grade_10.json/grade_11.json,
    regardless of what _fetch_pool_questions later filters for.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(quiz_service, "AsyncSessionLocal", session_factory)

    captured_kwargs = {}

    async def _fake_generate_questions(**kwargs):
        captured_kwargs.update(kwargs)
        return [{
            "question": "Q?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
            "explanation": "x", "lesson": kwargs["lesson_choices"][0],
        }]

    with patch.object(quiz_service, "generate_questions", AsyncMock(side_effect=_fake_generate_questions)):
        await quiz_service._replenish_pool_task("Mathematics", "easy", 10)

    assert captured_kwargs.get("lesson_choices") == curriculum_service.lessons_for(10, "Mathematics")

    await engine.dispose()
