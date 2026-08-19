"""
tests/test_quiz_generation_pool.py
─────────────────────────────────────
Covers the question-pool-first generation design in quiz_service.py:

- Quiz generation reads pre-generated questions for a (subject, difficulty)
  bucket first (_fetch_pool_questions), excluding whatever the requesting
  user has already been asked (via QuestionAttempt/QuizSession), and only
  calls Groq synchronously for whatever the pool doesn't cover.
- A bucket below settings.POOL_MIN_SIZE triggers a debounced, fire-and-
  forget background top-up (_kickoff_pool_replenish) -- never blocks the
  request that triggered it.
"""
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.models.question import Question
from app.models.quiz_session import QuestionAttempt, QuizSession
from app.services import quiz_service
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
        subject="Mathematics", lesson="Algebra", difficulty="easy",
    )
    db_session.add(seen_question)
    await db_session.flush()
    await _mark_seen(db_session, user.id, [seen_question])

    unseen_question = Question(
        question="Never seen question?", options=["A", "B", "C", "D"], correct_answer="A",
        subject="Mathematics", lesson="Algebra", difficulty="easy",
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
        subject="Mathematics", lesson="Algebra", difficulty="easy",
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
            subject="Mathematics", lesson="Algebra", difficulty="easy",
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
        subject="Mathematics", lesson="Algebra", difficulty="easy",
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
        subject="Mathematics", lesson="Algebra", difficulty="easy",
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
            subject="Mathematics", lesson="Algebra", difficulty="easy",
        ))
    await db_session.commit()

    with patch.object(quiz_service, "_kickoff_pool_replenish") as mock_kickoff:
        await _fetch_pool_questions(
            db_session, user_id=user.id, subject="Mathematics", difficulty="easy", grade=10,
            limit=1, exclude_ids=set(), exclude_texts=[],
        )

    mock_kickoff.assert_not_called()


def test_kickoff_pool_replenish_is_debounced():
    """Concurrent/repeated triggers for the same (subject, difficulty)
    bucket must only ever spawn one in-flight background task."""
    created_coros = []

    def _fake_create_task(coro):
        coro.close()  # never actually run it in this synchronous test
        created_coros.append(coro)

    quiz_service._replenishing_buckets.clear()
    with patch("asyncio.create_task", _fake_create_task):
        _real_kickoff_pool_replenish("Mathematics", "easy", 10)
        _real_kickoff_pool_replenish("Mathematics", "easy", 10)
        _real_kickoff_pool_replenish("Mathematics", "medium", 10)

    assert len(created_coros) == 2  # (Math, easy) once, (Math, medium) once
    quiz_service._replenishing_buckets.clear()
