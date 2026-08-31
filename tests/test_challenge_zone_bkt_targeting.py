"""
tests/test_challenge_zone_bkt_targeting.py
──────────────────────────────────────────────
Confirms BKT's p_know is actually blended into automatic quiz generation's
weak-lesson targeting (quiz_service.py's challenge_zone_active block) — the
math for the blend itself is unit-tested in test_difficulty_mastery_engine.py
and test_bkt_service.py; this test verifies the wiring end-to-end by
capturing the real Groq prompt generate_quiz() produces.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.models.lesson_mastery import LessonMastery
from app.services import bkt_service, groq_service
from app.services.quiz_service import generate_quiz, get_or_create_user
from app.schemas.quiz import GenerateQuizRequest
from tests.conftest import TEST_CLERK_ID


def _fake_chat_completion(questions: list[dict]):
    content = json.dumps({"questions": questions})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _canned_questions(lesson: str, count: int) -> list[dict]:
    return [
        {
            "question": f"Q{i}?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
            "explanation": "because", "lesson": lesson,
        }
        for i in range(count)
    ]


async def test_bkt_scores_influence_preferred_lessons_in_generation_prompt(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)

    # CEWM alone would already favor "Percentage" a little (60 vs 75), but
    # not enough to necessarily be picked over other Grade 10 Mathematics
    # lessons the curriculum defines — BKT (p_know=0.02, near-zero) is what
    # makes this lesson unambiguously the weakest once blended.
    db_session.add(LessonMastery(user_id=user.id, subject="Mathematics", lesson="Percentage", grade=10, difficulty="easy", mastery_score=60.0))
    db_session.add(LessonMastery(user_id=user.id, subject="Mathematics", lesson="Indices", grade=10, difficulty="easy", mastery_score=75.0))
    await db_session.commit()

    state_cache: bkt_service.StateCache = {}
    weak_state = await bkt_service.get_or_create_state(
        db_session, user_id=user.id, subject="Mathematics", lesson="Percentage", grade=10, cache=state_cache,
    )
    weak_state.p_know = 0.02
    weak_state.opportunities = 5
    strong_state = await bkt_service.get_or_create_state(
        db_session, user_id=user.id, subject="Mathematics", lesson="Indices", grade=10, cache=state_cache,
    )
    strong_state.p_know = 0.90
    strong_state.opportunities = 5
    await db_session.commit()

    mock_create = AsyncMock(return_value=_fake_chat_completion(_canned_questions("Percentage", 5)))
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        payload = GenerateQuizRequest(grade=10, subject="Mathematics", question_count=5)
        await generate_quiz(db_session, TEST_CLERK_ID, payload)

    assert mock_create.await_count >= 1
    sent_prompt = mock_create.await_args.kwargs["messages"][1]["content"]
    assert "Percentage" in sent_prompt
    assert "would benefit from extra practice" in sent_prompt


async def test_bkt_weight_zero_falls_back_to_pure_cewm_targeting(db_session, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "BKT_LESSON_TARGETING_WEIGHT", 0.0)

    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    # CEWM says "Indices" is weakest; BKT (if it were consulted) would say
    # the opposite. With weight=0.0 the blend must reproduce CEWM exactly.
    db_session.add(LessonMastery(user_id=user.id, subject="Mathematics", lesson="Percentage", grade=10, difficulty="easy", mastery_score=90.0))
    db_session.add(LessonMastery(user_id=user.id, subject="Mathematics", lesson="Indices", grade=10, difficulty="easy", mastery_score=10.0))
    await db_session.commit()

    state_cache: bkt_service.StateCache = {}
    percentage_state = await bkt_service.get_or_create_state(
        db_session, user_id=user.id, subject="Mathematics", lesson="Percentage", grade=10, cache=state_cache,
    )
    percentage_state.p_know = 0.02  # BKT thinks Percentage is the weak one
    indices_state = await bkt_service.get_or_create_state(
        db_session, user_id=user.id, subject="Mathematics", lesson="Indices", grade=10, cache=state_cache,
    )
    indices_state.p_know = 0.95
    await db_session.commit()

    mock_create = AsyncMock(return_value=_fake_chat_completion(_canned_questions("Indices", 5)))
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        payload = GenerateQuizRequest(grade=10, subject="Mathematics", question_count=5)
        await generate_quiz(db_session, TEST_CLERK_ID, payload)

    sent_prompt = mock_create.await_args.kwargs["messages"][1]["content"]
    assert "Indices" in sent_prompt
