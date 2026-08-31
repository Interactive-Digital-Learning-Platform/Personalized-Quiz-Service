"""
tests/test_groq_curriculum_lessons.py
────────────────────────────────────────
Covers the curriculum-constrained lesson path in
app.services.groq_service.generate_questions()/generate_shuffle_quiz_questions():
when `lesson_choices` (a fixed syllabus lesson list) is supplied, the prompt
sent to Groq must include it, and any returned "lesson" that doesn't match
the fixed list verbatim must be coerced onto one of the fixed lessons
(never dropped, never left as the AI's invented name) — see
app/services/curriculum_service.py for where the fixed list itself comes
from.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import groq_service


def _fake_chat_completion(questions: list[dict]):
    content = json.dumps({"questions": questions})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_lesson_choices_are_included_in_the_prompt():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Fractions"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=1,
            lesson_choices=["Fractions", "Percentages", "Equations"],
        )

    sent_messages = mock_create.call_args.kwargs["messages"]
    user_prompt = next(m["content"] for m in sent_messages if m["role"] == "user")
    assert "Fractions" in user_prompt
    assert "Percentages" in user_prompt
    assert "Equations" in user_prompt
    assert "EXACTLY this fixed syllabus list" in user_prompt


@pytest.mark.asyncio
async def test_matching_lesson_is_kept_verbatim():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Percentages"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=1,
            lesson_choices=["Fractions", "Percentages", "Equations"],
        )

    assert result[0]["lesson"] == "Percentages"


@pytest.mark.asyncio
async def test_non_matching_lesson_is_coerced_not_dropped():
    # The AI invents a topic name that isn't in the fixed list at all.
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Something Made Up"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=1,
            lesson_choices=["Fractions", "Percentages", "Equations"],
        )

    assert len(result) == 1  # not dropped
    assert result[0]["lesson"] in {"Fractions", "Percentages", "Equations"}


@pytest.mark.asyncio
async def test_lesson_matching_is_case_insensitive():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "fractions"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=1,
            lesson_choices=["Fractions", "Percentages", "Equations"],
        )

    assert result[0]["lesson"] == "Fractions"  # snapped to canonical casing


@pytest.mark.asyncio
async def test_without_lesson_choices_behaves_as_before():
    # No curriculum constraint supplied (e.g. a grade without curriculum
    # data) -- the AI's own invented lesson name is kept as-is.
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Whatever The AI Invented"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        result = await groq_service.generate_questions(
            grade=7, subject="Mathematics", difficulty="easy", question_count=1,
        )

    assert result[0]["lesson"] == "Whatever The AI Invented"
