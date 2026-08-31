"""
tests/test_groq_reference_material.py
──────────────────────────────────────
Covers the RAG grounding params on app.services.groq_service.
generate_questions()/generate_shuffle_quiz_questions(): reference_snippets,
reference_by_lesson, and reference_by_lesson_by_subject. These are optional
excerpts supplied by app/services/rag_service.py via app/services/
quiz_service.py's call sites (see tests/test_rag_service.py for rag_service
itself) -- when present, the prompt sent to Groq must actually include them
and the REFERENCE MATERIAL RULES instructions; when absent, the prompt must
be byte-for-byte unaffected so existing (non-RAG) behavior never changes.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import groq_service


def _fake_chat_completion(questions: list[dict]):
    content = json.dumps({"questions": questions})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _sent_prompt(mock_create, role: str) -> str:
    sent_messages = mock_create.call_args.kwargs["messages"]
    return next(m["content"] for m in sent_messages if m["role"] == role)


@pytest.mark.asyncio
async def test_lesson_pinned_prompt_includes_reference_snippets():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Fractions"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_questions(
            grade=10, subject="Mathematics", lesson="Fractions", difficulty="easy", question_count=1,
            reference_snippets=["A fraction represents a part of a whole."],
        )

    user_prompt = _sent_prompt(mock_create, "user")
    system_prompt = _sent_prompt(mock_create, "system")
    assert "A fraction represents a part of a whole." in user_prompt
    assert "REFERENCE MATERIAL" in user_prompt
    assert "REFERENCE MATERIAL RULES" in system_prompt


@pytest.mark.asyncio
async def test_random_lesson_curriculum_prompt_includes_per_lesson_reference():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Fractions"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_questions(
            grade=10, subject="Mathematics", difficulty="easy", question_count=1,
            lesson_choices=["Fractions", "Percentages"],
            reference_by_lesson={"Fractions": ["A fraction represents a part of a whole."]},
        )

    user_prompt = _sent_prompt(mock_create, "user")
    assert "A fraction represents a part of a whole." in user_prompt
    assert "Reference:" in user_prompt
    # Percentages has no reference material supplied -- shouldn't gain one.
    assert user_prompt.count("Reference:") == 1


@pytest.mark.asyncio
async def test_no_reference_material_leaves_prompt_unchanged():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_answer": "A",
             "explanation": "x", "lesson": "Fractions"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_questions(
            grade=10, subject="Mathematics", lesson="Fractions", difficulty="easy", question_count=1,
        )

    user_prompt = _sent_prompt(mock_create, "user")
    system_prompt = _sent_prompt(mock_create, "system")
    assert "REFERENCE MATERIAL" not in user_prompt
    assert "REFERENCE MATERIAL RULES" not in system_prompt


@pytest.mark.asyncio
async def test_shuffle_prompt_includes_per_subject_lesson_reference():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"subject": "Mathematics", "lesson": "Fractions", "question": "Q1?",
             "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "x"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_shuffle_quiz_questions(
            grade=10,
            subject_plan=[("Mathematics", "easy", 1)],
            lesson_choices_by_subject={"Mathematics": ["Fractions", "Percentages"]},
            reference_by_lesson_by_subject={"Mathematics": {"Fractions": ["A fraction is part of a whole."]}},
        )

    user_prompt = _sent_prompt(mock_create, "user")
    system_prompt = _sent_prompt(mock_create, "system")
    assert "A fraction is part of a whole." in user_prompt
    assert "Reference:" in user_prompt
    assert "REFERENCE MATERIAL RULES" in system_prompt


@pytest.mark.asyncio
async def test_shuffle_without_reference_material_leaves_prompt_unchanged():
    mock_create = AsyncMock(
        return_value=_fake_chat_completion([
            {"subject": "Mathematics", "lesson": "Fractions", "question": "Q1?",
             "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "x"},
        ])
    )
    with patch.object(groq_service._groq_client.chat.completions, "create", mock_create):
        await groq_service.generate_shuffle_quiz_questions(
            grade=10,
            subject_plan=[("Mathematics", "easy", 1)],
        )

    system_prompt = _sent_prompt(mock_create, "system")
    assert "REFERENCE MATERIAL RULES" not in system_prompt
