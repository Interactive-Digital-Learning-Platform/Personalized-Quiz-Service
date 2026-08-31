"""
tests/test_rag_service.py
──────────────────────────
Covers app.services.rag_service: the best-effort RAG grounding layer that
feeds curriculum excerpts (retrieved from Qdrant via app/services/
retrieval_service.py) into groq_service's prompts -- see
app/services/quiz_service.py's call sites. Every public function here must
degrade to an empty result on any failure, timeout, or disabled RAG; quiz
generation must succeed identically whether or not retrieval works, so
nothing here is ever allowed to raise.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services import rag_service


@pytest.fixture(autouse=True)
def _reset_singletons():
    # rag_service caches its embedder/reranker/retrieval_service as module
    # globals -- reset around each test so one test's monkeypatched state
    # never leaks into another.
    rag_service._embedder = None
    rag_service._reranker = None
    rag_service._retrieval_service = None
    yield
    rag_service._embedder = None
    rag_service._reranker = None
    rag_service._retrieval_service = None


def test_build_query_format():
    assert rag_service._build_query(10, "Mathematics", "Fractions") == "Grade 10 Mathematics: Fractions"


@pytest.mark.asyncio
async def test_get_snippets_for_lesson_short_circuits_when_rag_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", False)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")
    mock_ensure = AsyncMock()
    monkeypatch.setattr(rag_service, "_ensure_retrieval_service", mock_ensure)

    result = await rag_service.get_snippets_for_lesson(
        grade=10, subject="Mathematics", lesson="Fractions", max_snippets=3,
    )

    assert result == []
    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_get_snippets_for_lesson_short_circuits_when_qdrant_url_unset(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "")
    mock_ensure = AsyncMock()
    monkeypatch.setattr(rag_service, "_ensure_retrieval_service", mock_ensure)

    result = await rag_service.get_snippets_for_lesson(
        grade=10, subject="Mathematics", lesson="Fractions", max_snippets=3,
    )

    assert result == []
    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_get_snippets_for_lesson_returns_empty_on_retrieval_exception(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")

    fake_service = AsyncMock()
    fake_service.search = AsyncMock(side_effect=RuntimeError("qdrant unreachable"))
    monkeypatch.setattr(rag_service, "_ensure_retrieval_service", AsyncMock(return_value=fake_service))

    result = await rag_service.get_snippets_for_lesson(
        grade=10, subject="Mathematics", lesson="Fractions", max_snippets=3,
    )

    assert result == []


@pytest.mark.asyncio
async def test_get_snippets_for_lesson_returns_empty_on_timeout(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(settings, "RAG_TIMEOUT_SECONDS", 0.01)

    async def _slow_search(*args, **kwargs):
        await asyncio.sleep(1)
        return []

    fake_service = AsyncMock()
    fake_service.search = _slow_search
    monkeypatch.setattr(rag_service, "_ensure_retrieval_service", AsyncMock(return_value=fake_service))

    result = await rag_service.get_snippets_for_lesson(
        grade=10, subject="Mathematics", lesson="Fractions", max_snippets=3,
    )

    assert result == []


@pytest.mark.asyncio
async def test_get_snippets_for_lesson_truncates_and_caps_count(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(settings, "RAG_MAX_SNIPPET_CHARS", 10)

    long_text = "word " * 20
    results = [SearchResult(text=long_text, score=0.9, metadata={}) for _ in range(5)]
    fake_service = AsyncMock()
    fake_service.search = AsyncMock(return_value=results)
    monkeypatch.setattr(rag_service, "_ensure_retrieval_service", AsyncMock(return_value=fake_service))

    result = await rag_service.get_snippets_for_lesson(
        grade=10, subject="Mathematics", lesson="Fractions", max_snippets=2,
    )

    assert len(result) == 2
    assert all(len(s) <= 13 for s in result)  # 10 chars + "..." allowance


@pytest.mark.asyncio
async def test_get_snippets_by_lesson_omits_lessons_with_no_results(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")

    async def _fake_get_snippets_for_lesson(*, grade, subject, lesson, max_snippets):
        return ["a snippet"] if lesson == "Fractions" else []

    monkeypatch.setattr(rag_service, "get_snippets_for_lesson", _fake_get_snippets_for_lesson)

    result = await rag_service.get_snippets_by_lesson(
        grade=10, subject="Mathematics", lessons=["Fractions", "Percentages"],
        max_snippets_per_lesson=2, max_lessons=8,
    )

    assert result == {"Fractions": ["a snippet"]}


@pytest.mark.asyncio
async def test_get_snippets_by_lesson_respects_max_lessons_cap(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")

    seen_lessons = []

    async def _fake_get_snippets_for_lesson(*, grade, subject, lesson, max_snippets):
        seen_lessons.append(lesson)
        return ["x"]

    monkeypatch.setattr(rag_service, "get_snippets_for_lesson", _fake_get_snippets_for_lesson)

    lessons = [f"Lesson {i}" for i in range(10)]
    await rag_service.get_snippets_by_lesson(
        grade=10, subject="Mathematics", lessons=lessons,
        max_snippets_per_lesson=2, max_lessons=3,
    )

    assert len(seen_lessons) == 3


@pytest.mark.asyncio
async def test_get_snippets_by_lesson_for_subjects_respects_combined_cap(monkeypatch):
    monkeypatch.setattr(settings, "RAG_ENABLED", True)
    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")

    seen_pairs = []

    async def _fake_get_snippets_for_lesson(*, grade, subject, lesson, max_snippets):
        seen_pairs.append((subject, lesson))
        return ["x"]

    monkeypatch.setattr(rag_service, "get_snippets_for_lesson", _fake_get_snippets_for_lesson)

    subjects_lessons = {
        "Mathematics": ["Fractions", "Percentages", "Equations"],
        "Science": ["Photosynthesis", "Cells", "Forces"],
    }
    result = await rag_service.get_snippets_by_lesson_for_subjects(
        grade=10, subjects_lessons=subjects_lessons,
        max_snippets_per_lesson=2, max_total_lesson_queries=4,
    )

    # Combined cap of 4 across both subjects, not 4 per subject.
    assert len(seen_pairs) == 4
    assert sum(len(v) for v in result.values()) == 4
