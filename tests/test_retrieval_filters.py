"""
tests/test_retrieval_filters.py
───────────────────────────────
What reaches the AI as reference material: no contents/credits pages, no repeated passages, excerpts
spread across pages, and only chunks the reranker actually finds relevant.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services import retrieval_filters
from app.services.retrieval_service import RetrievalService

CONTENTS_PAGE = (
    "Contents\nPage\n1. Chemical basis of life 01\n1.1 Cabohydrates 02\n1.2 Proteins 07\n1.3 Lipids 10\n"
    "1.4 Nucleic acids 11\n1.5 Water 14\n2. Motion in a straight line\n\n23"
)
NUMBERED_WORKED_EXAMPLE = "1. A ship sails\n12 m from A on a bearing of 080\n50 m from P on a bearing of 120\n25"
CREDITS_PAGE = "Technical Assistance 1. Asanka Aravinda Mahakumarage - Educational Publications Department"
CHAPTER_TEXT = "Carbohydrates are organic compounds made of carbon, hydrogen and oxygen."


def _r(text: str, score: float = 0.9, **meta) -> SearchResult:
    return SearchResult(text=text, score=score, metadata=meta)


def test_tagged_chunks_trust_the_migration_tag():
    assert retrieval_filters.is_chapter_text(_r(CHAPTER_TEXT, content_type="content"))
    assert not retrieval_filters.is_chapter_text(_r(CHAPTER_TEXT, content_type="contents"))
    assert not retrieval_filters.is_chapter_text(_r(CHAPTER_TEXT, content_type="front_matter"))


def test_untagged_contents_and_credits_pages_are_filtered_by_heuristics():
    assert not retrieval_filters.is_chapter_text(_r(CONTENTS_PAGE))
    assert not retrieval_filters.is_chapter_text(_r(CREDITS_PAGE))
    assert retrieval_filters.is_chapter_text(_r(CHAPTER_TEXT))


def test_a_contents_list_mis_tagged_as_content_is_still_rejected():
    assert not retrieval_filters.is_chapter_text(_r(CONTENTS_PAGE, content_type="content"))


def test_numbered_maths_text_is_not_mistaken_for_contents():
    assert retrieval_filters.is_chapter_text(_r(NUMBERED_WORKED_EXAMPLE))
    assert retrieval_filters.is_chapter_text(_r(NUMBERED_WORKED_EXAMPLE, content_type="content"))


def test_dedupe_drops_the_same_passage_from_two_copies():
    unique = retrieval_filters.dedupe([
        _r(CHAPTER_TEXT, filename="part 1.pdf"),
        _r("  " + CHAPTER_TEXT.upper() + " ", filename="textbook.pdf"),
        _r("Proteins are made of amino acids."),
    ])
    assert [u.text for u in unique] == [CHAPTER_TEXT, "Proteins are made of amino acids."]


def test_diversify_prefers_excerpts_from_different_pages():
    ranked = [
        _r("a", filename="b.pdf", pages=[10, 11]),
        _r("b", filename="b.pdf", pages=[11]),
        _r("c", filename="b.pdf", pages=[20]),
    ]
    assert [r.text for r in retrieval_filters.diversify(ranked, 2)] == ["a", "c"]
    # Same pages are still used to fill up when there's nothing else.
    assert [r.text for r in retrieval_filters.diversify(ranked[:2], 2)] == ["a", "b"]


def test_filter_excludes_non_chapter_types_but_keeps_untagged_chunks():
    query_filter = RetrievalService._build_filter(grade=10, subject="Science")

    assert {c.key for c in query_filter.must} == {"grade", "subject"}
    (must_not,) = query_filter.must_not
    assert must_not.key == "content_type"
    assert set(must_not.match.any) == {"contents", "front_matter", "back_matter"}


@pytest.mark.asyncio
async def test_search_applies_absolute_and_relative_score_floors(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RELATIVE_SCORE_FLOOR", 0.5)
    service = RetrievalService.__new__(RetrievalService)
    service.collection = "pdf_knowledge_base"
    service.threshold = 0.3
    service.top_k = 5
    service.embedder = SimpleNamespace(embed_single=lambda q: [0.0])
    points = [
        SimpleNamespace(payload={"text": text, "content_type": "content", "filename": "b.pdf", "pages": [i]}, score=0.5)
        for i, text in enumerate(["best", "close", "weak", "noise", CONTENTS_PAGE])
    ]
    service.client = SimpleNamespace(query_points=AsyncMock(return_value=SimpleNamespace(points=points)))
    scores = {"best": 0.9, "close": 0.6, "weak": 0.35, "noise": 0.1}

    async def fake_rerank(query, candidates, top_k):
        ranked = sorted(candidates, key=lambda c: scores.get(c.text, 0.0), reverse=True)
        return [SearchResult(c.text, scores.get(c.text, 0.0), c.metadata) for c in ranked[:top_k]]

    service.reranker = SimpleNamespace(rerank=fake_rerank)

    results = await service.search("Chemical basis of life", grade=10, subject="Science", top_k=5)

    # "weak" clears 0.3 but not 0.5 x best (0.45); "noise" clears neither.
    assert [r.text for r in results] == ["best", "close"]
