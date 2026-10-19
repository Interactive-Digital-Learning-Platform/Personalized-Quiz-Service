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


def test_qdrant_filter_uses_only_the_indexed_grade_and_subject():
    # lesson / content_type are handled in search() -- Qdrant Cloud rejects filtering on unindexed fields.
    query_filter = RetrievalService._build_filter(grade=10, subject="Science")

    assert {c.key for c in query_filter.must} == {"grade", "subject"}
    assert not query_filter.must_not
    assert RetrievalService._build_filter(grade=None, subject=None) is None


def _service_with_points(points: list) -> RetrievalService:
    service = RetrievalService.__new__(RetrievalService)
    service.collection = "pdf_knowledge_base_v2"
    service.threshold = 0.3
    service.top_k = 5
    service.embedder = SimpleNamespace(embed_single=lambda q: [0.0])
    service.sparse_embedder = None
    service._layout_resolved, service.dense_vector_name, service.hybrid = True, "dense", False
    service.client = SimpleNamespace(query_points=AsyncMock(return_value=SimpleNamespace(points=points)))
    return service


@pytest.mark.asyncio
async def test_search_applies_absolute_and_relative_score_floors(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RELATIVE_SCORE_FLOOR", 0.5)
    points = [
        SimpleNamespace(payload={"text": text, "content_type": "content", "filename": "b.pdf", "pages": [i]}, score=0.5)
        for i, text in enumerate(["best", "close", "weak", "noise", CONTENTS_PAGE])
    ]
    service = _service_with_points(points)
    scores = {"best": 0.9, "close": 0.6, "weak": 0.35, "noise": 0.1}

    async def fake_rerank(query, candidates, top_k):
        ranked = sorted(candidates, key=lambda c: scores.get(c.text, 0.0), reverse=True)
        return [SearchResult(c.text, scores.get(c.text, 0.0), c.metadata) for c in ranked[:top_k]]

    service.reranker = SimpleNamespace(rerank=fake_rerank)

    results = await service.search("Chemical basis of life", grade=10, subject="Science", top_k=5)

    # "weak" clears 0.3 but not 0.5 x best (0.45); "noise" clears neither.
    assert [r.text for r in results] == ["best", "close"]



# ── v2: lessons matched on section headings, vector layout, hybrid search ──────────────────────────────────────

def test_matches_lesson_on_the_lesson_or_a_subtopic_heading():
    in_proteins = _r("text", section="1.2 Proteins", section_path=["1 Chemical basis of life", "1.2 Proteins"])
    in_lipids = _r("text", section="Lipids and fats", section_path=["1.3 Lipids and fats"])
    elsewhere = _r("text", section="14.3 Reproduction of man", section_path=["14.3 Reproduction of man"])
    phrases = ["Chemical basis of life", "Proteins", "Lipids"]

    assert retrieval_filters.matches_lesson(in_proteins, phrases)
    assert retrieval_filters.matches_lesson(in_lipids, phrases)  # "Lipids" ~ "lipid" either way
    assert not retrieval_filters.matches_lesson(elsewhere, phrases)
    assert not retrieval_filters.matches_lesson(_r("text"), phrases)  # no headings at all


@pytest.mark.asyncio
async def test_lesson_search_keeps_only_the_lessons_own_sections(monkeypatch):
    monkeypatch.setattr(settings, "RAG_RELATIVE_SCORE_FLOOR", 0.0)
    payloads = [
        {"text": "proteins are polymers", "section": "1.2 Proteins", "filename": "b.pdf", "pages": [1]},
        {"text": "reproduction text", "section": "14.3 Reproduction of man", "filename": "b.pdf", "pages": [2]},
    ]
    service = _service_with_points([SimpleNamespace(payload=p, score=0.5) for p in payloads])

    async def rerank(query, candidates, top_k):
        # Below the absolute 0.3 bar -- a lesson-scoped match must still be kept.
        return [SearchResult(c.text, 0.1, c.metadata) for c in candidates]

    service.reranker = SimpleNamespace(rerank=rerank)

    results = await service.search(
        "Chemical basis of life", grade=10, subject="Science", lesson="Chemical basis of life", keywords=["Proteins"],
    )

    assert [r.text for r in results] == ["proteins are polymers"]


@pytest.mark.asyncio
async def test_layout_detects_v1_unnamed_and_v2_hybrid_collections():
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    def service_for(vectors, sparse):
        service = RetrievalService.__new__(RetrievalService)
        service.collection, service._layout_resolved = "kb", False
        service.sparse_embedder = SimpleNamespace(embed_query=lambda t: None)
        params = SimpleNamespace(vectors=vectors, sparse_vectors=sparse)
        service.client = SimpleNamespace(
            get_collection=AsyncMock(return_value=SimpleNamespace(config=SimpleNamespace(params=params)))
        )
        return service

    v1 = service_for(VectorParams(size=768, distance=Distance.COSINE), None)
    await v1._resolve_layout()
    assert (v1.dense_vector_name, v1.hybrid) == (None, False)

    v2 = service_for({"dense": VectorParams(size=768, distance=Distance.COSINE)}, {"bm25": SparseVectorParams()})
    await v2._resolve_layout()
    assert (v2.dense_vector_name, v2.hybrid) == ("dense", True)


@pytest.mark.asyncio
async def test_hybrid_query_fuses_dense_and_bm25_candidates():
    from qdrant_client.models import FusionQuery, SparseVector

    service = _service_with_points([])
    service.hybrid = True
    service.sparse_embedder = SimpleNamespace(embed_query=lambda t: SparseVector(indices=[1], values=[1.0]))

    await service._query("photosynthesis", RetrievalService._build_filter(grade=11, subject="Science"), 10)

    kwargs = service.client.query_points.await_args.kwargs
    assert isinstance(kwargs["query"], FusionQuery)
    assert [p.using for p in kwargs["prefetch"]] == ["dense", "bm25"]
    assert all(p.filter is not None for p in kwargs["prefetch"])


@pytest.mark.asyncio
async def test_lesson_sections_are_kept_up_to_the_minimum_even_when_reranked_low(monkeypatch):
    monkeypatch.setattr(settings, "RAG_LESSON_MIN_EXCERPTS", 3)
    payloads = [
        {"text": f"lesson text {i}", "section": f"1.{i} Proteins", "filename": "b.pdf", "pages": [i]} for i in range(4)
    ]
    service = _service_with_points([SimpleNamespace(payload=p, score=0.5) for p in payloads])
    scores = {"lesson text 0": 0.02, "lesson text 1": 0.004, "lesson text 2": 0.003, "lesson text 3": 0.001}

    async def rerank(query, candidates, top_k):
        ranked = sorted(candidates, key=lambda c: scores[c.text], reverse=True)
        return [SearchResult(c.text, scores[c.text], c.metadata) for c in ranked]

    service.reranker = SimpleNamespace(rerank=rerank)

    results = await service.search("Chemical basis of life", grade=10, subject="Science", top_k=4,
                                   lesson="Chemical basis of life", keywords=["Proteins"])

    # The relative floor alone keeps only the best; the lesson minimum tops it up, best-first.
    assert [r.text for r in results] == ["lesson text 0", "lesson text 1", "lesson text 2"]


@pytest.mark.asyncio
async def test_each_subtopic_gets_its_own_search_for_a_lesson(monkeypatch):
    monkeypatch.setattr(settings, "RAG_LESSON_SUBTOPIC_QUERIES", 2)
    service = _service_with_points([])
    embedded: list[str] = []
    service.embedder = SimpleNamespace(embed_single=lambda q: embedded.append(q) or [0.0])

    async def rerank(query, candidates, top_k):
        return []

    service.reranker = SimpleNamespace(rerank=rerank)

    await service.search("Logic Gates", grade=10, subject="ICT", lesson="Logic Gates",
                         keywords=["Truth tables", "AND gate", "OR gate"])

    assert sorted(embedded) == sorted(["Logic Gates", "Logic Gates: Truth tables, AND gate, OR gate", "Truth tables", "AND gate"])


def test_long_lesson_titles_match_their_shorter_textbook_headings():
    basic_gates = _r("t", section="4.3.2 OR Gate", section_path=["4.3 Basic Logic Gates", "4.3.2 OR Gate"])
    other = _r("t", section="2.4 Basic Physical Components", section_path=["2.4 Basic Physical Components"])
    ict = ["Logic Gates with Boolean Functions", "Fundamentals of a computer system", "Electronic Spreadsheet"]
    shared = retrieval_filters.shared_title_words(ict)

    assert retrieval_filters.matches_lesson(basic_gates, [], title=ict[0], shared_words=shared)
    assert not retrieval_filters.matches_lesson(other, [], title=ict[0], shared_words=shared)


def test_words_many_lesson_titles_share_dont_make_a_match():
    # Every one of these is "... of/in Sri Lanka": that alone mustn't tie a heading to a lesson.
    geography = ["Natural resources of Sri Lanka", "The population of Sri Lanka", "Natural hazards in Sri Lanka"]
    shared = retrieval_filters.shared_title_words(geography)
    generic = _r("t", section="Sri Lanka Matha", section_path=["Unit 2 Sri Lanka", "Sri Lanka Matha"])
    population = _r("t", section="Population Structure", section_path=["The population of Sri Lanka"])

    assert {"sri", "lanka"} <= shared
    assert not retrieval_filters.matches_lesson(generic, [], title=geography[0], shared_words=shared)
    # A heading carrying the whole title still matches (the strict rule).
    assert retrieval_filters.matches_lesson(population, [], title=geography[1], shared_words=shared)


def test_one_distinctive_word_is_not_enough_for_a_loose_match():
    # "Fundamentals of a computer system": computer/system are shared, leaving only "fundamentals".
    ict = ["Fundamentals of a computer system", "Computer networks", "System development life cycle"]
    shared = retrieval_filters.shared_title_words(ict)
    presentation = _r("t", section="8.4 Fundamentals of presentation software", section_path=[])

    assert not retrieval_filters.matches_lesson(presentation, [], title=ict[0], shared_words=shared)
