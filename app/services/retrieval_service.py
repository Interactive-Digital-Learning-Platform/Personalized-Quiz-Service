import logging
from asyncio import to_thread

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText, MatchValue

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services import retrieval_filters
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService

logger = logging.getLogger(__name__)


class RetrievalService:

    def __init__(self, embedder: EmbeddingGenerator, reranker: RerankService):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection = settings.QUIZ_KNOWLEDGE_COLLECTION
        self.embedder = embedder
        self.top_k = settings.RAG_TOP_K_CHUNKS
        self.threshold = settings.RAG_SCORE_THRESHOLD
        self.reranker = reranker

    async def search(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        top_k: int | None = None,
        lesson: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[SearchResult]:
        """Gather candidates -> drop non-chapter text and repeats -> rerank -> threshold -> spread across pages.

        Candidates: when `lesson` is given and chunks are tagged with it (PDF-Ingestion-Backend-Service qdrant
        migration 004), only that chapter's chunks -- so reranking orders real chapter text instead of competing
        with pages that merely mention the title. Otherwise the whole grade/subject, widened with `keywords`
        (the lesson's section headings): a full-text match on them plus a second dense query that includes
        them. The pool is a fixed RAG_RERANK_CANDIDATES per query rather than a multiple of top_k.
        """
        k = top_k or self.top_k
        base_filter = self._build_filter(grade=grade, subject=subject)
        limit = max(settings.RAG_RERANK_CANDIDATES, k)
        query_vector = await to_thread(self.embedder.embed_single, query)

        candidates: list[SearchResult] = []
        scope = "subject"
        if lesson:
            candidates = await self._query(query_vector, self._with_lesson(base_filter, lesson), limit)
            scope = "lesson" if candidates else "subject"
        if not candidates:
            candidates = await self._query(query_vector, base_filter, limit)
            if keywords:
                expanded_vector = await to_thread(self.embedder.embed_single, f"{query}: {', '.join(keywords)}")
                candidates += await self._query(expanded_vector, base_filter, limit)
                candidates += await self._query(query_vector, self._with_keywords(base_filter, keywords), limit)

        # Qdrant already excluded chunks tagged as contents/front/back matter; this catches untagged ones.
        chapter_text = [c for c in candidates if retrieval_filters.is_chapter_text(c)]
        unique = retrieval_filters.dedupe(chapter_text)

        reranked = await self.reranker.rerank(query, unique, top_k=len(unique))
        best = reranked[0].score if reranked else 0.0
        # Within the lesson's own chapter, relevance is already established by the tag -- section text rarely
        # repeats the lesson title, so an absolute bar would wrongly reject it. Reranking then only orders the
        # chapter; the relative floor still drops its weakest chunks.
        threshold = 0.0 if scope == "lesson" else self.threshold
        relevant = [
            r for r in reranked
            if r.score >= threshold and r.score >= best * settings.RAG_RELATIVE_SCORE_FLOOR
        ]
        results = retrieval_filters.diversify(relevant, k)

        logger.info(
            f"Retrieval: query='{query[:50]}' grade={grade} subject={subject} scope={scope} → "
            f"{len(candidates)} candidates, {len(unique)} after filtering, {len(relevant)} relevant "
            f"(threshold={self.threshold}), returning {len(results)}"
        )
        return results

    async def _query(self, vector: list[float], query_filter: Filter, limit: int) -> list[SearchResult]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(
                text=(hit.payload or {}).get("text", ""),
                score=round(hit.score or 0.0, 4),
                metadata={key: value for key, value in (hit.payload or {}).items() if key != "text"},
            )
            for hit in response.points
        ]

    @staticmethod
    def _with_lesson(base: Filter, lesson: str) -> Filter:
        return Filter(
            must=[*(base.must or []), FieldCondition(key="lesson", match=MatchValue(value=lesson))],
            must_not=base.must_not,
        )

    @staticmethod
    def _with_keywords(base: Filter, keywords: list[str]) -> Filter:
        # Full-text index on `text` (migration 004): a chunk qualifies if it contains any one heading's words.
        return Filter(
            must=base.must,
            must_not=base.must_not,
            should=[FieldCondition(key="text", match=MatchText(text=keyword)) for keyword in keywords],
        )

    @staticmethod
    def _build_filter(*, grade: int | None, subject: str | None) -> Filter:
        must = []

        if grade is not None:
            must.append(FieldCondition(key="grade", match=MatchValue(value=grade)))
        if subject is not None:
            must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))

        # `must_not` (rather than must content_type == "content") keeps chunks that were never tagged.
        must_not = [
            FieldCondition(key="content_type", match=MatchAny(any=list(retrieval_filters.NON_CONTENT_TYPES)))
        ]
        return Filter(must=must, must_not=must_not)

    async def close(self) -> None:
        await self.client.close()
