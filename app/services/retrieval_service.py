import logging
from asyncio import gather, to_thread

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
    VectorParams,
)

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services import retrieval_filters
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService
from app.services.sparse_embedding_service import SparseEmbeddingGenerator

logger = logging.getLogger(__name__)


class RetrievalService:

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        reranker: RerankService,
        sparse_embedder: SparseEmbeddingGenerator | None = None,
    ):
        # An empty key is sent as "" rather than omitted, so only pass one when it's set.
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        self.collection = settings.QUIZ_KNOWLEDGE_COLLECTION
        self.embedder = embedder
        self.top_k = settings.RAG_TOP_K_CHUNKS
        self.threshold = settings.RAG_SCORE_THRESHOLD
        self.reranker = reranker
        self.sparse_embedder = sparse_embedder
        # Resolved from the collection itself on first search -- see _resolve_layout.
        self._layout_resolved = False
        self.dense_vector_name: str | None = None
        self.hybrid = False

    async def _resolve_layout(self) -> None:
        """Which vectors to query: a single unnamed vector (the v1 collection) or the named dense vector, plus the
        BM25 sparse vector when the collection has one (v2) -- in which case search is hybrid."""
        if self._layout_resolved:
            return
        params = (await self.client.get_collection(self.collection)).config.params
        vectors = params.vectors
        if isinstance(vectors, VectorParams):
            self.dense_vector_name = None
        elif isinstance(vectors, dict) and settings.QUIZ_KNOWLEDGE_DENSE_VECTOR_NAME in vectors:
            self.dense_vector_name = settings.QUIZ_KNOWLEDGE_DENSE_VECTOR_NAME
        else:
            raise ValueError(
                f"Qdrant collection '{self.collection}' has neither a single unnamed vector nor a "
                f"'{settings.QUIZ_KNOWLEDGE_DENSE_VECTOR_NAME}' vector"
            )
        has_sparse = settings.QUIZ_KNOWLEDGE_SPARSE_VECTOR_NAME in (params.sparse_vectors or {})
        self.hybrid = has_sparse and self.sparse_embedder is not None
        self._layout_resolved = True
        logger.info(
            f"Quiz knowledge base '{self.collection}': {'hybrid (dense + bm25)' if self.hybrid else 'dense'} search"
        )

    async def search(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        top_k: int | None = None,
        lesson: str | None = None,
        keywords: list[str] | None = None,
        subject_lessons: list[str] | None = None,
    ) -> list[SearchResult]:
        """Gather candidates -> keep the lesson's own sections if any -> drop non-chapter text and repeats ->
        rerank -> threshold -> spread across pages.

        Candidates come from the whole grade/subject (the only fields filtered on in Qdrant -- both indexed),
        once for the query and, given `keywords` (the lesson's subtopics), once more with them appended -- and,
        for a `lesson`, once per subtopic on its own (up to RAG_LESSON_SUBTOPIC_QUERIES), so the lesson's
        sections reach the pool even when its title alone doesn't find them. With a `lesson`, the candidates
        sitting under a heading that names the lesson or one of its subtopics (retrieval_filters.matches_lesson --
        `subject_lessons`, all the subject's lesson titles, says which title words are too common to count)
        are kept and the rest dropped; if none do, the whole subject's candidates are used. From the lesson's
        own sections at least RAG_LESSON_MIN_EXCERPTS are returned whatever their rerank scores. The pool is a
        fixed RAG_RERANK_CANDIDATES per query rather than a multiple of top_k.
        """
        await self._resolve_layout()
        k = top_k or self.top_k
        query_filter = self._build_filter(grade=grade, subject=subject)
        limit = max(settings.RAG_RERANK_CANDIDATES, k)

        queries = [query]
        if keywords:
            queries.append(f"{query}: {', '.join(keywords)}")
            if lesson:
                queries.extend(keywords[: settings.RAG_LESSON_SUBTOPIC_QUERIES])
        candidates = [
            result
            for batch in await gather(*(self._query(text, query_filter, limit) for text in queries))
            for result in batch
        ]

        scope = "subject"
        if lesson:
            shared = retrieval_filters.shared_title_words(subject_lessons or [])
            in_lesson = [
                c for c in candidates
                if retrieval_filters.matches_lesson(c, keywords or [], title=lesson, shared_words=shared)
            ]
            if in_lesson:
                candidates, scope = in_lesson, "lesson"

        # contents / front / back matter -- the content_type tag when a chunk has one, else text heuristics.
        chapter_text = [c for c in candidates if retrieval_filters.is_chapter_text(c)]
        unique = retrieval_filters.dedupe(chapter_text)

        reranked = await self.reranker.rerank(query, unique, top_k=len(unique))
        best = reranked[0].score if reranked else 0.0
        # Within the lesson's own sections, relevance is already established by the heading -- section text
        # rarely repeats the lesson title, so an absolute bar would wrongly reject it. Reranking then only orders
        # those sections; the relative floor still drops their weakest chunks.
        threshold = 0.0 if scope == "lesson" else self.threshold
        relevant = [
            r for r in reranked
            if r.score >= threshold and r.score >= best * settings.RAG_RELATIVE_SCORE_FLOOR
        ]
        if scope == "lesson":
            # The lesson's own sections are on-topic by their heading -- top up to the minimum in rerank order
            # even when the reranker scores them low (it compares chunks against a sometimes very broad title).
            floor_count = min(settings.RAG_LESSON_MIN_EXCERPTS, k)
            if len(relevant) < floor_count:
                relevant += [r for r in reranked if r not in relevant][: floor_count - len(relevant)]
        results = retrieval_filters.diversify(relevant, k)

        logger.info(
            f"Retrieval: query='{query[:50]}' grade={grade} subject={subject} scope={scope} "
            f"mode={'hybrid' if self.hybrid else 'dense'} → {len(candidates)} candidates, {len(unique)} after "
            f"filtering, {len(relevant)} relevant (threshold={self.threshold}), returning {len(results)}"
        )
        return results

    async def _vectors(self, text: str) -> tuple[list[float], SparseVector | None]:
        if self.hybrid and self.sparse_embedder is not None:
            dense, sparse = await gather(
                to_thread(self.embedder.embed_single, text), to_thread(self.sparse_embedder.embed_query, text),
            )
            return dense, sparse
        return await to_thread(self.embedder.embed_single, text), None

    async def _query(self, text: str, query_filter: Filter, limit: int) -> list[SearchResult]:
        dense, sparse = await self._vectors(text)
        if sparse is not None:
            prefetch_limit = max(settings.RAG_HYBRID_PREFETCH_LIMIT, limit)
            response = await self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(query=dense, using=self.dense_vector_name, filter=query_filter, limit=prefetch_limit),
                    Prefetch(
                        query=sparse, using=settings.QUIZ_KNOWLEDGE_SPARSE_VECTOR_NAME,
                        filter=query_filter, limit=prefetch_limit,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = await self.client.query_points(
                collection_name=self.collection,
                query=dense,
                using=self.dense_vector_name,
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
    def _build_filter(*, grade: int | None, subject: str | None) -> Filter | None:
        # Only indexed fields: Qdrant Cloud's strict mode rejects filtering on an unindexed one, so lesson and
        # content_type are handled in search() instead.
        must = []
        if grade is not None:
            must.append(FieldCondition(key="grade", match=MatchValue(value=grade)))
        if subject is not None:
            must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
        return Filter(must=must) if must else None

    async def close(self) -> None:
        await self.client.close()
