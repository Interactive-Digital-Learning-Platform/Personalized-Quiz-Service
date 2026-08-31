import logging
from asyncio import to_thread
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.schemas.retrieval import SearchResult
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
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:

        query_vector = await to_thread(self.embedder.embed_single, query)
        k = top_k or self.top_k

        query_filter = self._build_filter(grade=grade, subject=subject)

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=k * settings.RERANK_OVERFETCH,
            with_payload=True,
        )

        candidates = []
        for hit in response.points:

            payload = hit.payload or {}
            score = hit.score

            candidates.append(
                SearchResult(
                    text=payload.get("text", ""),
                    score=round(score or 0.0, 4),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        reranked = await self.reranker.rerank(query, candidates, top_k=k)

        logger.info(
            f"Retrieval: query='{query[:50]}' grade={grade} subject={subject} "
            f"→ {len(candidates)} candidates (threshold={self.threshold})"
        )

        return [r for r in reranked if r.score >= self.threshold]

    @staticmethod
    def _build_filter(*, grade: Optional[int], subject: Optional[str]) -> Optional[Filter]:
        must = []

        if grade is not None:
            must.append(FieldCondition(key="grade", match=MatchValue(value=grade)))
        if subject is not None:
            must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))

        return Filter(must=must) if must else None

    async def close(self) -> None:
        await self.client.close()
