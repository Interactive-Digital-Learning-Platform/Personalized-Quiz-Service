import logging
import math
from asyncio import to_thread

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.schemas.retrieval import SearchResult

logger = logging.getLogger(__name__)


class RerankService:

    def __init__(self, model_name: str = settings.RERANK_MODEL, device: str = settings.EMBEDDING_DEVICE):
        # Without an explicit device, sentence-transformers auto-detects MPS on Apple Silicon, whose first-use Metal shader compilation can hang indefinitely in sandboxed/headless dev environments -- pinned to CPU like embedding_service.EmbeddingGenerator to avoid that.
        self.model = CrossEncoder(model_name, device=device)

    async def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:

        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        logits = await to_thread(self.model.predict, pairs)

        # ms-marco cross-encoders output raw logits (roughly -11..11, model-specific scale); a sigmoid maps them to
        # 0-1 relevance so RAG_SCORE_THRESHOLD means the same thing whatever the query looks like.
        scores = [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(text=c.text, score=round(s, 4), metadata=c.metadata)
            for c, s in ranked[:top_k]
        ]
