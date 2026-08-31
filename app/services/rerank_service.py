import logging
from asyncio import to_thread

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.schemas.retrieval import SearchResult

logger = logging.getLogger(__name__)


class RerankService:

    def __init__(self, model_name: str = settings.RERANK_MODEL, device: str = settings.EMBEDDING_DEVICE):
        # Without an explicit device, sentence-transformers auto-detects and
        # picks MPS (Apple GPU) on Apple Silicon -- unlike embedding_service.
        # EmbeddingGenerator, which already pins this via the same setting.
        # First-use Metal shader compilation for this model has been
        # observed to hang indefinitely in sandboxed/headless local dev
        # environments (no interactive GPU/WindowServer access), so this
        # must stay pinned to CPU the same way, not just left to auto-detect.
        self.model = CrossEncoder(model_name, device=device)

    async def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:

        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = await to_thread(self.model.predict, pairs)

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(text=c.text, score=round(float(s), 4), metadata=c.metadata)
            for c, s in ranked[:top_k]
        ]
