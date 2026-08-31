import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        device: str = settings.EMBEDDING_DEVICE,
    ):
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading embedding model '{model_name}' on {device}...")

        self.model = SentenceTransformer(model_name, device=device)

        actual_dim = self.embedding_dimension()

        if actual_dim != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Model '{model_name}' outputs {actual_dim}-dim vectors "
                f"but EMBEDDING_DIM={settings.EMBEDDING_DIM} in config. "
                f"Query vectors won't match indexed vectors."
            )

        logger.info(f"Embedding model ready — dim={self.embedding_dimension()}")

    def embed_single(self, text: str) -> List[float]:
        prefixed = f"search_query: {text}"
        prefixed = self._truncate(prefixed, label="query")

        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embedding_dimension(self) -> int:
        dim = self.model.get_embedding_dimension()

        if dim is None:
            dim = len(self.model.encode("probe", convert_to_numpy=True))

        return dim

    def _truncate(self, text: str, label: str = "text") -> str:
        estimated = int(len(text.split()) * 1.3)

        if estimated > settings.RAG_MAX_QUERY_TOKENS:
            max_words = int(settings.RAG_MAX_QUERY_TOKENS / 1.3)
            text = " ".join(text.split()[:max_words])

            logger.warning(f"{label} truncated to {max_words} words")

        return text
