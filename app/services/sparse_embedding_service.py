import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from app.core.config import settings

logger = logging.getLogger(__name__)


class SparseEmbeddingGenerator:
    """BM25 query vectors for the knowledge base's sparse (keyword) vector -- the same model
    PDF-Ingestion-Backend-Service encodes chunks with, mirroring AI-Learning-Assistant-Service's own."""

    def __init__(self, model_name: str = settings.RAG_SPARSE_MODEL):
        self.model_name = model_name
        logger.info(f"Loading sparse model '{model_name}'...")
        self.model = SparseTextEmbedding(model_name=model_name)
        logger.info("Sparse model ready")

    def embed_query(self, text: str) -> SparseVector:
        embedding = next(iter(self.model.query_embed(text)))
        return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
