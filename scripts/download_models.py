"""
Downloads the RAG models into the image at `docker build` time (see Dockerfile), so a container starts with them
already on disk instead of fetching ~650 MB from Hugging Face on every new container:

- EMBEDDING_MODEL  (dense vectors)           -> $HF_HOME
- RERANK_MODEL     (cross-encoder reranker)   -> $HF_HOME
- RAG_SPARSE_MODEL (BM25 keyword vectors)     -> $FASTEMBED_CACHE_PATH

Loads them through the service's own classes, so the image holds exactly what rag_service will load at runtime.
Run with the same model settings the container will use -- a model name changed only at runtime isn't baked in.

    python scripts/download_models.py
"""
import os
import sys

# app.core.config refuses to load without these; their values are irrelevant here (nothing connects anywhere).
for name, placeholder in {
    "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
    "GROQ_API_KEY": "build-time-placeholder",
    "CLERK_JWKS_URL": "https://example.invalid/.well-known/jwks.json",
    "CLERK_ISSUER": "https://example.invalid",
}.items():
    os.environ.setdefault(name, placeholder)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService
from app.services.sparse_embedding_service import SparseEmbeddingGenerator


def main() -> None:
    print(f"Embedding model: {settings.EMBEDDING_MODEL}", flush=True)
    EmbeddingGenerator().embed_single("warm-up")
    print(f"Rerank model: {settings.RERANK_MODEL}", flush=True)
    RerankService()
    if settings.RAG_HYBRID_SEARCH_ENABLED:
        print(f"Sparse model: {settings.RAG_SPARSE_MODEL}", flush=True)
        SparseEmbeddingGenerator().embed_query("warm-up")
    print("Models downloaded.", flush=True)


if __name__ == "__main__":
    main()
