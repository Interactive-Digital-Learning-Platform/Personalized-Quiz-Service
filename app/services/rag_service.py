import asyncio
import logging
import time

from app.core.config import settings
from app.schemas.retrieval import SearchResult
from app.services import curriculum_service, source_labels
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService
from app.services.sparse_embedding_service import SparseEmbeddingGenerator

logger = logging.getLogger(__name__)

# Lazy module-level singletons -- mirrors the `_groq_client` singleton style already used in groq_service.py rather than introducing FastAPI app.state DI.
_embedder: EmbeddingGenerator | None = None
_reranker: RerankService | None = None
_sparse_embedder: SparseEmbeddingGenerator | None = None
_retrieval_service: RetrievalService | None = None
# (grade, subject, lesson, max_snippets) -> (expires_at monotonic, results). Per process, like the singletons.
_excerpt_cache: dict[tuple[int, str, str, int], tuple[float, list[SearchResult]]] = {}


def _is_available() -> bool:
    return settings.RAG_ENABLED and bool(settings.QDRANT_URL)


def _init_retrieval_service() -> RetrievalService:
    global _embedder, _reranker, _sparse_embedder, _retrieval_service

    if _retrieval_service is None:
        _embedder = _embedder or EmbeddingGenerator()
        _reranker = _reranker or RerankService()
        if settings.RAG_HYBRID_SEARCH_ENABLED:
            _sparse_embedder = _sparse_embedder or SparseEmbeddingGenerator()
        _retrieval_service = RetrievalService(
            embedder=_embedder, reranker=_reranker, sparse_embedder=_sparse_embedder,
        )

    return _retrieval_service


async def _ensure_retrieval_service() -> RetrievalService:
    # SentenceTransformer/CrossEncoder construction is synchronous and can take several seconds on a cold cache -- offloaded to a thread so a lazy first-call init never blocks the event loop.
    if _retrieval_service is None:
        await asyncio.to_thread(_init_retrieval_service)
    return _retrieval_service  # type: ignore[return-value]


def warm_up() -> None:
    # Eagerly loads the embedding/rerank models so the first real request doesn't pay the multi-second model-load cost; safe to call when RAG is disabled or Qdrant is unreachable.
    if not _is_available():
        logger.info("RAG disabled or QDRANT_URL unset — skipping model warm-up")
        return

    _init_retrieval_service()
    logger.info("RAG models warmed up")


def _build_query(grade: int, subject: str, lesson: str) -> str:
    # Just the lesson: grade and subject are already Qdrant filters, and prefixing them ("Grade 10 Science:
    # Motion in a straight line") made the ms-marco reranker score real chapter text so low that whole
    # lessons came back with no excerpts at all.
    return lesson


def clear_cache() -> None:
    _excerpt_cache.clear()


def _truncate_snippet(text: str) -> str:
    if len(text) <= settings.RAG_MAX_SNIPPET_CHARS:
        return text
    return text[: settings.RAG_MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."


async def retrieve_excerpts(
    *,
    grade: int,
    subject: str,
    lesson: str,
    max_snippets: int,
) -> list[SearchResult]:
    # Grounding excerpts for one (grade, subject, lesson) with their page/book metadata; [] on any failure,
    # timeout, disabled RAG, or no result above threshold -- never raises. Successful lookups are cached for
    # RAG_CACHE_TTL_SECONDS (textbooks don't change); failures aren't, so they're retried next time.
    if not _is_available():
        return []

    key = (grade, subject, lesson, max_snippets)
    cached = _excerpt_cache.get(key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]

    try:
        retrieval_service = await _ensure_retrieval_service()
        results = await asyncio.wait_for(
            retrieval_service.search(
                _build_query(grade, subject, lesson),
                grade=grade,
                subject=subject,
                top_k=max_snippets,
                # Matched against the chunks' section headings (the curriculum's exact lesson name plus its
                # subtopics); a free-text lesson that isn't in the curriculum simply searches the whole subject.
                lesson=curriculum_service.canonical_lesson(grade, subject, lesson),
                keywords=curriculum_service.subtopics_for(grade, subject, lesson) or None,
                subject_lessons=curriculum_service.lessons_for(grade, subject) or None,
            ),
            timeout=settings.RAG_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception(
            "RAG retrieval failed for grade=%s subject=%s lesson=%s — continuing without grounding",
            grade, subject, lesson,
        )
        return []

    results = results[:max_snippets]
    _excerpt_cache[key] = (time.monotonic() + settings.RAG_CACHE_TTL_SECONDS, results)
    return results


async def get_snippets_for_lesson(
    *,
    grade: int,
    subject: str,
    lesson: str,
    max_snippets: int,
) -> list[str]:
    # Excerpt texts for the prompt, truncated and prefixed with their "[Source: <book>, page <n>]" label so the
    # AI can say which excerpt each question came from -- see retrieve_excerpts() and source_labels.
    results = await retrieve_excerpts(grade=grade, subject=subject, lesson=lesson, max_snippets=max_snippets)
    snippets = []
    for r in results:
        label = source_labels.format_label(r.metadata or {})
        text = _truncate_snippet(r.text)
        snippets.append(f"{label} {text}" if label else text)
    return snippets


async def get_snippets_by_lesson(
    *,
    grade: int,
    subject: str,
    lessons: list[str],
    max_snippets_per_lesson: int,
    max_lessons: int,
    max_concurrent: int | None = None,
) -> dict[str, list[str]]:
    # Runs get_snippets_for_lesson() over lessons[:max_lessons] concurrently, bounded by a semaphore so a many-lesson subject doesn't fire off unbounded simultaneous calls; a lesson that errors or returns nothing is simply absent from the result.
    if not _is_available() or not lessons:
        return {}

    capped_lessons = lessons[:max_lessons]
    semaphore = asyncio.Semaphore(max_concurrent or settings.RAG_MAX_CONCURRENT_QUERIES)

    async def _fetch(lesson: str) -> tuple[str, list[str]]:
        async with semaphore:
            snippets = await get_snippets_for_lesson(
                grade=grade, subject=subject, lesson=lesson, max_snippets=max_snippets_per_lesson,
            )
            return lesson, snippets

    results = await asyncio.gather(*(_fetch(lesson) for lesson in capped_lessons))

    return {lesson: snippets for lesson, snippets in results if snippets}


async def get_snippets_by_lesson_for_subjects(
    *,
    grade: int,
    subjects_lessons: dict[str, list[str]],
    max_snippets_per_lesson: int,
    max_total_lesson_queries: int,
    max_concurrent: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    # Shuffle mode's variant of get_snippets_by_lesson() -- fans out across several subjects at once, sharing ONE combined cap on total (subject, lesson) queries so a many-subject shuffle can't multiply its retrieval fan-out.
    if not _is_available() or not subjects_lessons:
        return {}

    pairs = [
        (subject, lesson)
        for subject, lessons in subjects_lessons.items()
        for lesson in lessons
    ][:max_total_lesson_queries]

    if not pairs:
        return {}

    semaphore = asyncio.Semaphore(max_concurrent or settings.RAG_MAX_CONCURRENT_QUERIES)

    async def _fetch(subject: str, lesson: str) -> tuple[str, str, list[str]]:
        async with semaphore:
            snippets = await get_snippets_for_lesson(
                grade=grade, subject=subject, lesson=lesson, max_snippets=max_snippets_per_lesson,
            )
            return subject, lesson, snippets

    results = await asyncio.gather(*(_fetch(subject, lesson) for subject, lesson in pairs))

    grouped: dict[str, dict[str, list[str]]] = {}
    for subject, lesson, snippets in results:
        if snippets:
            grouped.setdefault(subject, {})[lesson] = snippets

    return grouped
