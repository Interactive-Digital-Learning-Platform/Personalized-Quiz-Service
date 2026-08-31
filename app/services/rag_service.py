import asyncio
import logging

from app.core.config import settings
from app.services.embedding_service import EmbeddingGenerator
from app.services.rerank_service import RerankService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Lazy module-level singletons -- mirrors the `_groq_client` singleton style
# already used in groq_service.py, rather than introducing FastAPI app.state
# DI that quiz_service.py's plain async functions don't otherwise use.
_embedder: EmbeddingGenerator | None = None
_reranker: RerankService | None = None
_retrieval_service: RetrievalService | None = None


def _is_available() -> bool:
    return settings.RAG_ENABLED and bool(settings.QDRANT_URL)


def _init_retrieval_service() -> RetrievalService:
    global _embedder, _reranker, _retrieval_service

    if _retrieval_service is None:
        _embedder = _embedder or EmbeddingGenerator()
        _reranker = _reranker or RerankService()
        _retrieval_service = RetrievalService(embedder=_embedder, reranker=_reranker)

    return _retrieval_service


async def _ensure_retrieval_service() -> RetrievalService:
    # SentenceTransformer/CrossEncoder construction is synchronous and can
    # take several seconds on a cold cache -- offloaded to a thread so a
    # lazy first-call init (warm_up() wasn't called, or failed at startup)
    # never blocks the event loop.
    if _retrieval_service is None:
        await asyncio.to_thread(_init_retrieval_service)
    return _retrieval_service  # type: ignore[return-value]


def warm_up() -> None:
    """Eagerly loads the embedding/rerank models so the first real request
    doesn't pay the multi-second model-load cost. Safe to call when RAG is
    disabled or Qdrant is unreachable -- callers (app/main.py) are expected
    to catch and log any failure here rather than block startup on it.
    """
    if not _is_available():
        logger.info("RAG disabled or QDRANT_URL unset — skipping model warm-up")
        return

    _init_retrieval_service()
    logger.info("RAG models warmed up")


def _build_query(grade: int, subject: str, lesson: str) -> str:
    return f"Grade {grade} {subject}: {lesson}"


def _truncate_snippet(text: str) -> str:
    if len(text) <= settings.RAG_MAX_SNIPPET_CHARS:
        return text
    return text[: settings.RAG_MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."


async def get_snippets_for_lesson(
    *,
    grade: int,
    subject: str,
    lesson: str,
    max_snippets: int,
) -> list[str]:
    """Best-effort retrieval of grounding excerpts for one (grade, subject,
    lesson). Returns [] on any failure, timeout, disabled RAG, or no result
    above the score threshold -- never raises. Quiz generation must succeed
    identically whether or not this returns anything.
    """
    if not _is_available():
        return []

    try:
        retrieval_service = await _ensure_retrieval_service()
        results = await asyncio.wait_for(
            retrieval_service.search(
                _build_query(grade, subject, lesson),
                grade=grade,
                subject=subject,
                top_k=max_snippets,
            ),
            timeout=settings.RAG_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception(
            "RAG retrieval failed for grade=%s subject=%s lesson=%s — continuing without grounding",
            grade, subject, lesson,
        )
        return []

    return [_truncate_snippet(r.text) for r in results[:max_snippets]]


async def get_snippets_by_lesson(
    *,
    grade: int,
    subject: str,
    lessons: list[str],
    max_snippets_per_lesson: int,
    max_lessons: int,
    max_concurrent: int | None = None,
) -> dict[str, list[str]]:
    """Runs get_snippets_for_lesson() over lessons[:max_lessons] concurrently,
    bounded by a semaphore so a many-lesson subject doesn't fire off an
    unbounded number of simultaneous embed/search/rerank calls on what is
    typically a single-worker service. A lesson that errors or returns
    nothing is simply absent from the result dict.
    """
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
    """Shuffle mode's variant of get_snippets_by_lesson() -- fans out across
    several subjects at once, sharing ONE combined cap on total (subject,
    lesson) retrieval queries across the whole batch (max_total_lesson_
    queries) rather than each subject getting its own allowance, so a
    many-subject shuffle quiz can't multiply its retrieval fan-out by
    subject count.
    """
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
