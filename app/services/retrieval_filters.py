"""Pure helpers that keep non-chapter text and repeats out of RAG grounding excerpts.

pdf_knowledge_base chunks are tagged with `content_type` by PDF-Ingestion-Backend-Service's qdrant
migration 004, and retrieval_service filters on that tag in Qdrant. These heuristics are the fallback
for chunks without the tag (anything ingested after that migration ran): a book's contents page and its
credits/preface pages mention every lesson title, so without a filter they outrank the chapters
themselves.
"""
import re
from collections.abc import Iterable

from app.schemas.retrieval import SearchResult

NON_CONTENT_TYPES = ("contents", "front_matter", "back_matter")

# Contents entries, one per line: "3. Structure of matter 52", "2.4 Acceleration\n\n32", "6 Word Processing 154".
# The title must be words starting with a capital -- numbered maths/science text ("12 m from A on a bearing of
# 080", "1 protons 1") must not count. Same pattern as PDF-Ingestion-Backend-Service qdrant migration 004.
_CONTENTS_ENTRY = re.compile(
    r"^[ \t]*\d{1,2}(?:\.\d{1,2})?\.?[ \t]+[A-Z][A-Za-z'’,:()\-–]*(?:[ \t]+[A-Za-z'’,:()\-–]+){0,12}[ \t]*(?:\n[ \t]*){0,2}\d{1,3}[ \t]*$",
    re.MULTILINE,
)
_CONTENTS_HEADING = re.compile(r"(?:^|\n)\s*(?:table of )?(?:contents?|index)\s*(?:\n|$)", re.IGNORECASE)
_FRONT_MATTER_MARKERS = re.compile(
    r"technical assistance|language editing|proof reading|board of (?:editors|writers|compilers)|"
    r"editorial board|message of the|foreword|preface|isbn|first print|all rights reserved|"
    r"cover (?:page|design)|computer (?:designing|typesetting)",
    re.IGNORECASE,
)
_CONTENTS_ENTRY_THRESHOLD = 4


def looks_like_contents(text: str) -> bool:
    return bool(_CONTENTS_HEADING.search(text)) or len(_CONTENTS_ENTRY.findall(text)) >= _CONTENTS_ENTRY_THRESHOLD


def looks_like_front_matter(text: str) -> bool:
    return bool(_FRONT_MATTER_MARKERS.search(text))


def is_chapter_text(result: SearchResult) -> bool:
    """True for real chapter content. Trusts migration 004's tag when present, else falls back to text heuristics."""
    content_type = (result.metadata or {}).get("content_type")
    if content_type is not None:
        # Safety net for a tagging miss: a run of contents entries is never teaching text. (Only the entry count
        # here -- a chapter can legitimately have an "Index" heading, e.g. ICT databases.)
        if len(_CONTENTS_ENTRY.findall(result.text)) >= _CONTENTS_ENTRY_THRESHOLD:
            return False
        return content_type not in NON_CONTENT_TYPES
    return not (looks_like_contents(result.text) or looks_like_front_matter(result.text))


def dedup_key(text: str) -> str:
    """Same passage indexed twice (or from two copies of a book) maps to the same key."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())[:400]


def dedupe(results: Iterable[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        key = dedup_key(result.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def pages_of(result: SearchResult) -> set[int]:
    metadata = result.metadata or {}
    pages = metadata.get("pages") or ([metadata["page_start"]] if "page_start" in metadata else [])
    return {int(p) for p in pages}


def diversify(ranked: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Best-first, but skip a result whose pages overlap one already picked — so two excerpts aren't the same
    page twice — unless there aren't enough other results to fill top_k."""
    picked: list[SearchResult] = []
    deferred: list[SearchResult] = []
    for result in ranked:
        if len(picked) >= top_k:
            break
        same_book = [p for p in picked if (p.metadata or {}).get("filename") == (result.metadata or {}).get("filename")]
        if any(pages_of(result) & pages_of(p) for p in same_book):
            deferred.append(result)
        else:
            picked.append(result)
    return (picked + deferred)[:top_k]
