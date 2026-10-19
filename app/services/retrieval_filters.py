"""Pure helpers that keep non-chapter text and repeats out of RAG grounding excerpts.

pdf_knowledge_base chunks are tagged with `content_type` by PDF-Ingestion-Backend-Service's qdrant
migration 004, and retrieval_service filters on that tag in Qdrant. These heuristics are the fallback
for chunks without the tag (anything ingested after that migration ran): a book's contents page and its
credits/preface pages mention every lesson title, so without a filter they outrank the chapters
themselves.
"""
import math
import re
from collections import Counter
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


# Words too generic to say which lesson a heading belongs to.
_HEADING_STOP_WORDS = frozenset(
    [
        "the", "and", "for", "with", "from", "into", "its", "their", "this", "that", "are", "was", "were",
        "how", "what", "why", "when", "which", "who", "about", "between", "of", "in", "on", "to", "by",
        "an", "a", "or", "as", "at", "is", "be",
    ]
)


def _heading_words(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) < 3 or word in _HEADING_STOP_WORDS:
            continue
        # "Proteins" ~ "protein", "Forces" ~ "force" -- enough to line headings up with curriculum names.
        words.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return words


def heading_text(result: SearchResult) -> str:
    """A chunk's place in its book -- its section heading and the headings above it (PDF-Ingestion-Backend-Service
    stores these as `section` / `section_path`)."""
    metadata = result.metadata or {}
    path = metadata.get("section_path") or []
    parts = path if isinstance(path, list) else [str(path)]
    return " ".join([*map(str, parts), str(metadata.get("section") or "")])


def shared_title_words(lesson_titles: Iterable[str]) -> frozenset[str]:
    """Words in two or more of a subject's lesson titles ("Sri Lanka", "British", "Computer") -- they don't say
    which lesson a heading belongs to."""
    counts = Counter(word for title in lesson_titles for word in _heading_words(title))
    return frozenset(word for word, n in counts.items() if n >= 2)


def matches_lesson(
    result: SearchResult,
    phrases: Iterable[str],
    *,
    title: str | None = None,
    shared_words: frozenset[str] = frozenset(),
) -> bool:
    """True when the chunk sits under a heading naming the lesson or one of its subtopics.

    Strict, for every phrase (the lesson's subtopics, and its title): every meaningful word of the name appears
    in the chunk's headings (or, for a long name, at least three quarters of them). Looser, for the lesson
    `title` only -- textbook headings rarely repeat a long curriculum title ("Logic Gates with Boolean
    Functions" vs. "4.3 Basic Logic Gates"): leaving out words several lesson titles share (`shared_words`),
    at least half the title's remaining words, and never fewer than two, appear in the headings.
    """
    heading = _heading_words(heading_text(result))
    if not heading:
        return False
    if title is not None:
        distinctive = _heading_words(title) - shared_words
        if len(distinctive) >= 2 and len(distinctive & heading) >= max(2, math.ceil(len(distinctive) / 2)):
            return True
    for phrase in (*phrases, *([title] if title else [])):
        wanted = _heading_words(phrase)
        if not wanted:
            continue
        hits = len(wanted & heading)
        if hits == len(wanted) or (len(wanted) >= 4 and hits >= 0.75 * len(wanted)):
            return True
    return False
