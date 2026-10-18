"""Source labels on RAG excerpts: "[Source: science grade 10 part 1.pdf, page 46]".

rag_service prefixes each excerpt with one; the AI copies the label of the excerpt it based a question on
into that question's "source" field; groq_service keeps it only if it is exactly one of the labels it was
given (so a made-up source never gets stored) and turns it into Question.source_pages.
"""
import re

_LABEL = re.compile(r"\[Source: (?P<file>[^\],]+?), page (?P<page>\d+)\]")


def format_label(metadata: dict) -> str | None:
    filename = metadata.get("filename") or metadata.get("source")
    page = metadata.get("page_start")
    if not filename or not page:  # page 0 = ingestion didn't record a page; nothing useful to cite
        return None
    return f"[Source: {filename}, page {int(page)}]"


def labels_in(snippets) -> set[str]:
    return {m.group(0) for snippet in snippets for m in _LABEL.finditer(snippet)}


def to_source_pages(label: object, allowed: set[str]) -> list[dict] | None:
    if not isinstance(label, str):
        return None
    match = _LABEL.search(label)
    if match is None or match.group(0) not in allowed:
        return None
    return [{"file": match.group("file"), "page": int(match.group("page"))}]
