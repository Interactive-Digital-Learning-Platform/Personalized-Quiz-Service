"""Score RAG retrieval quality against the labelled lessons in tests/data/retrieval_eval_set.json.

Uses the same retrieval path quiz generation uses (rag_service), against the live Qdrant collection, and
reports per lesson and overall:

- hit      : at least one chapter-text excerpt comes from the lesson's own pages (its contents-page range)
- empty    : no excerpt at all came back (the AI would get no reference material)
- non_text : excerpts that are contents / front matter / back matter rather than chapter text
- dupes    : excerpts repeated within one lesson's result
- keyword  : for lessons with subtopic keywords, at least one excerpt mentions one of them
- latency  : wall time per lesson lookup

Read-only: it only queries Qdrant. Run from the repo root:

    PYTHONPATH=. uv run python scripts/eval_retrieval.py                 # summary + per-lesson table
    PYTHONPATH=. uv run python scripts/eval_retrieval.py --save out.json # also save results for comparing
    PYTHONPATH=. uv run python scripts/eval_retrieval.py --compare out.json
"""
import argparse
import asyncio
import json
import logging
import re
import statistics
import time
from pathlib import Path

logging.disable(logging.WARNING)

from app.core.config import settings
from app.services import rag_service

EVAL_SET = Path(__file__).resolve().parent.parent / "tests" / "data" / "retrieval_eval_set.json"
_CONTENTS_LIKE = re.compile(r"(?:^|\s)\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Za-z][^\d]{2,80}?\s+\d{1,3}(?=\s|$)")
_FRONT_LIKE = re.compile(r"technical assistance|language editing|proof reading|board of (editors|writers)|first print", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


async def _retrieve(case: dict, k: int) -> list:
    """Excerpts (SearchResult-like: .text, .metadata) for one lesson, via the quiz service's own path."""
    if hasattr(rag_service, "retrieve_excerpts"):
        return await rag_service.retrieve_excerpts(
            grade=case["grade"], subject=case["subject"], lesson=case["lesson"], max_snippets=k,
        )
    # Pre-Phase-1 code: rag_service only returns strings, so call its retrieval step directly.
    service = await rag_service._ensure_retrieval_service()
    return await service.search(
        rag_service._build_query(case["grade"], case["subject"], case["lesson"]),
        grade=case["grade"], subject=case["subject"], top_k=k,
    )


def _is_non_text(result) -> bool:
    content_type = (result.metadata or {}).get("content_type")
    if content_type:
        return content_type != "content"
    return len(_CONTENTS_LIKE.findall(result.text)) >= 6 or bool(_FRONT_LIKE.search(result.text))


def _score(case: dict, results: list, latency_ms: float) -> dict:
    lo, hi = case["expected_pages"]
    pages_of = [set((r.metadata or {}).get("pages") or [(r.metadata or {}).get("page_start", -1)]) for r in results]
    texts = [_norm(r.text) for r in results]
    keywords = [_norm(k) for k in case.get("keywords", []) if k]
    return {
        "lesson": f'G{case["grade"]} {case["subject"]}: {case["lesson"]}',
        "n": len(results),
        # Contents pages can span a lesson's first page, so only chapter text counts as a hit.
        "hit": any(
            lo <= p <= hi for r, pages in zip(results, pages_of) if not _is_non_text(r) for p in pages
        ),
        "empty": not results,
        "non_text": sum(_is_non_text(r) for r in results),
        "dupes": len(texts) - len(set(texts)),
        "keyword": (any(k in t for k in keywords for t in texts) if keywords else None),
        "pages": [min(p) if p else None for p in pages_of],
        "latency_ms": round(latency_ms, 1),
    }


def _summary(rows: list[dict]) -> dict:
    n_results = sum(r["n"] for r in rows) or 1
    kw_rows = [r for r in rows if r["keyword"] is not None]
    return {
        "lessons": len(rows),
        "hit_rate": round(sum(r["hit"] for r in rows) / len(rows), 3),
        "empty_rate": round(sum(r["empty"] for r in rows) / len(rows), 3),
        "non_text_share": round(sum(r["non_text"] for r in rows) / n_results, 3),
        "duplicate_share": round(sum(r["dupes"] for r in rows) / n_results, 3),
        "keyword_rate": round(sum(r["keyword"] for r in kw_rows) / len(kw_rows), 3) if kw_rows else None,
        "latency_ms_mean": round(statistics.mean(r["latency_ms"] for r in rows), 1),
        "latency_ms_p95": round(sorted(r["latency_ms"] for r in rows)[int(0.95 * (len(rows) - 1))], 1),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=settings.RAG_SNIPPETS_PER_LESSON, help="excerpts per lesson")
    parser.add_argument("--save", type=Path, help="write results JSON here")
    parser.add_argument("--compare", type=Path, help="previous results JSON to compare against")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text())["cases"]
    await _retrieve(cases[0], args.k)  # warm-up: model loading shouldn't count as lesson latency

    rows = []
    for case in cases:
        start = time.perf_counter()
        results = await _retrieve(case, args.k)
        rows.append(_score(case, results, (time.perf_counter() - start) * 1000))

    for r in rows:
        flags = ("HIT " if r["hit"] else "miss") + (" EMPTY" if r["empty"] else "")
        flags += f" non_text={r['non_text']}" if r["non_text"] else ""
        flags += f" dupes={r['dupes']}" if r["dupes"] else ""
        print(f"  {flags:<26} pages={r['pages']!s:<14} {r['lesson'][:70]}")

    summary = _summary(rows)
    print("\nSUMMARY", json.dumps(summary))
    if args.compare:
        before = json.loads(args.compare.read_text())["summary"]
        print("\nmetric            before    after")
        for key in summary:
            print(f"  {key:<16} {before.get(key)!s:>8} {summary[key]!s:>8}")
    if args.save:
        args.save.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
