"""
services/question_fingerprint.py
──────────────────────────────────
Deterministic content fingerprint for `Question` rows.

The application always generates fresh questions via Groq on every quiz
(see quiz_service.generate_quiz()), so the exact same or an equivalent
question can end up stored as SEPARATE `Question` rows across different
generation calls — there is no existing `source_question_id`,
`duplicate_group_id`, or `original_question_id` column to identify repeats
by. This module provides the stable identifier used instead: a hash of the
question's normalized (subject, lesson, question text) triple.

Two rows with the same fingerprint are treated as "the same question" for
repeated-question/repeated-mistake analytics — see
analytics_service.get_user_analytics()'s repeated-question section and
scoring_service.compute_repeated_question_group_stats().

Uses hashlib.sha256, NOT Python's built-in hash() — the latter is salted
with a per-process random seed (PYTHONHASHSEED) for security reasons, so the
same string hashes differently across process restarts, which would silently
corrupt fingerprint-based grouping the moment the app (or a worker) restarts.
"""
import hashlib
import re


def _normalize(text: str | None) -> str:
    """Lowercase, strip, drop punctuation, collapse whitespace — so trivial
    formatting differences between two AI-generated renderings of the same
    underlying question (a trailing '?', extra spaces, a capitalized word)
    don't produce a different fingerprint."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_question_fingerprint(subject: str, lesson: str | None, question_text: str) -> str:
    """
    Deterministic (stable across processes/restarts) fingerprint of a
    question's content, scoped by subject and lesson.

    Scoping by subject/lesson is deliberate: two questions with identical
    text in different subjects are different questions for repeated-mistake
    purposes (e.g. a generic "What is the capital?" prompt could plausibly
    recur across unrelated subjects) — this fingerprint intentionally does
    NOT collapse across subjects the way a text-only match would.
    """
    payload = "|".join([_normalize(subject), _normalize(lesson), _normalize(question_text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
