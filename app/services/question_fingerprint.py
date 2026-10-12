import hashlib
import re


def _normalize(text: str | None) -> str:
    # Lowercase, strip punctuation, collapse whitespace — so two AI-generated renderings of the same question still hash to the same fingerprint.
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_question_fingerprint(subject: str, lesson: str | None, question_text: str) -> str:
    # No ID ties "the same question" across fresh Groq-generated rows, so this hash of (subject, lesson, text) fills that gap for repeated-question analytics; scoped by subject/lesson so identical wording in different subjects doesn't count as the same question. Uses sha256 (not hash()) since hash() is randomly seeded per process.
    payload = "|".join([_normalize(subject), _normalize(lesson), _normalize(question_text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
