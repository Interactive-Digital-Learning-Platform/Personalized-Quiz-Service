import json
from functools import cache
from pathlib import Path

_CURRICULUM_DIR = Path(__file__).resolve().parent.parent / "data" / "curriculum"


@cache
def _load_grade(grade: int) -> dict[str, list[str]] | None:
    path = _CURRICULUM_DIR / f"grade_{grade}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def has_curriculum(grade: int) -> bool:
    return _load_grade(grade) is not None


def subjects_for_grade(grade: int) -> list[str]:
    return list((_load_grade(grade) or {}).keys())


def lessons_for(grade: int, subject: str) -> list[str]:
    curriculum = _load_grade(grade) or {}
    canonical = canonical_subject(grade, subject)
    if canonical is None:
        return []
    return list(curriculum[canonical])


def canonical_subject(grade: int, subject: str) -> str | None:
    curriculum = _load_grade(grade)
    if curriculum is None:
        return None
    target = subject.strip().lower()
    for candidate in curriculum:
        if candidate.strip().lower() == target:
            return candidate
    return None


def canonical_lesson(grade: int, subject: str, lesson: str) -> str | None:
    lessons = lessons_for(grade, subject)
    target = lesson.strip().lower()
    for candidate in lessons:
        if candidate.strip().lower() == target:
            return candidate
    return None


@cache
def _load_subtopics(grade: int) -> dict[str, dict[str, list[str]]]:
    # subject -> lesson -> section headings, taken from each textbook's contents page (only books whose
    # contents list "1.1 Section 02"-style entries have them). Used to widen RAG retrieval for a lesson.
    path = _CURRICULUM_DIR / f"subtopics_grade_{grade}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def subtopics_for(grade: int, subject: str, lesson: str) -> list[str]:
    canonical_subj = canonical_subject(grade, subject)
    canonical_les = canonical_lesson(grade, subject, lesson)
    if canonical_subj is None or canonical_les is None:
        return []
    return list(_load_subtopics(grade).get(canonical_subj, {}).get(canonical_les, []))
