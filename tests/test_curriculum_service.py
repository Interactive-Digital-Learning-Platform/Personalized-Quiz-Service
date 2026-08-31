"""
tests/test_curriculum_service.py
─────────────────────────────────
Covers app.services.curriculum_service — the fixed Grade 10/11 Sri Lankan
syllabus loader (app/data/curriculum/grade_{N}.json). Pure functions, no DB
involved.
"""
from app.services import curriculum_service


def test_has_curriculum_true_for_grades_with_data():
    assert curriculum_service.has_curriculum(10) is True
    assert curriculum_service.has_curriculum(11) is True


def test_has_curriculum_false_for_grade_without_data():
    assert curriculum_service.has_curriculum(7) is False
    assert curriculum_service.has_curriculum(12) is False


def test_subjects_for_grade_matches_known_taxonomy():
    subjects10 = curriculum_service.subjects_for_grade(10)
    assert set(subjects10) == {
        "Mathematics", "Science", "Geography", "History", "English", "ICT",
    }
    subjects11 = curriculum_service.subjects_for_grade(11)
    assert set(subjects11) == {
        "Mathematics", "Science", "Geography", "History", "English", "ICT",
        "Health and Physical Education",
    }


def test_subjects_for_grade_empty_for_unknown_grade():
    assert curriculum_service.subjects_for_grade(7) == []


def test_lessons_for_returns_fixed_list():
    lessons = curriculum_service.lessons_for(10, "Mathematics")
    assert "Fractions" in lessons
    assert "Percentages" in lessons
    assert len(lessons) > 0


def test_lessons_for_unknown_subject_returns_empty():
    assert curriculum_service.lessons_for(10, "NotASubject") == []


def test_canonical_subject_is_case_insensitive():
    assert curriculum_service.canonical_subject(10, "mathematics") == "Mathematics"
    assert curriculum_service.canonical_subject(10, "MATHEMATICS") == "Mathematics"
    assert curriculum_service.canonical_subject(10, "  Mathematics  ".strip()) == "Mathematics"


def test_canonical_subject_returns_none_for_unknown():
    assert curriculum_service.canonical_subject(10, "NotASubject") is None
    assert curriculum_service.canonical_subject(7, "Mathematics") is None


def test_canonical_lesson_is_case_insensitive_and_scoped_to_subject():
    assert curriculum_service.canonical_lesson(10, "Mathematics", "fractions") == "Fractions"
    assert curriculum_service.canonical_lesson(10, "Mathematics", "FRACTIONS") == "Fractions"


def test_canonical_lesson_returns_none_when_lesson_not_in_that_subject():
    # "Photosynthesis" is a Grade 11 Science lesson, not a Grade 10 Mathematics one.
    assert curriculum_service.canonical_lesson(10, "Mathematics", "Photosynthesis") is None


def test_same_lesson_name_can_exist_under_different_grades():
    # The motivating scenario for making `grade` part of LessonMastery's
    # identity: "Percentages" is a real lesson in both Grade 10 and Grade 11
    # Mathematics.
    assert curriculum_service.canonical_lesson(10, "Mathematics", "Percentages") == "Percentages"
    assert curriculum_service.canonical_lesson(11, "Mathematics", "Percentages") == "Percentages"
