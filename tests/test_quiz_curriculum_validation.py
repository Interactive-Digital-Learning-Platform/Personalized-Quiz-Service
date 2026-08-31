"""
tests/test_quiz_curriculum_validation.py
──────────────────────────────────────────
Covers GenerateQuizRequest's curriculum-aware validation (app/schemas/quiz.py
validate_curriculum_fields) — subject/lesson are checked against the fixed
Grade 10/11 syllabus (app/services/curriculum_service.py) when curriculum
data exists for the request's grade, and left fully permissive (today's
free-text behavior) for grades without curriculum data.
"""
import pytest
from pydantic import ValidationError

from app.schemas.quiz import GenerateQuizRequest


def test_valid_subject_and_lesson_for_curriculum_grade_is_accepted():
    req = GenerateQuizRequest(grade=10, subject="Mathematics", lesson="Fractions", question_count=5)
    assert req.subject == "Mathematics"
    assert req.lesson == "Fractions"


def test_subject_casing_is_normalized_to_canonical():
    req = GenerateQuizRequest(grade=10, subject="mathematics", question_count=5)
    assert req.subject == "Mathematics"


def test_lesson_casing_is_normalized_to_canonical():
    req = GenerateQuizRequest(grade=10, subject="Mathematics", lesson="fractions", question_count=5)
    assert req.lesson == "Fractions"


def test_invalid_subject_for_curriculum_grade_is_rejected():
    with pytest.raises(ValidationError, match="not a valid Grade 10 subject"):
        GenerateQuizRequest(grade=10, subject="NotASubject", question_count=5)


def test_invalid_lesson_for_curriculum_grade_is_rejected():
    with pytest.raises(ValidationError, match="not a valid lesson"):
        GenerateQuizRequest(grade=10, subject="Mathematics", lesson="NotALesson", question_count=5)


def test_lesson_valid_in_wrong_subject_is_rejected():
    # "Photosynthesis" is a real Grade 11 Science lesson, not a Mathematics one.
    with pytest.raises(ValidationError, match="not a valid lesson"):
        GenerateQuizRequest(grade=10, subject="Mathematics", lesson="Photosynthesis", question_count=5)


def test_grade_without_curriculum_data_is_fully_permissive():
    req = GenerateQuizRequest(grade=7, subject="AnythingGoes", lesson="WhateverTopic", question_count=5)
    assert req.subject == "AnythingGoes"
    assert req.lesson == "WhateverTopic"


def test_shuffle_subjects_normalized_for_curriculum_grade():
    req = GenerateQuizRequest(grade=11, shuffle=True, subjects=["science", "ict"], question_count=10)
    assert req.subjects == ["Science", "ICT"]


def test_shuffle_invalid_subject_for_curriculum_grade_is_rejected():
    with pytest.raises(ValidationError, match="not a valid Grade 11 subject"):
        GenerateQuizRequest(grade=11, shuffle=True, subjects=["Science", "NotASubject"], question_count=10)


def test_shuffle_subjects_permissive_for_grade_without_curriculum():
    req = GenerateQuizRequest(grade=7, shuffle=True, subjects=["Whatever", "Anything"], question_count=10)
    assert req.subjects == ["Whatever", "Anything"]


def test_health_and_pe_only_valid_for_grade_11():
    GenerateQuizRequest(grade=11, subject="Health and Physical Education", question_count=5)
    with pytest.raises(ValidationError, match="not a valid Grade 10 subject"):
        GenerateQuizRequest(grade=10, subject="Health and Physical Education", question_count=5)
