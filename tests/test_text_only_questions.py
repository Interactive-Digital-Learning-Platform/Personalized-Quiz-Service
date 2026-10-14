"""
tests/test_text_only_questions.py
─────────────────────────────────
Practice questions must be answerable from their text alone: anything pointing
at a graph/diagram/table/image that can't be shown is dropped when generated
and skipped when read back from the pool. Same rules as Quiz-Battle-Service.
"""
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.models.question import Question
from app.services import groq_service, quiz_service
from app.services.groq_service import _checked_correct_answer as check
from app.services.groq_service import references_missing_visual
from app.services.quiz_service import get_or_create_user
from tests.conftest import TEST_CLERK_ID

NEEDS_A_PICTURE = [
    "Based on the bar graph below, which month had the highest rainfall?",
    "Refer to the following table. What is the total?",
    "In the diagram, what does label X represent?",
    "What is shown above?",
    "According to Figure 2, which gas is produced?",
    "Using the given pie chart, find the largest sector.",
    "Look at this picture. What is the animal doing?",
    "The diagram shows a food web. Which organism is a producer?",
    "A bar graph shows the number of books read by students: 10, 15, 12, 8. Which class read the most books?",
    "A pie chart shows how a family spends its income. Which item takes the largest share?",
    "The table shows the rainfall for four months. Which month was the wettest?",
]
TEXT_ONLY = [
    "Which type of graph is best for showing parts of a whole?",
    "Which element is in Group 1 of the periodic table?",
    "What happens to the water table during a drought?",
    "What is 7 x 8 in the 7 times table?",
    "A shop sold 12, 15 and 9 books on three days. What is the mean?",
    "What is the output of the truth table for an AND gate when both inputs are 1?",
    "Which of the following is a prime number?",
    "A bar chart is used to compare what?",
    "Why do scientists draw a diagram of an experiment?",
    "The graph of y = log2(x - 3) is obtained from y = log2 x by which transformation?",
    "A miniature map shows a circular pond with a radius of 2 cm. The map scale is 1 cm : 5 m. What is the actual perimeter?",
    "If a map shows north at the top, which direction does the right side of the map correspond to?",
    "How does the perimeter of a figure change when the figure is enlarged by a scale factor of 2?",
    "In a relational database, which key uniquely identifies each record in a table?",
    "Class A read 10 books, Class B 15, Class C 12 and Class D 8. Which class read the most books?",
]


@pytest.mark.parametrize("text", NEEDS_A_PICTURE)
def test_questions_pointing_at_a_missing_visual_are_dropped(text):
    assert references_missing_visual(text)
    assert check(text, ["A", "B", "C", "D"], "A", index=0, kind="q") is None


@pytest.mark.parametrize("text", TEXT_ONLY)
def test_text_only_questions_are_kept(text):
    assert not references_missing_visual(text)
    assert check(text, ["A", "B", "C", "D"], "A", index=0, kind="q") == "A"


def test_prompts_tell_the_ai_to_stay_text_only():
    # Both generators' system prompts (single-subject and shuffle), plus the textbook-excerpt rules.
    assert inspect.getsource(groq_service).count("- TEXT ONLY:") == 2
    assert "never base a" in groq_service._REFERENCE_MATERIAL_RULES


async def test_generated_picture_question_is_discarded():
    good = {"question": "What is 2 + 2?", "options": ["3", "4", "5", "6"], "correct_answer": "4",
            "explanation": "", "lesson": "Arithmetic"}
    picture = {"question": NEEDS_A_PICTURE[8], "options": ["A", "B", "C", "D"], "correct_answer": "B",
               "explanation": "", "lesson": "Statistics"}
    message = SimpleNamespace(content=json.dumps({"questions": [picture, good]}))
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))

    with patch.object(groq_service._groq_client.chat.completions, "create", create):
        result = await groq_service.generate_questions(
            grade=7, subject="Mathematics", difficulty="easy", question_count=2,
        )

    assert [q["question"] for q in result] == ["What is 2 + 2?"]


async def test_pool_reads_skip_existing_picture_questions(db_session):
    user = await get_or_create_user(db_session, TEST_CLERK_ID)
    db_session.add_all([
        Question(question=text, options=["A", "B", "C", "D"], correct_answer="A",
                 subject="Mathematics", lesson="Statistics", difficulty="medium", grade=7)
        for text in (NEEDS_A_PICTURE[8], NEEDS_A_PICTURE[1], TEXT_ONLY[4])
    ])
    await db_session.commit()

    picked = await quiz_service._fetch_pool_questions(
        db_session, user_id=user.id, subject="Mathematics", difficulty="medium", grade=7,
        limit=3, exclude_ids=set(), exclude_texts=[],
    )

    assert [q.question for q in picked] == [TEXT_ONLY[4]]
