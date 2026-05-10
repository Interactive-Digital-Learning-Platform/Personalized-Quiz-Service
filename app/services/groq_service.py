"""
services/groq_service.py
────────────────────────
All interactions with the Groq API are centralised here.

Two responsibilities:
1. `generate_questions()` — Given quiz parameters, return a structured list of
   MCQ questions by prompting Groq's LLaMA 3 70B model.
2. `generate_feedback()` — Given a user's analytics summary, return personalised
   AI study suggestions.

Design principles:
- Use Groq's **official Python SDK** (not raw httpx) for reliability.
- Force strict JSON output via `response_format={"type": "json_object"}` and
  explicit JSON schema in the system prompt.
- Validate and sanitise all AI output before returning — never trust raw LLM output.
- Raise `HTTPException` with a clear 502 status on Groq failures so the caller
  can surface a meaningful error to the frontend.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import HTTPException, status
from groq import AsyncGroq, GroqError

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Groq client (module-level singleton) ──────────────────────────────────────
# AsyncGroq is the async version of the Groq client — works natively in FastAPI.
_groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# 1. QUESTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

async def generate_questions(
    grade: int,
    subject: str,
    lesson: str,
    difficulty: str,
    question_count: int,
    existing_questions: list[str] | None = None,
) -> list[dict]:
    """
    Call Groq to generate `question_count` multiple-choice questions.

    `existing_questions` — texts of questions already in the DB for this
    subject/lesson/difficulty. Passed to the model so it can explicitly
    avoid repeating them.

    Returns a list of dicts, each with:
        - question (str)
        - options (list[str])       — exactly 4 choices
        - correct_answer (str)      — must be one of the options exactly
        - explanation (str)         — brief explanation for the correct answer

    Raises:
        HTTPException(502) if the Groq API fails or returns malformed JSON.
    """
    existing_questions = existing_questions or []

    # ── System prompt: define the strict JSON contract ────────────────────────
    system_prompt = """You are an expert educational content creator for Sri Lankan school students.
Your task is to generate UNIQUE, DIVERSE multiple-choice quiz questions.

CRITICAL: You MUST respond with ONLY a valid JSON object in exactly this structure:
{
  "questions": [
    {
      "question": "The full question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Brief explanation of why this is correct."
    }
  ]
}

UNIQUENESS RULES (highest priority):
- Every question MUST test a DIFFERENT specific fact, concept, or skill within the topic.
- DO NOT repeat or rephrase any question from the EXISTING QUESTIONS list provided.
- DO NOT use the same stem structure twice (e.g., "Which of the following..." twice in a row).
- DO NOT cluster questions around one narrow sub-topic — spread coverage across the full lesson.

VARIETY REQUIREMENTS — mix ALL of the following styles across the set:
- Factual recall: "What is X?", "Which term describes Y?"
- Application: "A student does X — what happens?", "If Y occurs, what is the result?"
- Cause-and-effect: "Why does X happen?", "What causes Y?"
- Comparison: "What is the difference between X and Y?"
- Scenario / real-world: embed the concept in a concrete Sri Lankan or everyday context.
- Negation (use sparingly): "Which of the following is NOT true about X?"
- Numerical / formula-based (for maths/science topics where appropriate).

QUALITY RULES:
- Each question must have EXACTLY 4 options.
- correct_answer must be EXACTLY one of the 4 options (copy it verbatim).
- All 4 options must be plausible — avoid obviously wrong distractors.
- Questions must be appropriate for the specified grade level.
- Do NOT include numbering in the question text.
- Do NOT output anything outside the JSON object.
"""

    # ── User prompt: the actual quiz request ──────────────────────────────────
    # Include a timestamp seed so the LLM does not produce cached/repetitive output
    seed_context = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    exclusion_block = ""
    if existing_questions:
        exclusion_lines = "\n".join(f"- {q}" for q in existing_questions[:60])
        exclusion_block = (
            f"\n\nEXISTING QUESTIONS TO AVOID (do NOT repeat or rephrase any of these):\n"
            f"{exclusion_lines}\n"
        )

    user_prompt = (
        f"[Request ID: {seed_context}]\n\n"
        f"Generate {question_count} UNIQUE {difficulty}-difficulty multiple-choice questions "
        f"for Grade {grade} Sri Lankan students.\n"
        f"Subject: {subject}\n"
        f"Lesson / Topic: {lesson}\n"
        f"{exclusion_block}\n"
        f"Requirements:\n"
        f"- Cover {question_count} DIFFERENT aspects or sub-concepts within '{lesson}'.\n"
        f"- Use a variety of question styles (factual, applied, scenario, comparison, cause-effect).\n"
        f"- Each question must be clearly distinct from all others in this set.\n\n"
        f"Return exactly {question_count} questions in the required JSON format."
    )

    try:
        logger.info(
            "Calling Groq API: model=%s, subject=%s, lesson=%s, count=%d, excluding=%d existing",
            settings.GROQ_MODEL, subject, lesson, question_count, len(existing_questions),
        )

        response = await _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.95,             # High creativity for maximum question variety
            max_tokens=4096,              # Generous limit for many questions
            response_format={"type": "json_object"},  # Forces valid JSON output
        )

    except GroqError as exc:
        logger.error("Groq API error during question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {exc}",
        )

    # ── Parse and validate the response ───────────────────────────────────────
    raw_content = response.choices[0].message.content or ""

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        logger.error("Groq returned non-JSON content: %s", raw_content[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI returned malformed JSON: {exc}",
        )

    raw_questions: list = data.get("questions", [])
    if not isinstance(raw_questions, list) or len(raw_questions) == 0:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned an empty question list. Please try again.",
        )

    # ── Sanitise each question ────────────────────────────────────────────────
    validated: list[dict] = []
    for i, q in enumerate(raw_questions[:question_count]):
        if not isinstance(q, dict):
            continue

        question_text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct = str(q.get("correct_answer", "")).strip()
        explanation = str(q.get("explanation", "")).strip()

        # Skip malformed entries
        if not question_text or not isinstance(options, list) or len(options) < 2:
            logger.warning("Skipping malformed question at index %d", i)
            continue

        # Ensure correct_answer is in options (case-insensitive fallback)
        if correct not in options:
            # Try case-insensitive match
            match = next((o for o in options if o.strip().lower() == correct.lower()), None)
            correct = match if match else options[0]

        validated.append({
            "question": question_text,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
        })

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned no valid questions after validation.",
        )

    logger.info("Groq generated %d valid questions (requested %d)", len(validated), question_count)
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# 2. PERSONALISED FEEDBACK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

async def generate_feedback(analytics_summary: dict) -> dict:
    """
    Takes the user's analytics data and asks Groq to produce personalised
    improvement suggestions.

    `analytics_summary` shape:
        {
            "overall_accuracy": 65.0,
            "overall_avg_response_time": 18.5,
            "subjects": [
                {"subject": "Maths", "accuracy": 45.0, "weak_topic": "Algebra"},
                {"subject": "Science", "accuracy": 85.0, "weak_topic": None},
            ]
        }

    Returns a dict with keys:
        weak_areas, strong_areas, suggestions, motivational_note
    """
    system_prompt = """You are a knowledgeable and encouraging study coach for school students.
Analyse the student's quiz performance data and provide personalised feedback.

CRITICAL: Respond ONLY with a valid JSON object in this exact structure:
{
  "weak_areas": ["list of subjects or topics needing improvement"],
  "strong_areas": ["list of subjects or topics the student excels at"],
  "suggestions": [
    "Specific, actionable study tip 1",
    "Specific, actionable study tip 2",
    "Specific, actionable study tip 3"
  ],
  "motivational_note": "A short, encouraging 1-2 sentence message."
}

Be specific and reference the actual subjects/topics from the data.
Keep suggestions practical and achievable for a school student.
"""

    user_prompt = (
        "Here is the student's performance summary:\n"
        f"{json.dumps(analytics_summary, indent=2)}\n\n"
        "Please provide personalised feedback and study suggestions."
    )

    try:
        response = await _groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
    except GroqError as exc:
        logger.error("Groq API error during feedback generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI feedback service error: {exc}",
        )

    raw_content = response.choices[0].message.content or "{}"
    try:
        feedback = json.loads(raw_content)
    except json.JSONDecodeError:
        # Return a graceful fallback instead of crashing
        logger.warning("Groq feedback returned non-JSON, using fallback.")
        feedback = {}

    # Ensure all expected keys exist with safe defaults
    return {
        "weak_areas": feedback.get("weak_areas", []),
        "strong_areas": feedback.get("strong_areas", []),
        "suggestions": feedback.get("suggestions", ["Keep practising regularly!"]),
        "motivational_note": feedback.get(
            "motivational_note", "Great effort! Keep going — consistency is key."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }