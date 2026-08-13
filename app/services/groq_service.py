import asyncio
import json
import logging
import random
from datetime import datetime, timezone

from fastapi import HTTPException, status
from groq import (
    APIConnectionError,
    APITimeoutError,
    AsyncGroq,
    GroqError,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

_groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)

# Transient only -- bad auth/request would just fail the same way again.
_TRANSIENT_GROQ_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


async def _create_chat_completion_with_retry(*, messages: list[dict]):
    # Backoff + jitter so one rate-limit blip doesn't fall back to cache.
    last_exc: Exception | None = None
    for attempt in range(settings.GROQ_MAX_RETRIES + 1):
        try:
            return await _groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.95,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except _TRANSIENT_GROQ_ERRORS as exc:
            last_exc = exc
            if attempt == settings.GROQ_MAX_RETRIES:
                break
            delay = settings.GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Transient Groq error (%s) on attempt %d/%d, retrying in %.1fs: %s",
                type(exc).__name__, attempt + 1, settings.GROQ_MAX_RETRIES + 1, delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_exc


async def generate_questions(
    grade: int,
    subject: str,
    difficulty: str,
    question_count: int,
    lesson: str | None = None,
    existing_questions: list[str] | None = None,
    avoid_lessons: list[str] | None = None,
    preferred_lessons: list[str] | None = None,
    adaptive_context: str | None = None,
    telemetry: dict | None = None,
) -> list[dict]:
    # Requests `question_count` MCQs; if `lesson` is omitted each question
    # gets its own random lesson. `preferred_lessons`/`adaptive_context` are
    # optional adaptive-mastery hints (weak lessons, plain-language standing)
    # -- bands/names only, never raw DB rows/scores. Raises HTTPException(502)
    # on any Groq/parsing failure.
    existing_questions = existing_questions or []
    avoid_lessons = avoid_lessons or []
    random_lessons = lesson is None

    system_prompt = """You are an expert educational content creator for Sri Lankan school students.
Your task is to generate UNIQUE, DIVERSE multiple-choice quiz questions.

CRITICAL: You MUST respond with ONLY a valid JSON object in exactly this structure:
{
  "questions": [
    {
      "question": "The full question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Brief explanation of why this is correct.",
      "lesson": "The specific lesson/topic this question covers"
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
- The "lesson" field must always be a real, specific topic name from the
  subject's standard syllabus (e.g. "Algebra", "Photosynthesis") — never the
  subject name itself and never generic ("General", "Miscellaneous", etc).
- Do NOT include numbering in the question text.
- Do NOT output anything outside the JSON object.
"""

    seed_context = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    exclusion_block = ""
    if existing_questions:
        exclusion_lines = "\n".join(f"- {q}" for q in existing_questions[:60])
        exclusion_block = (
            f"\n\nEXISTING QUESTIONS TO AVOID (do NOT repeat or rephrase any of these):\n"
            f"{exclusion_lines}\n"
        )

    if random_lessons:
        avoid_lessons_block = ""
        if avoid_lessons:
            avoid_lines = "\n".join(f"- {l}" for l in avoid_lessons[:10])
            avoid_lessons_block = (
                f"\n\nThis student was recently quizzed on these lessons for this subject — "
                f"prefer OTHER lessons where possible for variety:\n{avoid_lines}\n"
            )

        preferred_lessons_block = ""
        if preferred_lessons:
            preferred_lines = "\n".join(f"- {l}" for l in preferred_lessons[:10])
            preferred_lessons_block = (
                f"\n\nThis student would benefit from extra practice in these lessons — "
                f"include AT LEAST ONE question from each if possible, without making every "
                f"question come from only these:\n{preferred_lines}\n"
            )

        adaptive_context_block = f"\n\nStudent context: {adaptive_context}\n" if adaptive_context else ""

        user_prompt = (
            f"[Request ID: {seed_context}]\n\n"
            f"Generate {question_count} UNIQUE {difficulty}-difficulty multiple-choice questions "
            f"for Grade {grade} Sri Lankan students.\n"
            f"Subject: {subject}\n"
            f"{exclusion_block}"
            f"{avoid_lessons_block}"
            f"{preferred_lessons_block}"
            f"{adaptive_context_block}\n"
            f"Requirements:\n"
            f"- Do NOT focus on a single lesson. EACH question must come from a DIFFERENT, "
            f"randomly chosen lesson/topic within the full '{subject}' syllabus for this grade.\n"
            f"- Spread the {question_count} questions across the breadth of the subject — "
            f"avoid picking the same lesson for more than one question unless the subject "
            f"genuinely has too few lessons to avoid it.\n"
            f"- Set each question's \"lesson\" field to the specific topic IT individually covers.\n"
            f"- Use a variety of question styles (factual, applied, scenario, comparison, cause-effect).\n"
            f"- Each question must be clearly distinct from all others in this set.\n\n"
            f"Return exactly {question_count} questions in the required JSON format."
        )
    else:
        adaptive_context_block = f"\n\nStudent context: {adaptive_context}\n" if adaptive_context else ""
        user_prompt = (
            f"[Request ID: {seed_context}]\n\n"
            f"Generate {question_count} UNIQUE {difficulty}-difficulty multiple-choice questions "
            f"for Grade {grade} Sri Lankan students.\n"
            f"Subject: {subject}\n"
            f"Lesson / Topic: {lesson}\n"
            f"{exclusion_block}"
            f"{adaptive_context_block}\n"
            f"Requirements:\n"
            f"- Cover {question_count} DIFFERENT aspects or sub-concepts within '{lesson}'.\n"
            f"- Set every question's \"lesson\" field to exactly \"{lesson}\".\n"
            f"- Use a variety of question styles (factual, applied, scenario, comparison, cause-effect).\n"
            f"- Each question must be clearly distinct from all others in this set.\n\n"
            f"Return exactly {question_count} questions in the required JSON format."
        )

    try:
        logger.info(
            "Calling Groq API: model=%s, subject=%s, lesson=%s, count=%d, excluding=%d existing",
            settings.GROQ_MODEL, subject, lesson or "<random per question>", question_count, len(existing_questions),
        )

        response = await _create_chat_completion_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    except GroqError as exc:
        logger.error("Groq API error during question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {exc}",
        )

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

    validated: list[dict] = []
    for i, q in enumerate(raw_questions[:question_count]):
        if not isinstance(q, dict):
            continue

        question_text = str(q.get("question", "")).strip()
        options = q.get("options", [])
        correct = str(q.get("correct_answer", "")).strip()
        explanation = str(q.get("explanation", "")).strip()
        q_lesson = lesson if lesson else (str(q.get("lesson", "")).strip() or subject)

        # Drop rather than truncate a bad-count question -- could lose the correct answer.
        if not question_text or not isinstance(options, list) or len(options) != 4:
            logger.warning(
                "Skipping malformed question at index %d: expected 4 options, got %d",
                i, len(options) if isinstance(options, list) else 0,
            )
            continue

        # Drop if correct_answer matches no option -- beats defaulting to options[0].
        if correct not in options:
            match = next((o for o in options if o.strip().lower() == correct.strip().lower()), None)
            if match is None:
                logger.warning(
                    "Skipping question at index %d: correct_answer %r not found in options %r",
                    i, correct, options,
                )
                continue
            correct = match

        validated.append({
            "question": question_text,
            "options": options,
            "correct_answer": correct,
            "explanation": explanation,
            "lesson": q_lesson,
        })

    if telemetry is not None:
        invalid_count = len(raw_questions[:question_count]) - len(validated)
        telemetry["invalid_question_count"] = telemetry.get("invalid_question_count", 0) + invalid_count

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned no valid questions after validation.",
        )

    logger.info("Groq generated %d valid questions (requested %d)", len(validated), question_count)
    return validated


async def generate_feedback(analytics_summary: dict) -> dict:
    # `recommendations` is deterministic/DB-driven (recommendation_service.py) --
    # the prompt treats it as ground truth so the AI's wording stays consistent.
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

The input includes a "recommendations" array — a deterministic, already-computed
list of what to focus on next, ranked by priority, each with a "reason" and a
"recommended_action" grounded in real numbers from the database.
- If "recommendations" is non-empty: base "weak_areas" and "suggestions" on
  those entries (their subject/topic/reason/recommended_action), in priority
  order. You may rephrase them in a warmer, more encouraging tone, but do not
  contradict them or change which subjects/topics they point to.
- If "recommendations" is empty: there isn't enough data yet — give general,
  encouraging, non-numeric guidance instead (e.g. "keep practising regularly").

CRITICAL — do not invent numbers: every accuracy percentage, attempt count, or
other statistic you mention MUST come directly from the provided data. Never
fabricate or estimate a number that isn't present in the input. If you don't
have a specific figure for something, describe it qualitatively instead
(e.g. "you've been slipping in Algebra" rather than guessing a percentage).

Be specific and reference the actual subjects/topics from the data.
Keep suggestions practical and achievable for a school student.
"""

    user_prompt = (
        "Here is the student's performance summary, including a "
        "pre-computed, database-driven `recommendations` list ranked by "
        "priority (use it as ground truth — see the system instructions):\n"
        f"{json.dumps(analytics_summary, indent=2, default=str)}\n\n"
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
        logger.warning("Groq feedback returned non-JSON, using fallback.")
        feedback = {}

    return {
        "weak_areas": feedback.get("weak_areas", []),
        "strong_areas": feedback.get("strong_areas", []),
        "suggestions": feedback.get("suggestions", ["Keep practising regularly!"]),
        "motivational_note": feedback.get(
            "motivational_note", "Great effort! Keep going — consistency is key."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
