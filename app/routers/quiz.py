from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth import verify_clerk_jwt
from app.database.session import get_db
from app.schemas.quiz import GenerateQuizRequest, QuizCreateResponse, SubmitAnswersRequest, FeedbackResponse, AIQuestion
from app.services.quiz_service import create_quiz_and_questions, summarize_and_feedback
from app.services.analytics_service import compute_quiz_metrics
from app.models.question import Question

router = APIRouter(prefix="/quizzes", tags=["quizzes"])


@router.post("/generate", response_model=QuizCreateResponse)
async def generate_quiz(payload: GenerateQuizRequest, token_payload: dict = Depends(verify_clerk_jwt), db: AsyncSession = Depends(get_db)):
    clerk_id = token_payload.get("sub")
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")

    quiz_id, questions = await create_quiz_and_questions(db, clerk_id, payload.grade, payload.subject, payload.lesson, payload.difficulty, payload.count)

    ai_questions = [AIQuestion(**q) for q in questions]
    return QuizCreateResponse(quiz_id=quiz_id, questions=ai_questions)


@router.post("/{quiz_id}/submit", response_model=FeedbackResponse)
async def submit_answers(quiz_id: int, payload: SubmitAnswersRequest, token_payload: dict = Depends(verify_clerk_jwt), db: AsyncSession = Depends(get_db)):
    clerk_id = token_payload.get("sub")
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Persist responses
    from app.models.response import UserResponse

    # find user id
    res = await db.execute("SELECT id FROM users WHERE clerk_id = :cid", {"cid": clerk_id})
    row = res.first()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = row[0]

    for ans in payload.answers:
        # load question to check correct answer if available
        qrow = await db.execute("SELECT correct_answer FROM questions WHERE id = :qid", {"qid": ans.question_id})
        q = qrow.first()
        correct = None
        if q:
            correct = q[0]

        is_correct = None
        if correct is not None:
            is_correct = (str(correct).strip().lower() == str(ans.chosen_answer).strip().lower())

        ur = UserResponse(user_id=user_id, quiz_id=quiz_id, question_id=ans.question_id, chosen_answer=ans.chosen_answer, is_correct=is_correct, response_time=ans.response_time)
        db.add(ur)

    await db.commit()

    metrics = await compute_quiz_metrics(db, quiz_id, user_id)
    fb = await summarize_and_feedback(db, quiz_id, metrics)

    # merge
    return FeedbackResponse(accuracy=metrics["accuracy"], average_response_time=metrics["average_response_time"], weak_topics=fb.get("weak_topics", metrics.get("weak_topics", [])), suggestions=fb.get("suggestions", []))
