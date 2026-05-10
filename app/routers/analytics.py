from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth import verify_clerk_jwt
from app.database.session import get_db
from app.services.analytics_service import compute_quiz_metrics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/quiz/{quiz_id}")
async def quiz_analytics(quiz_id: int, token_payload: dict = Depends(verify_clerk_jwt), db: AsyncSession = Depends(get_db)):
    clerk_id = token_payload.get("sub")
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    # find user id
    res = await db.execute("SELECT id FROM users WHERE clerk_id = :cid", {"cid": clerk_id})
    row = res.first()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = row[0]

    metrics = await compute_quiz_metrics(db, quiz_id, user_id)
    return metrics
