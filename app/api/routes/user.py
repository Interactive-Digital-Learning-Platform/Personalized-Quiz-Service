"""
api/routes/user.py
──────────────────
User-related API endpoints:

    GET  /user/me        — Return the authenticated user's profile
    POST /user/sync      — Provision/sync a user record from Clerk
"""
import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.schemas.user import UserOut
from app.services.quiz_service import get_or_create_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["User"])


# ─────────────────────────────────────────────────────────────────────────────
# GET /user/me
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserOut,
    status_code=status.HTTP_200_OK,
    summary="Get my profile",
    description=(
        "Returns the authenticated user's database profile. "
        "Creates the user record automatically on first call (lazy provisioning)."
    ),
)
async def get_my_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    """
    Called by the React Native app on startup to confirm the user exists
    in our database and retrieve their internal ID.
    """
    clerk_id: str = current_user.get("sub", "")

    # get_or_create_user handles both first-time and returning users
    user = await get_or_create_user(db=db, clerk_id=clerk_id)

    # Optionally sync the username from the Clerk JWT (if present)
    clerk_username = current_user.get("username") or current_user.get("name")
    if clerk_username and user.username != clerk_username:
        user.username = clerk_username
        db.add(user)
        await db.commit()
        await db.refresh(user)

    logger.info("GET /user/me — clerk_id=%s, user_id=%d", clerk_id, user.id)
    return UserOut.model_validate(user)


# ─────────────────────────────────────────────────────────────────────────────
# POST /user/sync
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/sync",
    response_model=UserOut,
    status_code=status.HTTP_200_OK,
    summary="Sync user from Clerk",
    description=(
        "Explicitly provisions or updates the user record. "
        "Call this from the Clerk webhook or after login to ensure the user exists."
    ),
)
async def sync_user(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    """
    Idempotent — safe to call multiple times. Equivalent to GET /user/me
    but explicitly signals intent to sync/create the user.
    """
    clerk_id: str = current_user.get("sub", "")
    user = await get_or_create_user(db=db, clerk_id=clerk_id)

    logger.info("POST /user/sync — clerk_id=%s, user_id=%d", clerk_id, user.id)
    return UserOut.model_validate(user)
