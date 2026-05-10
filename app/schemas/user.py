"""
schemas/user.py
───────────────
Pydantic v2 schemas for user-related request and response bodies.
"""
from datetime import datetime
from pydantic import BaseModel


class UserOut(BaseModel):
    """Returned after a user is created or retrieved."""
    id: int
    clerk_id: str
    username: str | None
    created_at: datetime

    # Pydantic v2 way to enable ORM mode (formerly `orm_mode = True`)
    model_config = {"from_attributes": True}
