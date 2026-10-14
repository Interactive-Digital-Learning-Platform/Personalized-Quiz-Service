from datetime import datetime

from pydantic import BaseModel, Field


class UserOut(BaseModel):
    id: int
    clerk_id: str
    username: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserSyncRequest(BaseModel):
    # Clerk's default session token doesn't carry a username/name claim, so
    # the client sends it explicitly here (from Clerk's client-side user
    # profile) rather than the backend trying to pull it out of the JWT.
    # Constrained to Clerk's own username alphabet so the client can't push
    # arbitrary display text (it's shown to battle opponents/leaderboards);
    # when CLERK_SECRET_KEY is set the route ignores this entirely and reads
    # the username from Clerk's backend API instead.
    username: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
