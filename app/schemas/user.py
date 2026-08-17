from datetime import datetime

from pydantic import BaseModel


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
    username: str | None = None
