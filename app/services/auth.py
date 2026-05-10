import httpx
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.config import settings

security = HTTPBearer(auto_error=False)


async def verify_clerk_jwt(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    if settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS:
        return {
            "sub": "dev-user",
            "email": "dev-user@example.com",
            "name": "Development User",
            "username": "dev-user",
            "dev_auth_bypass": True,
        }

    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    token = credentials.credentials

    try:
        jwks_client = PyJWKClient(str(settings.CLERK_JWKS_URL))
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.CLERK_AUDIENCE or None,
            issuer=settings.CLERK_ISSUER,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")

    return payload
