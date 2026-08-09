import jwt
from jwt import PyJWKClient, PyJWKClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_bearer_scheme = HTTPBearer(auto_error=False)
_jwks_client = PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    # Verifies the Clerk JWT against Clerk's public keys and returns the
    # decoded payload (at least `sub`, the Clerk user ID). In dev, if
    # AUTH_BYPASS is on, this skips verification entirely and hands back a
    # fake "dev-user" identity — lets you hit any endpoint locally without a
    # real Clerk session.
    if settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS:
        return {
            "sub": "dev-user",
            "email": "dev-user@example.com",
            "name": "Development User",
            "username": "dev-user",
            "dev_auth_bypass": True,
        }

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed. Provide: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)

        payload: dict = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.CLERK_ISSUER,
            audience=settings.CLERK_AUDIENCE or None,
            options={
                "verify_exp": True,
                "verify_iat": True,
            },
        )

    except PyJWKClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not fetch Clerk public keys: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def require_admin_or_dev(current_user: dict = Depends(get_current_user)) -> dict:
    # Gate for internal/admin-only endpoints. There's no real role system
    # here, so you get in if you're either running in dev, or your Clerk ID
    # is in ADMIN_CLERK_IDS. Still runs get_current_user first, so this only
    # adds a stricter check on top — it never skips authentication.
    if settings.ENVIRONMENT == "development":
        return current_user

    admin_ids = {cid.strip() for cid in settings.ADMIN_CLERK_IDS.split(",") if cid.strip()}
    if current_user.get("sub") in admin_ids:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This endpoint is restricted to administrators.",
    )
