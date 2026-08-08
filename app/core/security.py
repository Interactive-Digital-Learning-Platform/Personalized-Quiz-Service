"""
core/security.py
────────────────
Clerk JWT verification dependency for FastAPI.

Clerk issues RS256-signed JWTs. We verify them by:
1. Fetching Clerk's public JWKS (JSON Web Key Set) over HTTPS.
2. Finding the key that matches the `kid` header in the incoming JWT.
3. Decoding and validating the JWT's signature, expiry, and issuer.

The JWKS is fetched per-request via PyJWT's PyJWKClient, which internally
caches the keyset, so we don't hammer Clerk's JWKS endpoint on every call.

Returned payload contains at minimum:
    - `sub`: the Clerk user ID (e.g. "user_2abc...")
    - `email`: user's primary email (if included in Clerk session token claims)
"""
import jwt
from jwt import PyJWKClient, PyJWKClientError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

# ── Bearer token extractor ────────────────────────────────────────────────────
# `auto_error=False` lets us return a 401 manually with a clear message
# instead of a generic FastAPI error.
_bearer_scheme = HTTPBearer(auto_error=False)

# ── JWKS Client (module-level singleton) ──────────────────────────────────────
# PyJWKClient caches the fetched keys in memory, so it won't fetch on every
# request — only when the key cache is cold or a key is rotated.
_jwks_client = PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency that:
    1. Extracts the Bearer token from the Authorization header.
    2. Verifies it against Clerk's JWKS endpoint.
    3. Returns the decoded JWT payload dict.

    Raises HTTP 401 if the token is missing, expired, or invalid.

    Usage in routes:
        async def my_route(user: dict = Depends(get_current_user)):
            clerk_id = user["sub"]
    """
    # ── Dev bypass ────────────────────────────────────────────────────────────
    if settings.ENVIRONMENT == "development" and settings.AUTH_BYPASS:
        return {
            "sub": "dev-user",
            "email": "dev-user@example.com",
            "name": "Development User",
            "username": "dev-user",
            "dev_auth_bypass": True,
        }

    # ── Step 1: Ensure the token was provided ─────────────────────────────────
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed. Provide: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # ── Step 2: Fetch the matching public key from Clerk's JWKS ──────────
        signing_key = _jwks_client.get_signing_key_from_jwt(token)

        # ── Step 3: Decode & validate the JWT ────────────────────────────────
        payload: dict = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],            # Clerk always uses RS256
            issuer=settings.CLERK_ISSUER,    # Reject tokens from other issuers
            # Only validate audience if it's explicitly set in config
            audience=settings.CLERK_AUDIENCE or None,
            options={
                "verify_exp": True,          # Reject expired tokens
                "verify_iat": True,          # Reject tokens with future issued-at
            },
        )

    except PyJWKClientError as exc:
        # Raised when the JWKS endpoint is unreachable or the key isn't found
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
    """
    FastAPI dependency for internal/admin-only endpoints (e.g.
    GET /analytics/system/ai-generation) — this project has no formal
    role/permission system, so access is granted when EITHER:
    - the app is running in development (ENVIRONMENT == "development"), or
    - the authenticated user's Clerk ID (the JWT "sub" claim) is listed in
      ADMIN_CLERK_IDS.

    Still requires a valid authenticated user first (via get_current_user) —
    this only adds an ADDITIONAL restriction on top, it never loosens
    authentication itself.

    Raises HTTP 403 if neither condition is met.
    """
    if settings.ENVIRONMENT == "development":
        return current_user

    admin_ids = {cid.strip() for cid in settings.ADMIN_CLERK_IDS.split(",") if cid.strip()}
    if current_user.get("sub") in admin_ids:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This endpoint is restricted to administrators.",
    )
