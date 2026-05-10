"""
core/config.py
──────────────
Central application settings using Pydantic v2 BaseSettings.
All values are loaded from environment variables (or the .env file).
This single source of truth prevents config from being scattered across the app.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── General ───────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Personalized Quiz Service"
    ENVIRONMENT: str = "development"

    # ── Database (Neon PostgreSQL) ────────────────────────────────────────────
    # Must use the asyncpg dialect: postgresql+asyncpg://...
    DATABASE_URL: str

    # ── Groq AI ───────────────────────────────────────────────────────────────
    GROQ_API_KEY: str
    # llama3-70b-8192 gives the best quality/speed trade-off on Groq's free tier
    GROQ_MODEL: str = "llama3-70b-8192"

    # ── Clerk Authentication ───────────────────────────────────────────────────
    # JWKS URL is used to fetch Clerk's public keys and verify RS256 JWTs
    CLERK_JWKS_URL: str
    CLERK_ISSUER: str
    # Audience is optional — only set if your Clerk app enforces it
    CLERK_AUDIENCE: Optional[str] = None
    # Allow local development and testing without a real Clerk session token.
    AUTH_BYPASS: bool = False

    # ── Pydantic v2 config ────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env in the project root
        env_file_encoding="utf-8",
        case_sensitive=True,      # Env vars are case-sensitive on Linux
        extra="ignore",           # Silently ignore unknown env vars
    )


# Singleton — import `settings` from this module everywhere
settings = Settings()
