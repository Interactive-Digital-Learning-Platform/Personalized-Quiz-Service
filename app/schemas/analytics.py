"""
schemas/analytics.py
─────────────────────
Pydantic v2 schemas for analytics and AI feedback endpoints.
"""
from datetime import datetime
from pydantic import BaseModel


# ── GET /analytics/me ─────────────────────────────────────────────────────────

class SubjectAnalytics(BaseModel):
    """Analytics for a single subject."""
    subject: str
    accuracy: float            # Percentage (0.0 – 100.0)
    avg_response_time: float   # Seconds
    weak_topic: str | None     # The topic they struggle with most


class UserAnalyticsResponse(BaseModel):
    """Full analytics profile for the authenticated user."""
    overall_accuracy: float
    overall_avg_response_time: float
    total_sessions: int
    subjects: list[SubjectAnalytics]
    # Subjects ordered best → worst accuracy
    strong_subjects: list[str]
    weak_subjects: list[str]


# ── GET /analytics/feedback ────────────────────────────────────────────────────

class AIFeedbackResponse(BaseModel):
    """AI-generated improvement suggestions from Groq."""
    weak_areas: list[str]        # Subject/topic areas needing work
    strong_areas: list[str]      # What the user is doing well
    suggestions: list[str]       # Concrete, actionable improvement tips
    motivational_note: str       # A short encouraging message from the AI
    generated_at: datetime       # When this feedback was generated
