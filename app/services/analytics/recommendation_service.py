"""
services/analytics/recommendation_service.py
────────────────────────────────────────────────
RecommendationService — thin wrapper around app.services.
recommendation_service.generate_recommendations() (imported below as
`recommendation_formulas`), which already takes the fully-assembled
analytics dict and needs no DB access of its own. This wrapper exists so
the orchestrator calls all 8 sections through a uniform, symmetrical
interface rather than special-casing this one as "just a function call".
"""
from app.services import recommendation_service as recommendation_formulas


class RecommendationService:
    def __init__(self, assembled_analytics: dict):
        self._assembled_analytics = assembled_analytics

    def build(self) -> list[dict]:
        return recommendation_formulas.generate_recommendations(self._assembled_analytics)
