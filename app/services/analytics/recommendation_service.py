from app.services import recommendation_service as recommendation_formulas


class RecommendationService:
    # Thin wrapper so the orchestrator can call all 8 sections through the
    # same uniform interface — the real logic has no DB access of its own
    # and lives in app.services.recommendation_service.generate_recommendations().
    def __init__(self, assembled_analytics: dict):
        self._assembled_analytics = assembled_analytics

    def build(self) -> list[dict]:
        return recommendation_formulas.generate_recommendations(self._assembled_analytics)
