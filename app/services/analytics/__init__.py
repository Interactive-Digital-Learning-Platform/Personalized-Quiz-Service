"""
services/analytics/
────────────────────
Layered architecture behind GET /analytics/me, split out of what used to be
a single ~1200-line function in app/services/analytics_service.py.

Layers (each a separate module in this package):

    types.py        — typed dataclasses for every DB row shape and
                       intermediate aggregate passed between layers
                       (requirement: "use typed internal data structures").

    queries.py       — ALL database access for this endpoint. Every query
                       here already existed in the pre-refactor function,
                       moved verbatim — no new queries were added, none were
                       removed. Every query is a single bounded, aggregate
                       SQL statement scoped to one user; none of them run in
                       a per-subject or per-topic loop (see
                       orchestrator.py's docstring for the full query count
                       and why it doesn't grow with the user's data volume).

    summary_service.py, subject_service.py, topic_service.py,
    trend_service.py, repeated_mistake_service.py, mastery_service.py,
    growth_service.py, recommendation_service.py
                     — the 8 calculation/orchestration services. Each owns
                       ONE section of the final response and contains NO
                       database access of its own — they're pure functions
                       over the typed data queries.py already fetched, so
                       adding/changing a calculation never risks adding a
                       new query. The actual formulas (weighted accuracy,
                       mastery score, growth score, etc.) still live in
                       their original dedicated modules (app/services/
                       scoring_service.py, mastery_service.py,
                       growth_service.py, recommendation_service.py) — these
                       new modules are the orchestration layer that feeds
                       those formulas the right data for each scope
                       (overall/subject/topic), not a reimplementation of them.

    timing.py        — lightweight phase-timing instrumentation, active
                       only when ENVIRONMENT == "development".

    orchestrator.py  — AnalyticsOrchestrationService: runs queries.py once,
                       then calls the 8 services in dependency order,
                       assembling the exact same response dict shape GET
                       /analytics/me returned before this refactor. This is
                       the only thing app/services/analytics_service.py's
                       get_user_analytics() delegates to now.
"""
