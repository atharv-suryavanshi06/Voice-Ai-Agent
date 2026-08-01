"""
recommendation package

Exposes the core RecommendationEngine and Policy classes for policy matching and scoring.
"""

from recommendation.policy import Policy
from recommendation.recommendation_engine import RecommendationEngine

__all__ = ["Policy", "RecommendationEngine"]
