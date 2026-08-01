"""
recommendation_engine.py

Contains the core RecommendationEngine logic, including policy catalog loading,
strict filtering based on user demographic and medical constraints, and heuristic
scoring to rank candidate policies.
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, TYPE_CHECKING
from recommendation.policy import Policy

if TYPE_CHECKING:
    from conversation.customer_profile import CustomerProfile

class RecommendationEngine:
    """Loads a catalog of insurance policies and recommends the best options for a customer."""

    def __init__(self, catalog_path: Optional[str] = None):
        """
        Initializes the Recommendation Engine.

        Args:
            catalog_path: Path to the JSON catalog file. If None, it defaults to
                          'policy_catalog.json' in the same directory as this file.
        """
        if catalog_path is None:
            catalog_path = os.path.join(os.path.dirname(__file__), "policy_catalog.json")
        self.catalog_path = catalog_path
        self.policies = self.load_catalog()

    def load_catalog(self) -> List[Policy]:
        """Loads and parses the policy catalog JSON file."""
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Policy catalog file not found at: {self.catalog_path}")
        
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return [
            Policy.from_dict(item)
            for item in data
            if item.get("_ingestion_status", "active") == "active"
        ]

    def filter_policies(self, profile: CustomerProfile, strict: bool = True) -> List[Policy]:
        """
        Filters policies based on strict eligibility criteria.

        Args:
            profile: The CustomerProfile to evaluate against.
            strict: If True, enforces strict budget and coverage limits. If False,
                    these limits are relaxed to allow showing the closest options.
        """
        is_family_needed = (
            (profile.family_members is not None and profile.family_members > 1) or
            profile.parents_included is True or
            profile.children_included is True
        )
        is_individual_needed = (
            profile.family_members == 1 and
            profile.parents_included is not True and
            profile.children_included is not True
        )

        eligible = []
        for policy in self.policies:
            # 1. Plan Type strict filtering
            if is_family_needed and policy.plan_type != "Family Floater":
                continue
            if is_individual_needed and policy.plan_type != "Individual":
                continue

            # 2. Age Boundaries
            if profile.age is not None:
                max_age = policy.max_age if policy.max_age is not None else 99
                min_age = policy.min_age if policy.min_age is not None else 0
                if not (min_age <= profile.age <= max_age):
                    continue

            # 3. Smoker Exclusions
            if profile.smoker is True and not policy.smoker_allowed:
                continue

            # 4. Medical Exclusions (Diabetes & Hypertension)
            if profile.existing_diseases:
                has_diabetes = any(
                    "diabet" in d.lower() or "sugar" in d.lower()
                    for d in profile.existing_diseases
                )
                has_hypertension = any(
                    "hypertens" in d.lower() or "bp" in d.lower() or "blood pressure" in d.lower() or "tension" in d.lower()
                    for d in profile.existing_diseases
                )

                if has_diabetes and not policy.covers_diabetes:
                    continue
                if has_hypertension and not policy.covers_hypertension:
                    continue

            # 5. Family Inclusions
            if profile.parents_included is True and not policy.parents_allowed:
                continue
            if profile.children_included is True and not policy.children_allowed:
                continue

            # 6. Budget constraint (relaxed if not strict)
            if strict and profile.budget is not None and profile.budget > 0:
                if policy.premium > profile.budget:
                    continue

            # 7. Minimum Coverage Required constraint (relaxed if not strict)
            if strict and profile.coverage_required is not None and profile.coverage_required > 0:
                if policy.sum_insured < profile.coverage_required:
                    continue

            eligible.append(policy)
        return eligible

    def calculate_match_score(self, profile: CustomerProfile, policy: Policy) -> float:
        """
        Calculates a heuristic match score (base 100.0) for ranking eligible policies.
        Higher score means a better recommendation fit.
        """
        score = 100.0

        # 1. Budget Savings Bonus (up to +30 points) or Overage Penalty (up to -50 points)
        # Reward policies within budget, or penalize those exceeding budget (when filters are relaxed).
        if profile.budget is not None and profile.budget > 0:
            savings = profile.budget - policy.premium
            if savings >= 0:
                # Greater savings = higher bonus
                score += (savings / profile.budget) * 30.0
            else:
                # Overage penalty: subtract points based on how much premium exceeds budget
                overage_ratio = abs(savings) / profile.budget
                score -= min(overage_ratio * 20.0, 50.0)

        # 2. Coverage Exceedance Bonus (up to +20 points) or Deficit Penalty (up to -30 points)
        # Reward policies meeting/exceeding requirements, or penalize those falling short.
        if profile.coverage_required is not None and profile.coverage_required > 0:
            coverage_ratio = policy.sum_insured / profile.coverage_required
            if coverage_ratio >= 1.0:
                # Score bonus scaled with coverage ratio, capped at 20.0 points
                score += min((coverage_ratio - 1.0) * 10.0, 20.0)
            else:
                # Deficit penalty: subtract points based on coverage shortfall
                deficit_ratio = (profile.coverage_required - policy.sum_insured) / profile.coverage_required
                score -= deficit_ratio * 30.0

        # 3. Preferred Insurer Bonus (+25 points)
        if profile.preferred_insurer and profile.preferred_insurer.lower() != "none":
            pref = profile.preferred_insurer.lower().strip()
            pol_ins = policy.insurer.lower().strip()
            if pref in pol_ins or pol_ins in pref:
                score += 25.0

        # 4. Plan Type Fit Bonus (+20 points) / Penalty (-10 points)
        # Determine if a family plan is required based on members or specific requests.
        is_family_needed = (
            (profile.family_members is not None and profile.family_members > 1) or
            profile.parents_included is True or
            profile.children_included is True
        )
        if is_family_needed:
            if policy.plan_type == "Family Floater":
                score += 20.0
            else:
                score -= 10.0
        else:
            if policy.plan_type == "Individual":
                score += 20.0
            else:
                score -= 10.0

        return score

    def recommend(self, profile: CustomerProfile, limit: int = 3, strict: bool = True) -> List[Policy]:
        """
        Recommends the top matching policies for the given customer profile.

        Args:
            profile: The CustomerProfile of the caller.
            limit: The maximum number of recommendations to return.
            strict: If True, uses strict budget and coverage filters. If no policies
                    are found, it will try again with relaxed criteria.
        """
        # Attempt strict filtering first
        eligible = self.filter_policies(profile, strict=strict)
        
        # If strict returns nothing, try relaxing budget/coverage restrictions
        if not eligible and strict:
            eligible = self.filter_policies(profile, strict=False)

        # Score the eligible policies
        scored_policies = []
        for policy in eligible:
            score = self.calculate_match_score(profile, policy)
            scored_policies.append((score, policy))

        # Sort by score descending (highest score first)
        scored_policies.sort(key=lambda x: x[0], reverse=True)

        # Extract just the Policy objects
        return [policy for score, policy in scored_policies[:limit]]
