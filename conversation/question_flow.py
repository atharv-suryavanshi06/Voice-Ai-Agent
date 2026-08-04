"""
question_flow.py

The ordered list of things the agent needs to know before it can recommend
a policy, plus the (smaller) list of nice-to-have extras it can pick up
opportunistically along the way.

Required questions gate `is_profile_complete()` / `should_recommend_policy()`
and are asked in a fixed order - this is the core mechanism that keeps the
call from becoming an unstructured, LLM-improvised chat. Optional questions
never block the flow; they're only useful context for the recommendation if
the caller happens to mention them.

Note on scope vs. the original spec: `smoker` was added to the required list
even though it wasn't in the original example question list, because
smoking status materially changes premiums and eligibility for health/term
insurance - leaving it out would make the collected profile unusable for a
real recommendation. `name`, `marital_status`, `occupation`, `annual_income`,
and `preferred_insurer` were moved to "optional" rather than required,
because asking all 16 fields one by one would make for a very long,
interrogation-style call; income and marital status are useful context but
not strictly necessary to propose a plan type and ballpark coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .customer_profile import CustomerProfile


@dataclass(frozen=True)
class Question:
    """One thing the agent may need to ask about."""

    field_name: str  # must match a CustomerProfile attribute
    topic: str  # short human label, e.g. "family size"
    ask_hint: str  # guidance for the LLM on *what* to ask - not a literal script


REQUIRED_QUESTIONS: List[Question] = [
    Question("name", "name", "Ask the caller for their name if not already provided."),
    Question("age", "age", "Ask their exact age in years - needed to check policy eligibility."),
    Question("family_members", "plan type and family size", "Ask whether they are looking for an individual plan or a family floater plan. If they say a family floater plan, ask how many family members they would like to cover."),
    Question("smoker", "smoking / tobacco habit", "Ask whether they smoke or use tobacco - it affects premium and eligibility."),
    Question("budget", "budget", "Ask what annual premium budget they have in mind."),
    Question("coverage_required", "coverage amount", "Ask what sum insured or coverage amount they're looking for, if they have a figure in mind."),
]

OPTIONAL_QUESTIONS: List[Question] = [
    Question("marital_status", "marital status", "If it hasn't come up naturally, ask if they're single, married, divorced, or widowed."),
    Question("occupation", "occupation", "If relevant and not already known, ask what they do for a living."),
    Question("annual_income", "annual income", "If it hasn't come up, ask their approximate annual income - useful for suggesting adequate coverage."),
    Question("preferred_insurer", "preferred insurer", "Ask if they have a preferred insurance company, if any."),
]


def get_next_required_question(profile: CustomerProfile) -> Optional[Question]:
    """The first required question whose field is still unanswered, or None if done."""
    for question in REQUIRED_QUESTIONS:
        if not profile.is_filled(question.field_name):
            return question
    return None


def get_next_optional_question(profile: CustomerProfile) -> Optional[Question]:
    """The first optional question whose field is still unanswered, or None if done."""
    for question in OPTIONAL_QUESTIONS:
        if not profile.is_filled(question.field_name):
            return question
    return None


def missing_required_fields(profile: CustomerProfile) -> List[str]:
    """Human-readable topics for every required field not yet answered."""
    return [q.topic for q in REQUIRED_QUESTIONS if not profile.is_filled(q.field_name)]


def is_required_complete(profile: CustomerProfile) -> bool:
    """True once every required field has an answer."""
    return all(profile.is_filled(q.field_name) for q in REQUIRED_QUESTIONS)
