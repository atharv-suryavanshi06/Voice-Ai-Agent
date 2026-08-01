"""
models.py

Defines the PolicyMetadata data model which holds all the extracted fields
from the policy document.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class PolicyMetadata:
    """Represents the extracted metadata for an insurance policy."""
    policy_id: str
    policy_name: str
    insurer: str
    plan_type: str  # "Individual" or "Family Floater"
    premium: float  # Annual premium in INR
    min_age: int    # Minimum age of entry (years)
    max_age: int    # Maximum age of entry (years)
    sum_insured: float  # Total coverage limit in INR
    smoker_allowed: bool
    covers_diabetes: bool
    covers_hypertension: bool
    parents_allowed: bool
    children_allowed: bool
    policy_code: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyMetadata":
        """Instantiates a PolicyMetadata object from a dictionary."""
        return cls(
            policy_id=str(data["policy_id"]),
            policy_name=str(data["policy_name"]),
            insurer=str(data["insurer"]),
            plan_type=str(data["plan_type"]),
            premium=float(data.get("premium") or 0.0),
            min_age=int(data.get("min_age") or 18),
            max_age=int(data["max_age"]) if data.get("max_age") is not None else 99,
            sum_insured=float(data.get("sum_insured") or 0.0),
            smoker_allowed=bool(data.get("smoker_allowed", True)),
            covers_diabetes=bool(data.get("covers_diabetes", False)),
            covers_hypertension=bool(data.get("covers_hypertension", False)),
            parents_allowed=bool(data.get("parents_allowed", False)),
            children_allowed=bool(data.get("children_allowed", False)),
            policy_code=(
                str(data["policy_code"]).strip()
                if data.get("policy_code") is not None
                else None
            ),
        )


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the PolicyMetadata object to a dictionary."""
        return asdict(self)
