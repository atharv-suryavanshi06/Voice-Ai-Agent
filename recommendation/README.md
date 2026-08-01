# Recommendation Engine Module

This folder contains a simple, modular, rule-based Recommendation Engine for the Insurance Voice AI Agent. It filters insurance policies from a JSON catalog based on hard customer constraints (demographics, exclusions) and ranks the surviving policies using a heuristic scoring formula.

---

## Architecture & Design Decisions

### 1. Zero External Dependencies (`dataclasses`)
Instead of using heavy NLU frameworks or libraries like Pydantic, the module uses Python's standard library `dataclasses` module. This keeps the execution path fast, lightweight, and perfectly compatible with the existing `CustomerProfile` dataclass.

### 2. File Structure
The module is self-contained under the `recommendation/` directory:
- [__init__.py](file:///d:/Voice-Ai-Agent/recommendation/__init__.py): Exposes the package's public API.
- [policy.py](file:///d:/Voice-Ai-Agent/recommendation/policy.py): Contains the `Policy` dataclass and deserialization helper.
- [policy_catalog.json](file:///d:/Voice-Ai-Agent/recommendation/policy_catalog.json): Stores a list of static policy profiles.
- [recommendation_engine.py](file:///d:/Voice-Ai-Agent/recommendation/recommendation_engine.py): Implements the filtering and ranking engine.
- [README.md](file:///d:/Voice-Ai-Agent/recommendation/README.md): This documentation file.

---

## 1. The `Policy` Data Model

The `Policy` model contains the following parameters:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `policy_id` | `str` | A unique identifier (e.g., `POL_IND_01`). |
| `policy_name` | `str` | Public name of the policy. |
| `insurer` | `str` | The insurance provider (e.g., "Star Health"). |
| `plan_type` | `str` | Type of plan: `"Individual"` or `"Family Floater"`. |
| `premium` | `float` | The annual premium amount in INR (₹). |
| `min_age` | `int` | Minimum age required to buy this policy (inclusive). |
| `max_age` | `int` | Maximum age allowed to buy this policy (inclusive). |
| `sum_insured` | `float` | The maximum coverage/payout amount in INR (₹). |
| `smoker_allowed` | `bool` | Whether the policy accepts smokers. |
| `covers_diabetes` | `bool` | Whether the policy covers diabetic customers. |
| `covers_hypertension` | `bool` | Whether the policy covers customers with high blood pressure / hypertension. |
| `parents_allowed` | `bool` | Whether parents can be included in the coverage. |
| `children_allowed` | `bool` | Whether children can be included in the coverage. |

---

## 2. Business Logic: Strict Filtering

Before any scoring occurs, the engine performs a pass to discard ineligible policies. A policy is rejected if:

1. **Age Incompatibility**: Customer's age is less than `min_age` or greater than `max_age`.
2. **Smoker Exclusion**: Customer is a smoker (`smoker == True`), but the policy does not support smokers (`smoker_allowed == False`).
3. **Medical Exclusions**:
   - Customer has a pre-existing diabetic condition (scans for `"diabet"` or `"sugar"` in `existing_diseases`), but `covers_diabetes` is `False`.
   - Customer has a hypertensive/BP condition (scans for `"hypertens"`, `"bp"`, `"blood pressure"`, or `"tension"`), but `covers_hypertension` is `False`.
4. **Family Inclusions**:
   - Customer explicitly wants to include parents (`parents_included == True`), but `parents_allowed` is `False`.
   - Customer explicitly wants to include children (`children_included == True`), but `children_allowed` is `False`.
5. **Budget Check (Strict)**: The policy premium exceeds the customer's budget limit (skipped if `strict == False` during fallback).
6. **Coverage Check (Strict)**: The policy sum insured is less than the customer's requested coverage limit (skipped if `strict == False` during fallback).

> [!NOTE]
> If strict filtering results in 0 matches (e.g., customer budget is too low for any available policy), the engine will automatically run a fallback pass with `strict=False` to relax the budget and coverage required checks, ensuring some options can still be recommended.

---

## 3. Scoring & Ranking Heuristic

For all policies that pass the filters, a match score is calculated. The score starts at **100.0**, and the following bonuses/penalties are applied:

### A. Budget Savings Bonus (Up to +30 points)
Users prefer saving money. If their budget is specified and greater than the premium:
$$\text{Bonus} = \left( \frac{\text{Budget} - \text{Premium}}{\text{Budget}} \right) \times 30$$

### B. Coverage Exceedance Bonus (Up to +20 points)
Users prefer more coverage value for their money. If coverage required is specified:
$$\text{Bonus} = \min\left(\left( \frac{\text{Sum Insured}}{\text{Coverage Required}} - 1.0 \right) \times 10, \, 20.0\right)$$
*(This applies only when Sum Insured $\ge$ Coverage Required).*

### C. Preferred Insurer Bonus (+25 points)
If the customer has a preferred insurer, and it matches (case-insensitive substring match) the policy's insurer, a flat **+25 points** bonus is awarded.

### D. Family / Plan Type Fit (Bonus +20 / Penalty -10)
- **Family Coverage Needed** (defined as `family_members > 1` or `parents_included == True` or `children_included == True`):
  - **+20 points** if the policy is a `"Family Floater"`.
  - **-10 points** if the policy is `"Individual"`.
- **Individual Coverage Needed** (defined as single customer with no family/parents/children):
  - **+20 points** if the policy is `"Individual"`.
  - **-10 points** if the policy is a `"Family Floater"`.

---

## 4. Usage Example

Here is how you can use the module within your application:

```python
from conversation.customer_profile import CustomerProfile
from recommendation import RecommendationEngine

# 1. Initialize the engine (automatically loads policy_catalog.json)
engine = RecommendationEngine()

# 2. Setup a customer profile
profile = CustomerProfile(
    name="Arjun",
    age=35,
    family_members=3,
    children_included=True,
    existing_diseases=["diabetes mellitus"],
    budget=20000.0,
    coverage_required=800000.0,
    preferred_insurer="Star Health"
)

# 3. Get top 3 recommendations
recommendations = engine.recommend(profile, limit=3)

# 4. Display results
for i, policy in enumerate(recommendations, 1):
    print(f"{i}. {policy.policy_name} by {policy.insurer}")
    print(f"   Premium: ₹{policy.premium:,.2f} | Sum Insured: ₹{policy.sum_insured:,.2f}")
    print(f"   Plan Type: {policy.plan_type}")
```
