"""
metadata_extractor.py

Implements structured insurance policy metadata extraction from policy text
using the Google Gemini API and Pydantic validation.
"""

import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from ingestion.models import PolicyMetadata

# Configure Google Gemini API key and model
try:
    from core import config
    api_key = config.GOOGLE_API_KEY
    model_name = config.GEMINI_MODEL

except ImportError:
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

client = None
if api_key:
    client = genai.Client(api_key=api_key)


from typing import Optional, Tuple

# Core keywords commonly found in insurance policy documents
INSURANCE_KEYWORDS = {
    "policy", "insurance", "insurer", "insured", "sum insured", "premium", 
    "mediclaim", "floater", "deductible", "co-payment", "claim", "waiting period", 
    "coverage", "benefit", "policyholder", "schedule", "endorsement"
}

def is_likely_insurance_text(text: str) -> Tuple[bool, str]:
    """
    Performs a fast, local keyword pre-check to verify if the text resembles
    an insurance policy document before invoking the Gemini API.

    Returns:
        A tuple of (is_valid, reason_string).
    """
    if not text or len(text.strip()) < 50:
        return False, "Document text is too short or empty to be a valid insurance policy."
        
    text_lower = text.lower()
    matches = [kw for kw in INSURANCE_KEYWORDS if kw in text_lower]
    
    # Require at least 2 distinct insurance domain terms
    if len(matches) < 2:
        return False, f"Document text lacks essential insurance keywords (matched terms: {matches})."
        
    return True, f"Found insurance keywords: {matches[:5]}"


class PolicyExtractionSchema(BaseModel):
    """Schema defining classification and structured metadata fields to extract from a document."""
    
    is_insurance_policy: bool = Field(
        ...,
        description="Set to True ONLY if the document is an official health, life, or motor insurance policy/proposal document. Set to False for invoices, receipts, resumes, general articles, user manuals, or non-insurance documents."
    )
    document_type: str = Field(
        ...,
        description="Detected document type (e.g., 'health_insurance_policy', 'life_insurance_policy', 'invoice', 'resume', 'unknown')."
    )
    rejection_reason: Optional[str] = Field(
        None,
        description="If is_insurance_policy is False, provide a brief explanation why (e.g., 'Document is a commercial invoice, not an insurance policy')."
    )
    policy_id: Optional[str] = Field(
        None,
        description="Unique policy number / identifier as stated in the policy schedule (e.g. ACH/EL/2026/00721904, TSG/HP/2026/00562481). Extract strictly from the document. Do not invent fake IDs."
    )
    policy_code: Optional[str] = Field(
        None,
        description="Product or policy code stated in the document (for example AC/EHP/2026/D2 or SFHS-2026). Keep it distinct from the policy number and do not invent one."
    )
    policy_name: Optional[str] = Field(
        None,
        description="Exact marketing/commercial name of the insurance policy as stated in the text (e.g. Star Health Assure Individual, ApexCare Elevate Health Plan)."
    )
    insurer: Optional[str] = Field(
        None,
        description="Exact insurance company providing the policy as stated in the text (e.g. Star Health, Care Health, HDFC Ergo, SecureLife Insurance, ApexCare Health Insurance)."
    )
    plan_type: Optional[str] = Field(
        "Individual",
        description="Type of the plan. MUST be exactly 'Individual' or 'Family Floater'. If the text mentions Single Cover / Individual, use 'Individual'. If it mentions Family, Floater, or covers dependents/children/spouse, use 'Family Floater'."
    )
    premium: Optional[float] = Field(
        0.0,
        description="Total annual premium payable in INR (₹) as stated in the policy schedule/table (including base premium and taxes/GST). Do NOT extract partial base premiums, rider add-ons separately, or installment amounts if the total payable premium is present."
    )
    min_age: Optional[int] = Field(
        18,
        description="Minimum entry age required to buy this policy in years. Extract only the numeric value."
    )
    max_age: Optional[int] = Field(
        65,
        description="Maximum ENTRY age allowed to buy/purchase this policy in years (e.g., 65, 69, 75). Extract the specific Maximum ENTRY Age integer as stated in the eligibility table/text. Do NOT confuse Maximum ENTRY Age with Maximum Renewal Age / Lifelong Renewability. Set max_age to 99 ONLY if there is no maximum ENTRY age limit specified."
    )
    sum_insured: Optional[float] = Field(
        500000.0,
        description="Primary total Sum Insured / Coverage limit in INR (₹) as explicitly stated in the policy schedule or benefit summary. Do NOT extract cumulative bonus limits, restore benefit limits, or total pool limits instead of the primary Sum Insured."
    )
    smoker_allowed: bool = Field(
        True,
        description="Whether tobacco users/smokers are allowed to buy this policy. Set to True unless the text explicitly states smokers are prohibited."
    )
    covers_diabetes: bool = Field(
        True,
        description="Whether pre-existing diabetes or blood sugar conditions are covered under the policy (either immediately OR after a standard waiting period / Pre-existing Disease clause). Set to True unless diabetes is explicitly listed as a permanent exclusion."
    )
    covers_hypertension: bool = Field(
        True,
        description="Whether pre-existing hypertension or high BP conditions are covered under the policy (either immediately OR after a standard waiting period / Pre-existing Disease clause). Set to True unless hypertension is explicitly listed as a permanent exclusion."
    )
    parents_allowed: bool = Field(
        False,
        description="Whether parents or parents-in-law can be included in this policy. Set to True if explicitly allowed or included. Set to False if excluded or single-individual."
    )
    children_allowed: bool = Field(
        False,
        description="Whether dependent children can be included in this policy. Set to True if explicitly allowed or included. Set to False if excluded or single-individual."
    )


def parse_metadata_from_text(text: str) -> PolicyMetadata:
    """
    Parses and extracts metadata required by the Recommendation Engine from policy text
    using Google Gemini API with structured document validation and strict grounding.

    Args:
        text: Complete policy document text.

    Returns:
        A PolicyMetadata instance populated with the extracted values.

    Raises:
        ValueError: If document fails local keyword validation, LLM classification,
                   or missing required policy metadata fields.
    """
    if not api_key or not client:
        raise ValueError(
            "GOOGLE_API_KEY is not configured. Please set the GOOGLE_API_KEY environment variable "
            "or populate it in your .env file."
        )

    # Layer 1: Fast local keyword pre-check
    is_valid_kw, kw_reason = is_likely_insurance_text(text)
    if not is_valid_kw:
        raise ValueError(f"Document rejected (Layer 1 Keyword Check): {kw_reason}")

    # Layer 2: Combined classification and extraction via Gemini API with strict anti-hallucination prompt
    prompt = f"""
    You are an expert insurance document analyzer and classifier.
    
    STRICT GROUNDING & ANTI-HALLUCINATION RULES:
    1. Base all extracted fields strictly on the provided text below.
    2. Do NOT invent, assume, or fabricate any policy details, policy names, policy IDs, policy/product codes, or premiums from external knowledge.
    3. If the document is NOT an insurance policy (e.g. invoice, receipt, resume, user manual), set is_insurance_policy to false.
    4. Pay close attention to numerical values:
       - TOTAL PREMIUM: Extract the final total gross premium payable including taxes/GST.
       - SUM INSURED: Extract the primary base Sum Insured (ignore restore benefits or bonus caps).
       - MAXIMUM ENTRY AGE: Extract the exact Maximum ENTRY Age (e.g. 69 years, 65 years) allowed to buy/enter the policy. Do NOT confuse Maximum Entry Age with Maximum Renewal Age or Lifelong Renewability. Set max_age to 99 ONLY if there is no upper entry age limit.
       - PRE-EXISTING CONDITIONS (Diabetes / Hypertension): Set covers_diabetes and covers_hypertension to true if covered (including after waiting periods). Set to false ONLY if explicitly listed as permanent exclusions.
    
    Policy Document Text:
    ---
    {text}
    ---
    """


    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PolicyExtractionSchema,
            temperature=0.0
        )
    )
    
    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse structured JSON from Gemini output: {response.text}") from e
    
    # Layer 2 Check: Verify Gemini document classification
    is_insurance = bool(data.get("is_insurance_policy", False))
    if not is_insurance:
        reason = data.get("rejection_reason") or f"Document classified as '{data.get('document_type', 'non-insurance')}'."
        raise ValueError(f"Document rejected (Layer 2 LLM Guard): {reason}")

    # Layer 3 Check: Verify presence of critical mandatory fields
    policy_name = data.get("policy_name")
    insurer = data.get("insurer")
    policy_id = data.get("policy_id") or "POL_GEN_01"

    if not policy_name or not insurer or policy_name.strip().lower() in ["none", "unknown", "n/a"]:
        raise ValueError("Document rejected (Layer 3 Sanity Check): Missing valid policy name or insurer details.")

    # Map raw dictionary to our PolicyMetadata dataclass
    return PolicyMetadata(
        policy_id=str(policy_id),
        policy_name=str(policy_name),
        insurer=str(insurer),
        plan_type=str(data.get("plan_type") or "Individual"),
        premium=float(data.get("premium") or 0.0),
        min_age=int(data.get("min_age") or 18),
        max_age=int(data.get("max_age") or 65),
        sum_insured=float(data.get("sum_insured") or 500000.0),
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
