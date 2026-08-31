"""
prompts.py

Builds the system prompt handed to Gemini for each turn. Nothing here is one
big hardcoded string - small functions each contribute one section (persona,
ground rules, known profile, missing info, state-specific instructions), and
`build_system_prompt()` assembles the ones relevant to the current state.
This is what lets the Conversation Manager change what the LLM is told to
do, turn by turn, without the LLM ever having to decide the flow itself.
"""

from __future__ import annotations

from typing import List, Optional, Any

from .customer_profile import CustomerProfile
from .question_flow import Question
from .state import ConversationState
from rag.models import RetrievedChunk
from rag.grounding import GROUNDING_RULES, INSUFFICIENT_EVIDENCE_RESPONSE, format_retrieved_context
from .amounts import format_indian_currency_for_speech
from recommendation.policy_identity import policy_display_labels

AGENT_NAME = "Riya"
COMPANY_DESCRIPTION = "an insurance marketplace that compares policies from multiple insurers"
POLICY_SERVICE_UNAVAILABLE_RESPONSE = (
    "I'm sorry, I can't access the policy information right now. Please try again shortly."
)


def _persona() -> str:
    return (
        f"You are {AGENT_NAME}, a warm, professional insurance advisor calling on behalf of "
        f"{COMPANY_DESCRIPTION}. This is a live phone call, so keep every reply short and "
        "conversational - the way a helpful human advisor would speak, not like a written "
        "document. Never use bullet points, headers, or markdown; you are being heard, not read."
    )


def _ground_rules() -> str:
    return (
        "Ground rules for every reply:\n"
        "- Ask at most ONE question per reply.\n"
        "- Follow the required question order strictly: Age -> Plan Type (Individual vs Family Floater) -> Smoking Habit -> Budget -> Coverage Amount.\n"
        "- For Question 2 (Plan Type), ask explicitly: 'Are you looking for an individual plan or a family floater plan?'\n"
        "- Never skip any required question. If a question was skipped or un-answered earlier, ask it now before proposing policies.\n"
        "- Never ask about anything already listed under 'Known so far' below.\n"
        "- EMAIL CONFIRMATION DIRECTIVE: Never state 'I have sent the email' or 'I will ensure details are sent immediately' until the user explicitly confirms the spelled-out email address. If an email address is provided or updated, you MUST spell it out letter-by-letter and ask for explicit confirmation first.\n"
        "- Stay strictly within insurance and this call; politely decline unrelated requests.\n"
        "- Keep replies short: 1-3 sentences, suitable for being spoken aloud.\n"
        "- When all profile information is gathered, directly present policy recommendations right NOW. Do NOT ask permission to recommend or say 'I will look up policies' / 'I will let you know'."
    )




def _profile_summary_block(profile: CustomerProfile) -> str:
    known = profile.to_summary_dict()
    if not known:
        return "Known so far: nothing yet - this is the start of the conversation."
    lines = "; ".join(f"{label}: {value}" for label, value in known.items())
    return f"Known so far: {lines}."


def _missing_info_block(missing_fields: List[str]) -> str:
    if not missing_fields:
        return "Missing information: none - all required information has been collected."
    return f"Required information still missing (MUST be asked in order): {', '.join(missing_fields)}."


def _greeting_instructions() -> str:
    return (
        "You are at the very start of the call. Greet the caller warmly, introduce yourself "
        "and the company in one short sentence, and ask for their name if you don't already "
        "know it. Do not ask any other questions yet."
    )


def _collecting_information_instructions(next_question: Optional[Question]) -> str:
    if next_question is None:
        return "All required information has been collected - do not ask any more questions."

    if next_question.field_name == "family_members":
        return (
            "STRICT QUESTION FLOW DIRECTIVE FOR QUESTION 2:\n"
            "- You MUST ask the caller explicitly: 'Are you looking for an individual plan or a family floater plan?'\n"
            "- Do NOT ask how many family members to cover yet. First ask if they want an individual plan or a family floater plan.\n"
            "- Ask ONLY this one question in your reply."
        )

    return (
        f"STRICT QUESTION FLOW DIRECTIVE:\n"
        f"- You MUST ask the required questions in strict order: Age -> Plan Type (Individual vs Family Floater) -> Smoking/Tobacco Habit -> Budget -> Coverage Amount.\n"
        f"- The current required question to ask right NOW is: {next_question.ask_hint}\n"
        f"- Do NOT skip this question. If any previous question was missed or un-answered, you MUST ask it now.\n"
        f"- Ask about ONLY this one thing in your next reply. Do NOT suggest policies or pricing until ALL required questions are answered."
    )


def _repeat_question_instructions(question: Question) -> str:
    if question.field_name == "name":
        ask = "May I have your name, please?"
    elif question.field_name == "family_members":
        ask = "Are you looking for an individual plan or a family floater plan?"
    else:
        ask = question.ask_hint.replace("Ask ", "Could you please tell me ", 1).rstrip(".") + "?"
    return (
        "The caller's last answer did not provide a usable value for the current required field. "
        f"Start your reply with: 'Sorry, I didn't catch that.' Then ask exactly this one question: '{ask}' "
        "Do not infer a value, move to another question, recommend a policy, or mention pricing."
    )


def _answering_policy_question_instructions(
    next_question: Optional[Question],
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
) -> str:
    instructions = (
        "The caller just asked a question about insurance. Answer it directly, accurately, "
        "and briefly, in plain conversational language, using ONLY the provided retrieved policy context below.\n\n"
        f"Ground Rules:\n- {GROUNDING_RULES}\n"
        "- Special Exception for Email Requests: If the caller is asking to send or receive policy documents or quotes via email (e.g. 'can you email me?'), confirm warmly: 'Yes, absolutely! Please tell me your email address so I can send the complete policy document to you right away.'\n"
        f"- If evidence is insufficient, respond EXACTLY with: '{INSUFFICIENT_EVIDENCE_RESPONSE}'"
    )

    
    if retrieved_chunks:
        context_text = format_retrieved_context(retrieved_chunks)
        
        instructions += (
            f"\n\nRetrieved Policy Context:\n"
            f"=========================================\n"
            f"{context_text}\n"
            f"=========================================\n"
        )
    else:
        instructions += f"\n\nNo accepted policy evidence is available. Respond EXACTLY with: '{INSUFFICIENT_EVIDENCE_RESPONSE}'"

    if next_question is not None:
        instructions += (
            f"\nAfter answering, smoothly bring the conversation back to their "
            f"{next_question.topic}, since that's the next thing you still need to know."
        )
    return instructions


def _policy_service_unavailable_instructions(next_question: Optional[Question]) -> str:
    instructions = (
        "The policy retrieval service failed temporarily. Do not say that the policy information "
        "is unknown or unavailable in the policy document, and do not guess an answer. "
        f"Respond EXACTLY with: '{POLICY_SERVICE_UNAVAILABLE_RESPONSE}'"
    )
    if next_question is not None:
        instructions += (
            f" After that, return to the caller's {next_question.topic}, which is the next "
            "required detail."
        )
    return instructions


def _recommending_policy_instructions(
    profile: Optional[CustomerProfile] = None,
    recommendations: Optional[List[Any]] = None,
    duplicate_policy_names: Optional[List[str]] = None,
) -> str:
    name_part = profile.name if (profile and profile.name) else "Valued Customer"
    age_part = f"{profile.age} years old" if (profile and profile.age is not None) else "your age"
    
    is_family = False
    if profile:
        is_family = (
            (profile.family_members is not None and profile.family_members > 1)
            or profile.parents_included is True
            or profile.children_included is True
        )
    plan_type_part = "a family floater plan" if is_family else "an individual plan"
    
    budget_part = (
        format_indian_currency_for_speech(profile.budget)
        if (profile and profile.budget is not None)
        else "your budget"
    )
    coverage_part = (
        format_indian_currency_for_speech(profile.coverage_required)
        if (profile and profile.coverage_required is not None)
        else "your desired coverage"
    )

    recap_example = (
        f"{name_part}, you are {age_part} looking for {plan_type_part} with a budget of "
        f"{budget_part} and looking for {coverage_part} coverage."
    )

    if not recommendations:
        return (
            "CRITICAL DIRECTIVE - NO MATCHING POLICY IN CATALOG:\n"
            f"All customer details have been gathered. You MUST start your response by repeating all user details in this exact sentence structure:\n"
            f"\"{recap_example}\"\n\n"
            "HOWEVER, no matching insurance policy for their specific age, plan type, or criteria was found in our policy catalog.\n"
            "STRICTLY FORBIDDEN: Do NOT hallucinate, invent, or suggest any fake insurance policies, ballpark prices, or external plans.\n"
            "WHAT YOU MUST SAY:\n"
            "After repeating their details in the exact recap sentence above, directly and politely inform the caller: "
            "'However, an insurance policy matching your age and requirements is not currently available in our policy catalog.'\n"
            "Keep the response short, warm, and conversational in 2-3 sentences."
        )
    
    policy_lines = []
    labels = policy_display_labels(recommendations, duplicate_policy_names or [])
    for i, (p, policy_label) in enumerate(zip(recommendations, labels), 1):
        policy_lines.append(
            f"{i}. {policy_label} by {p.insurer}: Annual premium {format_indian_currency_for_speech(p.premium)}, "
            f"sum insured {format_indian_currency_for_speech(p.sum_insured)}. "
            f"Diabetes Covered: {'Yes' if p.covers_diabetes else 'No'}, "
            f"Hypertension Covered: {'Yes' if p.covers_hypertension else 'No'}."
        )
    policy_list_str = "\n".join(policy_lines)

    return (
        "CRITICAL DIRECTIVE - POLICY RECOMMENDATIONS & MANDATORY RECAP:\n"
        "All customer details have been gathered. You MUST start your response by repeating all user details in this exact sentence structure:\n"
        f"\"{recap_example} Here are some options:\"\n\n"
        "STRICTLY FORBIDDEN:\n"
        "- Do NOT ask 'Would you like me to recommend policies?' or stall.\n"
        "- Do NOT say 'I will look up policies' or 'I will check policies for you'.\n"
        "- Do NOT omit the opening recap sentence repeating their details.\n\n"
        "WHAT YOU MUST DO RIGHT NOW:\n"
        "1. Open with the exact profile recap sentence above.\n"
        "2. Directly state the specific recommended policies from our catalog:\n"
        f"{policy_list_str}\n\n"
        "CRITICAL EMAIL DIRECTIVE:\n"
        "- Email delivery is available only after the caller provides and confirms an address.\n"
        "- Ask the caller for their email address so the system can attempt to send the complete policy document: "
        "'Please tell me your email address so I can send you the complete policy details and document.'"
    )


def _email_delivery_instructions(email_state: str) -> str:
    messages = {
        "sent": "VERIFIED DELIVERY SUCCESS: email delivery succeeded because the SMTP send operation returned success. You may say the policy details were sent.",
        "failed": "VERIFIED DELIVERY FAILURE: email delivery failed after all three SMTP attempts. Say you could not send the document, offer to retry using the confirmed address, and do not say or imply it was sent.",
        "disabled": "VERIFIED DELIVERY UNAVAILABLE: the email service is not configured. Say you cannot send it right now; never claim success.",
        "pending": "VERIFIED DELIVERY PENDING: the send operation has not returned success. Never say, imply, or confirm that an email was sent.",
        "invalid": "VERIFIED DELIVERY BLOCKED: there is no valid confirmed email address. Do not claim that anything was sent.",
        "not_requested": "VERIFIED DELIVERY NOT REQUESTED: do not mention that an email was sent.",
    }
    return (
        f"EMAIL DELIVERY STATE: {email_state}. {messages.get(email_state, messages['not_requested'])} "
        "The exact phrases 'I have sent the email', 'I sent the email', and 'the details were sent' "
        "are forbidden unless the state is sent."
    )


def _ending_call_instructions(email_state: str = "not_requested") -> str:
    return (
        "Wrap up the call warmly and thank the caller for their time. "
        f"{_email_delivery_instructions(email_state)} "
        "Say a polite goodbye in one or two sentences."
    )



_STATE_INSTRUCTION_BUILDERS = {
    ConversationState.GREETING: lambda next_question: _greeting_instructions(),
    ConversationState.COLLECTING_INFORMATION: _collecting_information_instructions,
    ConversationState.ANSWERING_POLICY_QUESTIONS: _answering_policy_question_instructions,
    ConversationState.RECOMMENDING_POLICY: lambda next_question: _recommending_policy_instructions(),
    ConversationState.ENDING_CALL: lambda next_question: _ending_call_instructions(),
}


from .customer_profile import CustomerProfile, spell_out_email


def _email_verification_instructions(pending_email: str) -> str:
    spelled = spell_out_email(pending_email)
    return (
        "CRITICAL EMAIL VERIFICATION DIRECTIVE (HIGHEST PRIORITY OVERRIDE):\n"
        f"- The candidate email address to verify is: '{pending_email}'.\n"
        f"- You MUST spell out this email address letter-by-letter in your response: '{spelled}'.\n"
        f"- Ask the caller explicitly for confirmation: 'Did I get your email right as {spelled} so I can send you the policy documents?'\n"
        "- STRICTLY FORBIDDEN: Do NOT state 'I have sent the email' or 'I will ensure details are sent immediately'. You MUST ask for their confirmation first!\n"
        f"- If the caller just corrected a wrong email address, acknowledge politely, spell out this new address ('{spelled}'), and ask if this new one is correct."
    )



def build_system_prompt(
    state: ConversationState,
    profile: CustomerProfile,
    next_question: Optional[Question],
    missing_fields: List[str],
    recommendations: Optional[List[Any]] = None,
    retrieved_chunks: Optional[List[RetrievedChunk]] = None,
    email_state: str = "not_requested",
    policy_selection_required: bool = False,
    recommended_policies: Optional[List[Any]] = None,
    retry_question: Optional[Question] = None,
    retrieval_error: bool = False,
    duplicate_policy_names: Optional[List[str]] = None,
) -> str:
    """Compose the full system prompt for the current turn."""
    sections = [_persona(), _ground_rules(), _profile_summary_block(profile)]

    if profile.pending_email and not profile.email_confirmed:
        sections.append(_email_verification_instructions(profile.pending_email))

    if state in (ConversationState.COLLECTING_INFORMATION, ConversationState.ANSWERING_POLICY_QUESTIONS):
        sections.append(_missing_info_block(missing_fields))

    if policy_selection_required:
        names = ", ".join(policy_display_labels(recommended_policies or []))
        sections.append(
            "The caller asked a policy-specific question but has not identified one of several recommendations. "
            f"Ask one brief clarification question naming the available policies: {names}. "
            "When two options share a name, say each option's code and policy number exactly as shown. "
            "Do not guess, retrieve from another policy, or answer the detail yet."
        )
    elif retrieval_error and state == ConversationState.ANSWERING_POLICY_QUESTIONS:
        sections.append(_policy_service_unavailable_instructions(next_question))
    elif retry_question is not None:
        sections.append(_repeat_question_instructions(retry_question))
    elif state == ConversationState.RECOMMENDING_POLICY:
        sections.append(
            _recommending_policy_instructions(profile, recommendations, duplicate_policy_names)
        )
    else:
        instruction_builder = _STATE_INSTRUCTION_BUILDERS[state]
        if state == ConversationState.ENDING_CALL:
            sections.append(_ending_call_instructions(email_state))
        elif state == ConversationState.ANSWERING_POLICY_QUESTIONS:
            sections.append(instruction_builder(next_question, retrieved_chunks))
        else:
            sections.append(instruction_builder(next_question))

    if state != ConversationState.ENDING_CALL:
        sections.append(_email_delivery_instructions(email_state))

    return "\n\n".join(sections)
