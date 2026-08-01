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

AGENT_NAME = "Riya"
COMPANY_DESCRIPTION = "an insurance marketplace that compares policies from multiple insurers"


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


def _recommending_policy_instructions(recommendations: Optional[List[Any]] = None) -> str:
    if not recommendations:
        return (
            "All questions have been answered and you now have all customer details.\n"
            "CRITICAL DIRECTIVE: You MUST directly state and suggest policy recommendations right NOW in this turn!\n"
            "Do NOT ask the customer if they want recommendations, do NOT say 'I will look up policies', and do NOT tell them you will get back to them later.\n"
            "In 2-3 spoken sentences, briefly summarize their needs and directly suggest the general type of plan that fits them "
            "(for example a family floater health plan or an individual plan) along with a ballpark coverage amount and estimated premium range."
        )
    
    policy_lines = []
    for i, p in enumerate(recommendations, 1):
        policy_lines.append(
            f"{i}. {p.policy_name} by {p.insurer}: Annual premium {format_indian_currency_for_speech(p.premium)}, "
            f"sum insured {format_indian_currency_for_speech(p.sum_insured)}. "
            f"Diabetes Covered: {'Yes' if p.covers_diabetes else 'No'}, "
            f"Hypertension Covered: {'Yes' if p.covers_hypertension else 'No'}."
        )
    policy_list_str = "\n".join(policy_lines)

    return (
        "CRITICAL DIRECTIVE: All customer information has been collected. You MUST directly and immediately suggest the policy options below right NOW in this response.\n"
        "STRICTLY FORBIDDEN:\n"
        "- Do NOT ask 'Would you like me to recommend policies?' or 'Do you want me to suggest policies?'\n"
        "- Do NOT say 'I will look up the policies and let you know' or 'I will check policies for you'.\n"
        "- Do NOT ask any more profile questions or stall.\n\n"
        "WHAT YOU MUST DO RIGHT NOW:\n"
        "Summarize their profile details and directly state the specific recommended policies from our catalog:\n"
        f"{policy_list_str}\n\n"
        "CRITICAL EMAIL DIRECTIVE:\n"
        "- Email delivery is available only after the caller provides and confirms an address.\n"
        "- Do not claim delivery until the delivery state below says it succeeded.\n"
        "- Ask the caller for their email address so the system can attempt to send the complete policy document: "
        "'Please tell me your email address so I can send you the complete policy details and document.'"
    )




def _email_delivery_instructions(email_state: str) -> str:
    messages = {
        "sent": "Email delivery succeeded. You may truthfully say the policy details were sent.",
        "failed": "Email delivery failed. Say you were unable to send it and offer a polite retry; do not expose technical details.",
        "disabled": "Email delivery is unavailable because the service is not configured. Say you cannot send it right now; do not claim success.",
        "pending": "Email delivery has not completed. Do not claim that it was sent.",
        "invalid": "There is no valid confirmed email address. Do not claim that anything was sent.",
        "not_requested": "The caller did not request or confirm email delivery. Do not mention that an email was sent.",
    }
    return f"EMAIL DELIVERY STATE: {email_state}. {messages.get(email_state, messages['not_requested'])}"


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
) -> str:
    """Compose the full system prompt for the current turn."""
    sections = [_persona(), _ground_rules(), _profile_summary_block(profile)]

    if profile.pending_email and not profile.email_confirmed:
        sections.append(_email_verification_instructions(profile.pending_email))

    if state in (ConversationState.COLLECTING_INFORMATION, ConversationState.ANSWERING_POLICY_QUESTIONS):
        sections.append(_missing_info_block(missing_fields))

    if policy_selection_required:
        names = ", ".join(str(policy.policy_name) for policy in (recommended_policies or []))
        sections.append(
            "The caller asked a policy-specific question but has not identified one of several recommendations. "
            f"Ask one brief clarification question naming the available policies: {names}. "
            "Do not guess, retrieve from another policy, or answer the detail yet."
        )
    elif retry_question is not None:
        sections.append(_repeat_question_instructions(retry_question))
    elif state == ConversationState.RECOMMENDING_POLICY:
        sections.append(_recommending_policy_instructions(recommendations))
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
