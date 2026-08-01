"""
conversation_manager.py

Orchestrates a single insurance-recommendation phone call: tracks state,
owns the customer profile, decides what to ask next (or whether to answer a
side question, recommend a policy, or end the call), and builds the system
prompt for each turn.

This module does NOT call Gemini and does NOT touch the existing Pipecat
pipeline (transport / STT / LLM / TTS). It only prepares what the existing
pipeline's LLM step should use for each turn: conversation history, the
customer profile, the current state, and a freshly generated system prompt.

Integration sketch (not wired in here, since the existing pipeline is
off-limits per the brief): whatever code currently receives the user's
transcribed text before it reaches GoogleLLMService would call
`process_user_message(transcript)`, then use `get_llm_messages()` to update
the LLMContext for that turn before running the LLM. After the LLM responds,
call `record_assistant_reply(response_text)` to keep history in sync.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set


from .customer_profile import CustomerProfile, _parse_email, normalize_stt_aliases
from .question_flow import Question, get_next_required_question, is_required_complete, missing_required_fields, REQUIRED_QUESTIONS, OPTIONAL_QUESTIONS
from .state import ConversationState, InvalidStateTransitionError, can_transition
from . import prompts
from recommendation.recommendation_engine import RecommendationEngine
from rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class EmailDeliveryState(str, Enum):
    NOT_REQUESTED = "not_requested"
    DISABLED = "disabled"
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    INVALID = "invalid"

_END_CALL_PHRASES = {
    "bye", "goodbye", "that's all", "thats all", "no more questions",
    "nothing else", "no thanks", "no thank you", "that is all",
}

_POLICY_QUESTION_KEYWORDS = {
    "policy", "premium", "claim", "cover", "coverage", "deductible",
    "co-pay", "copay", "waiting period", "network hospital", "rider",
    "sum insured", "tax benefit", "80d", "renewal", "cashless",
    "exclusion", "grace period", "maturity", "nominee", "benefit",
    "eligibility", "document", "details", "securelife", "star health",
    "apexcare", "trustshield", "rsbn", "vitalcare", "wellnest",
    "care freedom", "hdfc", "niva bupa", "icici", "lic", "aditya birla",
    "max life", "bajaj", "tata"
}

_QUESTION_STARTERS = (
    "what", "how", "why", "when", "does", "do ", "is ", "can ", "will ",
    "tell me", "explain", "details", "info", "information", "show", "describe",
    "give me", "about"
)

# These are deliberately phrase-level signals rather than a replacement for
# the existing keyword gate. They make common spoken policy questions visible
# even when Deepgram does not add a trailing question mark.
_POLICY_DETAIL_TERMS = {
    "included", "include", "limit", "limits", "hospitalisation", "hospitalization",
    "ambulance", "room rent", "treatment", "charges", "procedure", "documents",
    "dialysis", "surgery", "medical", "number", "code", "id",
}

_POLICY_REFERENCE_TERMS = ("this policy", "that policy", "the policy", "first policy", "second policy", "third policy")
_NAME_QUESTION = Question("name", "name", "Ask the caller for their name.")
_REQUIRED_FIELD_NAMES = {question.field_name for question in REQUIRED_QUESTIONS}
_EMAIL_FRAGMENT_MARKER_RE = re.compile(
    r"\b(email|e-mail|mail|at the rate|at rate|\bat\b|dot|gmail|yahoo|outlook|hotmail)\b",
    flags=re.IGNORECASE,
)
_EMAIL_DOMAIN_FRAGMENT_RE = re.compile(r"^(gmail|yahoo|outlook|hotmail)\b", flags=re.IGNORECASE)
_EMAIL_FRAGMENT_RESTART_RE = re.compile(
    r"^\s*(?:(?:my\s+)?(?:email|e-mail|mail)(?:\s+(?:address|id))?\s+is|(?:gmail|yahoo|outlook|hotmail)\s+is)\b",
    flags=re.IGNORECASE,
)

_FIELD_KEYWORDS: Dict[str, List[str]] = {
    "age": ["age", "years", "year", "old"],
    "family_members": ["family", "member", "members", "people", "persons", "floater", "individual", "single", "myself", "self"],
    "smoker": ["smoke", "smoker", "smoking", "tobacco", "cigarette"],
    "budget": ["budget", "premium", "spend", "pay", "cost", "afford"],
    "coverage_required": ["coverage", "cover", "sum insured", "lakh", "lakhs", "crore"],
    "name": ["name", "call me", "this is"],
    "email": ["email", "mail", "gmail", "yahoo", "outlook", "@", "at", "dot com", "dot"],
}



class ConversationManager:
    """Owns and drives the state for one phone call."""

    def __init__(self, session_id: Optional[str] = None, db_manager: Any = None) -> None:
        self.session_id: str = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self.state: ConversationState = ConversationState.GREETING
        self.profile: CustomerProfile = CustomerProfile()
        self.history: List[Dict[str, str]] = []
        self.pending_question: Optional[Question] = None
        self.asked_fields: Set[str] = set()
        self.recommendation_delivered: bool = False
        self.email_sent: bool = False
        self.email_state: EmailDeliveryState = EmailDeliveryState.NOT_REQUESTED
        self.last_sent_email: Optional[str] = None
        self._message_sequence: int = 0
        self._persistence_records: List[Dict[str, Any]] = []
        self._queued_message_ids: Set[str] = set()
        self._persisted_message_ids: Set[str] = set()
        self.rec_engine: RecommendationEngine = RecommendationEngine()
        self.db_manager: Any = db_manager
        self.last_recommended_policies: List[Any] = []
        self.active_policy_id: Optional[str] = None
        self.active_policy_name: Optional[str] = None
        self.policy_discussion_turns_remaining: int = 0
        self.policy_selection_required: bool = False
        self.last_unrecognized_question: Optional[Question] = None
        self._email_fragment_buffer: str = ""
        self._collecting_email_fragments: bool = False

    def save_to_db(self, db_manager: Any = None) -> bool:
        """Saves the profile and complete ordered conversation to PostgreSQL."""
        target_db = db_manager or self.db_manager
        if target_db and hasattr(target_db, "save_profile"):
            # Copy the list so a DB adapter cannot observe later in-memory
            # mutations while an upsert is being prepared.
            history = [dict(message) for message in self.history]
            return target_db.save_profile(
                self.session_id,
                self.profile.to_dict(),
                history,
            )
        return False

    def maybe_trigger_email(self, email_service: Any, db_manager: Any = None) -> bool:
        """
        Triggers email dispatch ONLY IF customer email is confirmed by the caller.
        """
        if self.profile.pending_email and not self.profile.email_confirmed:
            self.email_state = EmailDeliveryState.PENDING
            return False
        if not self.profile.email:
            self.email_state = EmailDeliveryState.NOT_REQUESTED
            return False
        if not self.profile.email_confirmed or "@" not in self.profile.email:
            self.email_state = EmailDeliveryState.INVALID
            return False

        current_email = self.profile.email.strip().lower()
        if self.last_sent_email and current_email == self.last_sent_email:
            self.email_state = EmailDeliveryState.SENT
            return True

        if not email_service or not hasattr(email_service, "send_policy_recommendation_email"):
            self.email_state = EmailDeliveryState.DISABLED
            return False
        service_configured = not hasattr(email_service, "is_configured") or email_service.is_configured()

        target_db = db_manager or self.db_manager
        recommendations = self.rec_engine.recommend(self.profile, limit=3)
        if not recommendations:
            self.email_state = EmailDeliveryState.FAILED
            return False

        self.email_state = EmailDeliveryState.PENDING
        try:
            success = email_service.send_policy_recommendation_email(
                recipient_email=self.profile.email,
                customer_name=self.profile.name,
                policies=recommendations,
                db_manager=target_db,
                session_id=self.session_id,
            )
        except Exception:
            logger.exception("Email delivery failed", extra={"session_id": self.session_id})
            success = False

        if success:
            self.email_sent = True
            self.email_state = EmailDeliveryState.SENT
            self.last_sent_email = current_email
            return True

        self.email_sent = False
        self.email_state = EmailDeliveryState.FAILED if service_configured else EmailDeliveryState.DISABLED
        return False

    async def maybe_trigger_email_async(self, email_service: Any, db_manager: Any = None) -> bool:
        """Run SMTP/database email work off the event loop while preserving its result."""
        return await asyncio.to_thread(self.maybe_trigger_email, email_service, db_manager)

    def _append_history_message(self, role: str, content: str) -> Dict[str, Any]:
        content = content.strip()
        self._message_sequence += 1
        message_id = f"{self.session_id}:{self._message_sequence}"
        self.history.append({"role": role, "content": content})
        record = {
            "message_id": message_id,
            "sequence": self._message_sequence,
            "role": role,
            "content": content,
        }
        self._persistence_records.append(record)
        return record

    def pending_persistence_messages(self) -> List[Dict[str, Any]]:
        """Return messages that are neither persisted nor currently queued."""
        return [
            dict(record)
            for record in self._persistence_records
            if record["message_id"] not in self._persisted_message_ids
            and record["message_id"] not in self._queued_message_ids
        ]

    def mark_persistence_queued(self, message_ids: List[str]) -> None:
        self._queued_message_ids.update(message_ids)

    def mark_persistence_complete(self, message_ids: List[str], success: bool) -> None:
        self._queued_message_ids.difference_update(message_ids)
        if success:
            self._persisted_message_ids.update(message_ids)



    # --- conversation lifecycle ---------------------------------------------

    def start_conversation(self) -> str:
        """Call once, before the first LLM turn, to get the opening system prompt."""
        self.state = ConversationState.GREETING
        prompt = self.build_system_prompt()
        return prompt

    def process_user_message(
        self,
        message: str,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        is_policy_question: Optional[bool] = None,
    ) -> str:
        """
        Call once per user turn with the transcribed text. Updates the
        profile and state, and returns the system prompt to use for
        generating this turn's reply.
        """
        message = normalize_stt_aliases(message).strip()
        self._append_history_message("user", message)

        # Capture the field this turn was expected to answer before extraction
        # changes the profile. Name is a conversational prerequisite even
        # though it remains optional for recommendation eligibility.
        expected_question = self.pending_question
        if expected_question is None and self.state == ConversationState.GREETING and not self.profile.is_filled("name"):
            expected_question = _NAME_QUESTION

        # Always extract customer profile information from user turns
        updated_fields = self._extract_information(message)

        if is_policy_question is None:
            is_policy_question = self._looks_like_policy_question(message)

        if (
            expected_question is not None
            and not is_policy_question
            and not self.profile.is_filled(expected_question.field_name)
            and not updated_fields
        ):
            self.last_unrecognized_question = expected_question
        else:
            self.last_unrecognized_question = None

        if self.is_profile_complete():
            if not self.recommendation_delivered:
                self.recommendation_delivered = True
                self.transition_state(ConversationState.RECOMMENDING_POLICY)
            elif is_policy_question:
                self.transition_state(ConversationState.ANSWERING_POLICY_QUESTIONS)
            else:
                self._advance_state(message)
        else:
            if is_policy_question:
                self.transition_state(ConversationState.ANSWERING_POLICY_QUESTIONS)
            else:
                self._advance_state(message)

        self.pending_question = get_next_required_question(self.profile)
        if self.pending_question:
            self.asked_fields.add(self.pending_question.field_name)

        return self.build_system_prompt(retrieved_chunks=retrieved_chunks)

    def record_assistant_reply(self, message: str) -> None:
        """Call once per turn with the bot's spoken reply, to keep history accurate."""
        self._append_history_message("assistant", message)


    # --- profile / question flow --------------------------------------------

    def update_customer_profile(self, field_name: str, value: Any) -> None:
        """Directly set a profile field, bypassing text parsing."""
        if not hasattr(self.profile, field_name):
            raise AttributeError(f"CustomerProfile has no field '{field_name}'")
        setattr(self.profile, field_name, value)

    def get_next_question(self) -> Optional[Question]:
        """The next required question that hasn't been answered yet, if any."""
        return get_next_required_question(self.profile)

    def is_profile_complete(self) -> bool:
        """Whether every required field has an answer."""
        return is_required_complete(self.profile)

    def should_recommend_policy(self) -> bool:
        """Whether it's time to move to a recommendation (and hasn't already happened)."""
        return self.is_profile_complete() and not self.recommendation_delivered

    # --- prompt / LLM integration --------------------------------------------

    def build_system_prompt(self, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Generate the system prompt for the current state and profile."""
        recommendations = None
        if self.state == ConversationState.RECOMMENDING_POLICY:
            recommendations = self.rec_engine.recommend(self.profile, limit=3)
            self.last_recommended_policies = list(recommendations)
            # A single recommendation is an unambiguous active policy. For
            # multiple recommendations we wait for a name or ordinal reference.
            if len(recommendations) == 1:
                self._set_active_policy(recommendations[0])

        return prompts.build_system_prompt(
            state=self.state,
            profile=self.profile,
            next_question=self.pending_question,
            missing_fields=missing_required_fields(self.profile),
            recommendations=recommendations,
            retrieved_chunks=retrieved_chunks,
            email_state=self.email_state.value,
            policy_selection_required=self.policy_selection_required,
            recommended_policies=self.last_recommended_policies,
            retry_question=self.last_unrecognized_question,
        )

    def _set_active_policy(self, policy: Any) -> None:
        self.active_policy_id = str(policy.policy_id)
        self.active_policy_name = str(policy.policy_name)

    def _resolve_recommended_policy(self, message: str) -> Optional[Any]:
        """Resolve a named or ordinal reference against the last recommendations."""
        text = message.lower().strip()
        ordinal_index = None
        ordinal_terms = (
            (0, ("first", "1st", "one")),
            (1, ("second", "2nd", "two")),
            (2, ("third", "3rd", "three")),
        )
        for index, terms in ordinal_terms:
            if any(term in text for term in terms):
                ordinal_index = index
                break
        if ordinal_index is not None and ordinal_index < len(self.last_recommended_policies):
            return self.last_recommended_policies[ordinal_index]

        for policy in self.last_recommended_policies:
            name = str(getattr(policy, "policy_name", "")).lower()
            insurer = str(getattr(policy, "insurer", "")).lower()
            if name and name in text:
                return policy
            # Insurer names provide a useful voice-friendly shorthand.
            insurer_tokens = [token for token in re.findall(r"[a-z0-9]+", insurer) if len(token) >= 4]
            if insurer_tokens and any(token in text for token in insurer_tokens):
                return policy
        return None

    def should_retrieve_policy_context(self, message: str) -> bool:
        """Decide whether this turn needs RAG without relying on STT punctuation."""
        text = normalize_stt_aliases(message).lower().strip()
        explicit_policy_question = self._looks_like_policy_question(text)
        has_detail_term = any(term in text for term in _POLICY_DETAIL_TERMS)
        question_like = self._is_question_like(text)
        recent_follow_up = self.policy_discussion_turns_remaining > 0 and question_like
        return explicit_policy_question or (question_like and has_detail_term) or recent_follow_up

    def prepare_policy_context(self, message: str) -> Optional[str]:
        """Resolve the policy scope for a retrieval turn and retain it for follow-ups."""
        self.policy_selection_required = False
        message = normalize_stt_aliases(message)
        selected = self._resolve_recommended_policy(message)
        if selected is not None:
            self._set_active_policy(selected)

        text = message.lower().strip()
        is_deictic = any(term in text for term in _POLICY_REFERENCE_TERMS)
        if (
            self.should_retrieve_policy_context(message)
            and self.active_policy_id is None
            and len(self.last_recommended_policies) > 1
            and (is_deictic or any(term in text for term in ("policy number", "policy code", "policy id")))
        ):
            self.policy_selection_required = True
            return None

        self.policy_discussion_turns_remaining = 2
        return self.active_policy_id

    def complete_policy_turn(self, was_policy_question: bool) -> None:
        if was_policy_question:
            self.policy_discussion_turns_remaining = 2
        elif self.policy_discussion_turns_remaining:
            self.policy_discussion_turns_remaining -= 1

    @staticmethod
    def _is_question_like(text: str) -> bool:
        return bool(text) and (
            text.endswith("?")
            or any(text.startswith(starter) for starter in _QUESTION_STARTERS)
        )

    def get_llm_messages(self) -> List[Dict[str, str]]:
        """
        The exact message list to hand to Gemini for this turn: a freshly
        built system prompt followed by the conversation so far. Gemini
        never sees anything beyond this - no RAG, no tools, no external
        memory.
        """
        return [{"role": "system", "content": self.build_system_prompt()}, *self.history]

    # --- state machine --------------------------------------------------------

    def transition_state(self, new_state: ConversationState) -> None:
        """Move to `new_state`, or raise if that transition isn't allowed."""
        if not can_transition(self.state, new_state):
            raise InvalidStateTransitionError(f"Cannot go from {self.state} to {new_state}")
        self.state = new_state

    def _advance_state(self, message: str) -> None:
        """
        Recompute the 'real' state from current progress. Used whenever the
        turn wasn't a policy-question side trip - including right after one,
        which is how ANSWERING_POLICY_QUESTIONS naturally resolves back to
        the right place without needing a manual state stack.
        """
        if self.state == ConversationState.ENDING_CALL:
            return  # terminal

        if self.state == ConversationState.RECOMMENDING_POLICY and self._user_wants_to_end(message):
            self.transition_state(ConversationState.ENDING_CALL)
            return

        if self.is_profile_complete():
            self.recommendation_delivered = True
            self.transition_state(ConversationState.RECOMMENDING_POLICY)
        else:
            self.transition_state(ConversationState.COLLECTING_INFORMATION)
    # --- extraction / heuristics ------------------------------------------------

    def _extract_information(self, message: str) -> Set[str]:
        """Extract profile values and return every field successfully updated."""
        updated_fields: Set[str] = set()
        # 1. Handle user confirmation or rejection if an email is pending verification
        msg_lower = message.lower()
        tokens = set(re.findall(r"[a-z']+", msg_lower))

        rejected_email = False
        if self.profile.pending_email and not self.profile.email_confirmed:
            affirmative_words = {"yes", "yeah", "yep", "ha", "haan", "correct", "right", "true", "exact", "exactly", "sure", "ok", "okay"}
            negative_words = {"no", "nope", "nah", "nahi", "wrong", "incorrect", "false"}

            if any(w in tokens for w in affirmative_words) or "that's right" in msg_lower or "thats right" in msg_lower or "is correct" in msg_lower:
                self.profile.email = self.profile.pending_email
                self.profile.email_confirmed = True
                self.profile.pending_email = None
                self.email_state = EmailDeliveryState.PENDING
                updated_fields.add("email")
            elif any(w in tokens for w in negative_words) or "not right" in msg_lower or "wrong" in msg_lower:
                self.profile.pending_email = None
                self.profile.email_confirmed = False
                self.email_state = EmailDeliveryState.INVALID
                updated_fields.add("email")
                self._begin_email_fragment_capture()
                rejected_email = True

        # A correction can arrive after an LLM-only confirmation prompt, where
        # no pending address exists yet because STT split the original email.
        if not self.profile.pending_email and re.match(r"^\s*no\b", msg_lower) and "it is" in msg_lower:
            self._begin_email_fragment_capture()
            rejected_email = True

        # 2. Extract new candidate email if present in message
        extracted_email = None if rejected_email else self._capture_email_candidate(message)
        if extracted_email:
            if extracted_email != self.profile.email or not self.profile.email_confirmed:
                self.profile.pending_email = extracted_email
                self.profile.email_confirmed = False
                self.email_state = EmailDeliveryState.PENDING
                self.email_sent = False
                updated_fields.add("email")

        pending_field = self.pending_question.field_name if self.pending_question else None
        extracted_pending = False
        if pending_field is not None and pending_field != "email":
            extracted_pending = self.profile.apply_raw_answer(pending_field, message)
            if extracted_pending:
                updated_fields.add(pending_field)
        elif self.state == ConversationState.GREETING:
            if self.profile.apply_raw_answer("name", message):
                updated_fields.add("name")

        for q in REQUIRED_QUESTIONS + OPTIONAL_QUESTIONS:
            field_name = q.field_name
            if field_name == "email":
                continue
            if field_name == pending_field and extracted_pending:
                continue

            kws = _FIELD_KEYWORDS.get(field_name, [field_name, q.topic])
            is_explicit_mention = any(kw in msg_lower for kw in kws)
            # Required fields may be captured opportunistically so callers can
            # answer more than one question at once. Optional parsers (notably
            # free-text medical conditions) run only when their topic is
            # actually mentioned; otherwise a phrase such as "not sure" can be
            # mistakenly treated as profile data.
            should_try_extract = (
                (field_name in _REQUIRED_FIELD_NAMES and not self.profile.is_filled(field_name))
                or is_explicit_mention
            )
            if should_try_extract:
                if self.profile.apply_raw_answer(field_name, message):
                    updated_fields.add(field_name)

        return updated_fields

    def _begin_email_fragment_capture(self) -> None:
        self._email_fragment_buffer = ""
        self._collecting_email_fragments = True

    def _capture_email_candidate(self, message: str) -> Optional[str]:
        """Parse a complete email or combine adjacent STT email fragments.

        Deepgram can emit a local part and its domain as separate final
        transcripts. The buffer is session-local and bounded; it is cleared as
        soon as an address is recognised, so unrelated speech is never kept.
        """
        # Phrases such as "Gmail is" are introductions, not part of the
        # username. They also reliably signal that the caller is restarting or
        # correcting a previously spoken address, so discard stale fragments.
        restarting = bool(_EMAIL_FRAGMENT_RESTART_RE.match(message))
        fragment = _EMAIL_FRAGMENT_RESTART_RE.sub("", message).strip() if restarting else message.strip()
        if restarting:
            self._begin_email_fragment_capture()
            self.profile.pending_email = None
            self.profile.email_confirmed = False
            self.email_state = EmailDeliveryState.PENDING
            self.email_sent = False

        if not fragment:
            return None

        complete_email = _parse_email(fragment)
        if complete_email:
            self._email_fragment_buffer = ""
            self._collecting_email_fragments = False
            return complete_email

        is_email_fragment = bool(_EMAIL_FRAGMENT_MARKER_RE.search(fragment))
        if not (self._collecting_email_fragments or is_email_fragment):
            return None

        if self._email_fragment_buffer and "@" not in self._email_fragment_buffer and _EMAIL_DOMAIN_FRAGMENT_RE.match(fragment):
            self._email_fragment_buffer += " at "
        self._email_fragment_buffer = f"{self._email_fragment_buffer} {fragment}".strip()[-512:]
        self._collecting_email_fragments = True

        combined_email = _parse_email(self._email_fragment_buffer)
        if combined_email:
            self._email_fragment_buffer = ""
            self._collecting_email_fragments = False
            return combined_email
        return None



    @staticmethod
    def _looks_like_policy_question(message: str) -> bool:
        """
        Detects if a user message is asking for policy details or document information.
        Handles real-time spoken transcriptions which may not end with question marks.
        """
        text = message.lower().strip()
        has_keyword = any(keyword in text for keyword in _POLICY_QUESTION_KEYWORDS)
        has_detail_term = any(term in text for term in _POLICY_DETAIL_TERMS)
        has_starter = ConversationManager._is_question_like(text)
        
        # Trigger policy question state if it combines policy context with question phrasing/syntax
        return ((has_keyword or has_detail_term) and has_starter) or (text.endswith("?") and len(text.split()) >= 2)


    @staticmethod
    def _user_wants_to_end(message: str) -> bool:
        text = message.lower()
        return any(phrase in text for phrase in _END_CALL_PHRASES)


if __name__ == "__main__":
    # Minimal smoke test / usage example. Run with: python -m conversation.conversation_manager
    manager = ConversationManager()
    print("SYSTEM:", manager.start_conversation())

    turns = [
        "Hi, this is Rahul.",
        "I'm 34 years old.",
        "Male.",
        "I live in Pune.",
        "It'll be 4 of us.",
        "Yes, please include my parents too.",
        "No children yet.",
        "None, we're all healthy.",
        "No I don't smoke.",
        "What is a waiting period in health insurance?",
        "Around 25000 per year.",
        "Maybe 10 lakh coverage.",
        "No I don't have any policy currently.",
        "No that's all, thank you!",
    ]
    for turn in turns:
        print("\nUSER:", turn)
        system_prompt = manager.process_user_message(turn)
        print(f"[state={manager.state.value}]")
        manager.record_assistant_reply("(simulated LLM reply)")

    print("\nFinal profile:", manager.profile)
    print("Profile complete:", manager.is_profile_complete())
