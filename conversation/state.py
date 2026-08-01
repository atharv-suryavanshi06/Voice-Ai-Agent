"""
state.py

The conversation's state machine. Keeping an explicit state, plus an
explicit table of which transitions are legal, is what lets the Conversation
Manager control the call instead of letting the LLM decide the flow -
`transition_state()` in the manager refuses to move to a state that isn't
reachable from the current one.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class ConversationState(Enum):
    """Where the call currently is."""

    GREETING = "greeting"
    COLLECTING_INFORMATION = "collecting_information"
    ANSWERING_POLICY_QUESTIONS = "answering_policy_questions"
    RECOMMENDING_POLICY = "recommending_policy"
    ENDING_CALL = "ending_call"


class InvalidStateTransitionError(ValueError):
    """Raised when the manager attempts a transition that isn't allowed."""


# ANSWERING_POLICY_QUESTIONS is a transient "side trip": the caller
# interrupts with a question, the manager answers it for one turn, and then
# falls back to whatever state the underlying progress (profile completeness
# / call-ending intent) says is correct. That's why it can lead almost
# anywhere, and almost every other state can lead into it.
ALLOWED_TRANSITIONS: Dict[ConversationState, Set[ConversationState]] = {
    ConversationState.GREETING: {
        ConversationState.GREETING,
        ConversationState.COLLECTING_INFORMATION,
        ConversationState.ANSWERING_POLICY_QUESTIONS,
        ConversationState.RECOMMENDING_POLICY,
    },
    ConversationState.COLLECTING_INFORMATION: {
        ConversationState.COLLECTING_INFORMATION,
        ConversationState.ANSWERING_POLICY_QUESTIONS,
        ConversationState.RECOMMENDING_POLICY,
    },
    ConversationState.ANSWERING_POLICY_QUESTIONS: {
        ConversationState.ANSWERING_POLICY_QUESTIONS,
        ConversationState.COLLECTING_INFORMATION,
        ConversationState.RECOMMENDING_POLICY,
        ConversationState.ENDING_CALL,
    },
    ConversationState.RECOMMENDING_POLICY: {
        ConversationState.RECOMMENDING_POLICY,
        ConversationState.ANSWERING_POLICY_QUESTIONS,
        ConversationState.COLLECTING_INFORMATION,  # e.g. caller revises an earlier answer
        ConversationState.ENDING_CALL,
    },
    ConversationState.ENDING_CALL: {
        ConversationState.ENDING_CALL,  # terminal
    },
}


def can_transition(current: ConversationState, target: ConversationState) -> bool:
    """Whether moving from `current` to `target` is a legal transition."""
    return target in ALLOWED_TRANSITIONS.get(current, set())
