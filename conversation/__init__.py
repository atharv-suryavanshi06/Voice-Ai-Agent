"""
Conversation Manager package for the insurance policy recommendation voice
agent.

This package is self-contained (stdlib only, no new pip dependencies) and
does not import anything from, or get imported by, the existing Pipecat
pipeline. It only prepares system prompts and message lists for whatever
code currently owns the GoogleLLMService call.

Public API:
    ConversationManager  - orchestrates one call
    ConversationState    - the state machine's states
    CustomerProfile      - structured data collected about the caller
"""

from .conversation_manager import ConversationManager
from .customer_profile import CustomerProfile
from .state import ConversationState

__all__ = ["ConversationManager", "CustomerProfile", "ConversationState"]
