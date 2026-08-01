"""Canonical, ambiguity-safe policy identity resolution helpers.

Policy numbers are the primary identity throughout the application.  Names and
insurer brands are conversational aliases only and are never allowed to select
the first of several matching policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple


_ORDINAL_TERMS = (
    (0, ("first", "1st")),
    (1, ("second", "2nd")),
    (2, ("third", "3rd")),
)

_CONTEXTUAL_ORDINALS = (
    (0, ("one", "1")),
    (1, ("two", "2")),
    (2, ("three", "3")),
)


def normalize_identifier(value: Any) -> str:
    """Return a punctuation/spacing-insensitive representation of an ID."""
    return "".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def normalize_label(value: Any) -> str:
    """Normalize human-facing names while retaining word boundaries."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _contains_identifier(message: str, identifier: Any) -> bool:
    tokens = re.findall(r"[a-z0-9]+", str(identifier or "").casefold())
    if not tokens:
        return False

    # Codes and policy numbers are commonly spoken letter-by-letter and with
    # inconsistent separators. Preserve that flexibility while enforcing
    # exact outer boundaries so a known value is never accepted as a prefix of
    # a longer, unknown identifier.
    token_patterns = []
    for token in tokens:
        if token.isalpha() and len(token) <= 8:
            token_patterns.append(r"[^a-z0-9]*".join(map(re.escape, token)))
        else:
            token_patterns.append(re.escape(token))
    body = r"[^a-z0-9]*".join(token_patterns)
    pattern = re.compile(
        rf"(?<![a-z0-9/_.-]){body}"
        rf"(?![a-z0-9]|[/_-][^a-z0-9]*[a-z0-9]|\.[a-z0-9])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(message.casefold()))


def _contains_label(message: str, label: Any) -> bool:
    value = normalize_label(label)
    if not value:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", normalize_label(message)))


def _deduplicate(policies: Iterable[Any]) -> Tuple[Any, ...]:
    unique = {}
    for policy in policies:
        policy_id = str(getattr(policy, "policy_id", "") or "")
        if policy_id:
            unique.setdefault(policy_id, policy)
    return tuple(unique.values())


@dataclass(frozen=True)
class PolicyIdentityResolution:
    """Result of resolving policy identity from one caller utterance."""

    status: str
    policy: Optional[Any] = None
    candidates: Tuple[Any, ...] = ()
    matched_by: Optional[str] = None

    @classmethod
    def matched(cls, policy: Any, matched_by: str) -> "PolicyIdentityResolution":
        return cls("matched", policy=policy, candidates=(policy,), matched_by=matched_by)

    @classmethod
    def ambiguous(
        cls,
        candidates: Iterable[Any],
        matched_by: str,
    ) -> "PolicyIdentityResolution":
        return cls("ambiguous", candidates=_deduplicate(candidates), matched_by=matched_by)

    @classmethod
    def no_match(cls) -> "PolicyIdentityResolution":
        return cls("none")


class PolicyIdentityResolver:
    """Resolve policy IDs, codes and spoken aliases with strict precedence."""

    def __init__(self, policies: Sequence[Any]) -> None:
        self.policies = _deduplicate(policies)

    @staticmethod
    def _conflicts(selected: Any, lower_priority_matches: Sequence[Any]) -> bool:
        if not lower_priority_matches:
            return False
        selected_id = str(getattr(selected, "policy_id", ""))
        return selected_id not in {
            str(getattr(policy, "policy_id", "")) for policy in lower_priority_matches
        }

    @staticmethod
    def _contains_other_policy(selected: Any, matches: Sequence[Any]) -> bool:
        selected_id = str(getattr(selected, "policy_id", ""))
        return any(
            str(getattr(policy, "policy_id", "")) != selected_id
            for policy in matches
        )

    def _identifier_matches(self, message: str, attribute: str) -> Tuple[Any, ...]:
        return _deduplicate(
            policy
            for policy in self.policies
            if _contains_identifier(message, getattr(policy, attribute, None))
        )

    def _name_matches(self, message: str) -> Tuple[Any, ...]:
        return _deduplicate(
            policy
            for policy in self.policies
            if _contains_label(message, getattr(policy, "policy_name", None))
        )

    def _insurer_matches(self, message: str) -> Tuple[Any, ...]:
        text = normalize_label(message)
        matches = []
        ignored = {
            "insurance", "health", "general", "company", "limited", "ltd",
            "private", "pvt", "co", "nigam",
        }
        for policy in self.policies:
            insurer = str(getattr(policy, "insurer", "") or "")
            full_label = normalize_label(insurer)
            brand_tokens = [
                token for token in full_label.split()
                if len(token) >= 4 and token not in ignored
            ]
            full_match = bool(full_label and _contains_label(text, full_label))
            brand_match = any(
                re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text)
                for token in brand_tokens
            )
            if full_match or brand_match:
                matches.append(policy)
        return _deduplicate(matches)

    @staticmethod
    def _ordinal_match(message: str, recent_policies: Sequence[Any]) -> Optional[Any]:
        text = normalize_label(message)
        for index, terms in _ORDINAL_TERMS:
            if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
                if index < len(recent_policies):
                    return recent_policies[index]
                return None
        option_label = r"(?:policy|plan|option|recommendation)"
        for index, terms in _CONTEXTUAL_ORDINALS:
            if any(
                re.search(
                    rf"\b(?:{option_label}\s+(?:number\s+)?{re.escape(term)}|"
                    rf"(?:number|no)\s+{re.escape(term)}|"
                    rf"{re.escape(term)}\s+{option_label})\b",
                    text,
                )
                for term in terms
            ):
                if index < len(recent_policies):
                    return recent_policies[index]
                return None
        return None

    def resolve(
        self,
        message: str,
        recent_policies: Sequence[Any] = (),
    ) -> PolicyIdentityResolution:
        """Resolve an utterance using the documented precedence.

        A more specific identifier may disambiguate a duplicate name.  If two
        explicit signals point to incompatible policies, the result is
        ambiguous instead of silently selecting either one.
        """
        id_matches = self._identifier_matches(message, "policy_id")
        code_matches = self._identifier_matches(message, "policy_code")
        name_matches = self._name_matches(message)
        insurer_matches = self._insurer_matches(message)

        if len(id_matches) > 1:
            return PolicyIdentityResolution.ambiguous(id_matches, "policy_id")
        if id_matches:
            selected = id_matches[0]
            if (
                self._contains_other_policy(selected, code_matches)
                or self._conflicts(selected, name_matches)
                or self._conflicts(selected, insurer_matches)
            ):
                return PolicyIdentityResolution.ambiguous(
                    (*id_matches, *code_matches, *name_matches, *insurer_matches),
                    "conflicting_identifiers",
                )
            return PolicyIdentityResolution.matched(selected, "policy_id")

        if len(code_matches) > 1:
            return PolicyIdentityResolution.ambiguous(code_matches, "policy_code")
        if code_matches:
            selected = code_matches[0]
            if self._conflicts(selected, name_matches) or self._conflicts(selected, insurer_matches):
                return PolicyIdentityResolution.ambiguous(
                    (*code_matches, *name_matches, *insurer_matches),
                    "conflicting_identifiers",
                )
            return PolicyIdentityResolution.matched(selected, "policy_code")

        ordinal = self._ordinal_match(message, recent_policies)
        if ordinal is not None:
            if self._conflicts(ordinal, name_matches) or self._conflicts(ordinal, insurer_matches):
                return PolicyIdentityResolution.ambiguous(
                    (ordinal, *name_matches, *insurer_matches),
                    "conflicting_identifiers",
                )
            return PolicyIdentityResolution.matched(ordinal, "ordinal")

        if len(recent_policies) > 1 and re.search(
            r"\b(?:which|the|this|that)\s+one\b",
            normalize_label(message),
        ):
            return PolicyIdentityResolution.ambiguous(recent_policies, "ordinal")

        if len(name_matches) == 1:
            return PolicyIdentityResolution.matched(name_matches[0], "policy_name")
        if len(name_matches) > 1:
            return PolicyIdentityResolution.ambiguous(name_matches, "policy_name")

        if len(insurer_matches) == 1:
            return PolicyIdentityResolution.matched(insurer_matches[0], "insurer")
        if len(insurer_matches) > 1:
            return PolicyIdentityResolution.ambiguous(insurer_matches, "insurer")

        return PolicyIdentityResolution.no_match()


def policy_display_labels(
    policies: Sequence[Any],
    duplicate_names: Iterable[str] = (),
) -> Tuple[str, ...]:
    """Return labels, adding code/number only where marketing names collide."""
    normalized_names = [normalize_label(getattr(policy, "policy_name", "")) for policy in policies]
    known_duplicates = {normalize_label(name) for name in duplicate_names if normalize_label(name)}
    labels = []
    for policy, normalized_name in zip(policies, normalized_names):
        name = str(getattr(policy, "policy_name", "") or "Policy")
        if normalized_name and (
            normalized_names.count(normalized_name) > 1
            or normalized_name in known_duplicates
        ):
            code = str(getattr(policy, "policy_code", "") or "").strip()
            policy_id = str(getattr(policy, "policy_id", "") or "").strip()
            identifiers = []
            if code:
                identifiers.append(f"code {code}")
            if policy_id:
                identifiers.append(f"policy number {policy_id}")
            if identifiers:
                name = f"{name} ({'; '.join(identifiers)})"
        labels.append(name)
    return tuple(labels)


def duplicate_policy_name_keys(policies: Sequence[Any]) -> frozenset[str]:
    """Return normalized marketing names that identify multiple policy IDs."""
    ids_by_name = {}
    for policy in policies:
        key = normalize_label(getattr(policy, "policy_name", ""))
        policy_id = str(getattr(policy, "policy_id", "") or "")
        if key and policy_id:
            ids_by_name.setdefault(key, set()).add(policy_id)
    return frozenset(key for key, policy_ids in ids_by_name.items() if len(policy_ids) > 1)
