"""Strict, side-effect-free policy source discovery by embedded policy ID."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


_POLICY_NUMBER_PATTERNS = (
    re.compile(
        r"(?im)^\s*\|\s*(?:\*\*)?\s*Policy\s*(?:Number|No\.?|ID)\s*"
        r"(?:\*\*)?\s*\|\s*(?P<value>[^|\r\n]+)"
    ),
    re.compile(
        r"(?im)^\s*(?:\*\*)?\s*Policy\s*(?:Number|No\.?|ID)\s*"
        r"(?:\*\*)?\s*:\s*(?P<value>[^\r\n]+)"
    ),
)
_PRODUCT_CODE_PATTERNS = (
    re.compile(
        r"(?im)^\s*\|\s*(?:\*\*)?\s*(?:Product|Policy)\s+Code(?:\s*\(UIN\))?\s*"
        r"(?:\*\*)?\s*\|\s*(?P<value>[^|\r\n]+)"
    ),
    re.compile(
        r"(?im)^\s*(?:\*\*)?\s*(?:Product|Policy)\s+Code(?:\s*\(UIN\))?\s*"
        r"(?:\*\*)?\s*:\s*(?P<value>[^\r\n]+)"
    ),
)


class MarkdownSourceValidationError(ValueError):
    """Raised when Markdown-to-policy identity is not exactly one-to-one."""


@dataclass(frozen=True)
class MarkdownPolicySource:
    policy_id: str
    policy_name: str
    policy_code: Optional[str]
    path: Path
    text: str
    source_hash: str


def _clean_labeled_value(value: str) -> str:
    return value.strip().strip("*`_").strip()


def _extract_values(text: str, patterns: Sequence[re.Pattern]) -> List[str]:
    values = []
    for pattern in patterns:
        values.extend(_clean_labeled_value(match.group("value")) for match in pattern.finditer(text))
    return [value for value in values if value]


def _catalog_by_id(
    catalog_entries: Iterable[Union[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    normalized_ids: Dict[str, str] = {}
    for raw_entry in catalog_entries:
        if isinstance(raw_entry, str):
            entry = {"policy_id": raw_entry, "policy_name": ""}
        else:
            entry = dict(raw_entry)
            if entry.get("_ingestion_status", "active") != "active":
                continue
        policy_id = str(entry.get("policy_id") or "").strip()
        if not policy_id:
            raise MarkdownSourceValidationError("Active catalog entry is missing policy_id")
        normalized = policy_id.casefold()
        if normalized in normalized_ids:
            raise MarkdownSourceValidationError(f"Duplicate active catalog policy_id: {policy_id}")
        normalized_ids[normalized] = policy_id
        result[policy_id] = entry
    if not result:
        raise MarkdownSourceValidationError("The active policy catalog is empty")
    return result


def resolve_markdown_policy_sources(
    data_dir: Union[str, Path],
    catalog_entries: Iterable[Union[str, Mapping[str, Any]]],
) -> Dict[str, MarkdownPolicySource]:
    """Resolve every active policy to exactly one Markdown file by embedded ID.

    Repeated labeled occurrences of the same policy number in one document are
    valid. Missing, unknown, multi-ID and duplicate-source documents fail as a
    group before callers perform any database or network operation. Filename
    similarity is never used as identity.
    """
    catalog = _catalog_by_id(catalog_entries)
    canonical_by_normalized = {policy_id.casefold(): policy_id for policy_id in catalog}
    directory = Path(data_dir)
    if not directory.is_dir():
        raise MarkdownSourceValidationError(f"Markdown data directory does not exist: {directory}")

    paths = sorted(directory.glob("*.md"), key=lambda path: path.name.casefold())
    if not paths:
        raise MarkdownSourceValidationError(f"No Markdown policy sources found in: {directory}")

    resolved: Dict[str, MarkdownPolicySource] = {}
    errors: List[str] = []
    for path in paths:
        try:
            source_bytes = path.read_bytes()
            text = source_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{path.name}: cannot read UTF-8 source ({exc})")
            continue

        # A set intentionally permits repeated labeled occurrences of one ID.
        labeled_numbers = set(_extract_values(text, _POLICY_NUMBER_PATTERNS))
        normalized_numbers = {value.casefold(): value for value in labeled_numbers}
        unknown = sorted(
            original
            for normalized, original in normalized_numbers.items()
            if normalized not in canonical_by_normalized
        )
        known = sorted(
            canonical_by_normalized[normalized]
            for normalized in normalized_numbers
            if normalized in canonical_by_normalized
        )
        if unknown:
            errors.append(f"{path.name}: unknown policy number(s): {', '.join(unknown)}")
            continue
        if len(known) != 1:
            detail = "no labeled policy number" if not known else f"multiple policy numbers: {', '.join(known)}"
            errors.append(f"{path.name}: {detail}")
            continue

        policy_id = known[0]
        if policy_id in resolved:
            errors.append(
                f"{policy_id}: duplicate Markdown sources: "
                f"{resolved[policy_id].path.name}, {path.name}"
            )
            continue

        entry = catalog[policy_id]
        source_codes = sorted(set(_extract_values(text, _PRODUCT_CODE_PATTERNS)))
        catalog_code = str(entry.get("policy_code") or "").strip() or None
        policy_code = catalog_code or (source_codes[0] if len(source_codes) == 1 else None)
        if catalog_code and source_codes and catalog_code.casefold() not in {
            value.casefold() for value in source_codes
        }:
            errors.append(
                f"{path.name}: catalog policy_code '{catalog_code}' does not match source code(s) "
                f"{', '.join(source_codes)}"
            )
            continue
        if len(source_codes) > 1:
            errors.append(f"{path.name}: multiple product/policy codes: {', '.join(source_codes)}")
            continue

        resolved[policy_id] = MarkdownPolicySource(
            policy_id=policy_id,
            policy_name=str(entry.get("policy_name") or "").strip(),
            policy_code=policy_code,
            path=path.resolve(),
            text=text,
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
        )

    missing = sorted(set(catalog) - set(resolved))
    if missing:
        errors.append(f"missing Markdown source(s): {', '.join(missing)}")
    if errors:
        raise MarkdownSourceValidationError("; ".join(errors))
    return {policy_id: resolved[policy_id] for policy_id in sorted(resolved)}
