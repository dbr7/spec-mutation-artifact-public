"""Shared survivor-analysis helpers.

The main mutation run no longer performs runtime LLM survivor triage.
Post-hoc reporting scripts still share the lightweight result record
and redundancy heuristic defined here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass
class SurvivorTriage:
    label: str
    confidence: str
    reason: str
    probe_count: int = 0
    valid_probe_count: int = 0
    distinguishing_input: Any | None = None
    reference_output: Any | None = None
    mutant_output: Any | None = None
    reference_status: str | None = None
    mutant_status: str | None = None


_TOKEN_RE = re.compile(
    r"`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|"
    r"\\?[A-Za-z_][A-Za-z_0-9]*|"
    r"-?\d+(?:\.\d+)?|<=|>=|!=|==|<|>|="
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "can", "each", "for",
    "from", "given", "if", "in", "input", "is", "it", "line", "of",
    "on", "or", "output", "print", "standard", "the", "then", "there",
    "to", "with", "you",
}


def _safe_json_value(value: Any, limit: int = 4000) -> Any:
    """Return a JSON-friendly, bounded representation."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + "...[truncated]"
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v, limit) for v in value[:50]]
    if isinstance(value, dict):
        return {
            str(k): _safe_json_value(v, limit)
            for k, v in list(value.items())[:50]
        }
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _semantic_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip()
        bare = token.strip("`\"'")
        lower = bare.lower()
        if lower in _STOPWORDS:
            continue
        if bare.isalpha() and len(bare) < 3:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _deleted_or_replaced_fragments(original: str, mutated: str) -> list[str]:
    fragments: list[str] = []
    matcher = SequenceMatcher(None, original or "", mutated or "")
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            frag = original[i1:i2].strip()
            if frag:
                fragments.append(frag)
    return fragments


def _possible_redundancy_signal(
    original: str, mutated: str, quote: str = ""
) -> tuple[bool, str]:
    """Detect old changed tokens that still appear elsewhere.

    This is intentionally a medium-confidence heuristic, not a proof
    of equivalence. It is useful for ranking survivors whose observed
    behaviour did not diverge on probes.
    """
    candidates: list[str] = []
    for frag in _deleted_or_replaced_fragments(original, mutated):
        candidates.extend(_semantic_tokens(frag))
    if quote:
        candidates.extend(_semantic_tokens(quote))

    seen: set[str] = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        bare = token.strip("`\"'")
        if len(bare) < 2:
            continue
        if original.count(bare) <= 1:
            continue
        if mutated.count(bare) > 0:
            return True, f"old changed token still appears elsewhere: {bare!r}"
    return False, ""
