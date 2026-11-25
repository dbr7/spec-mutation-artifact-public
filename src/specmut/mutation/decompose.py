"""Decompose a specification into atomic clauses.

Each clause is returned as a ``Clause`` record carrying four fields:

* ``text``       : paraphrased behavioural constraint.
* ``quote``      : verbatim substring of the specification that the
                   clause most directly corresponds to, or the empty
                   string if the clause is genuinely distributed.
* ``type_tag``   : one of the structural types declared in
                   ``VALID_TYPES``; ``"unknown"`` if the decomposer
                   failed to classify.
* ``applicable_ops`` : list of mutation operator names that are
                   meaningful for this clause; defaults to the active
                   LLM operators when the decomposer did not specify.

This enrichment lets the mutation step skip operator-clause pairs
that are structurally nonsensical (for example, ``perturb`` on a
clause that contains no numeric boundary), which was a dominant cause
of noop mutants in early runs.
"""

import logging
import re
from dataclasses import dataclass, field

from specmut.config import Config
from specmut.llm import LLMClient
from specmut.mutation.mutate import OPERATORS

logger = logging.getLogger(__name__)

VALID_TYPES = {
    "core_behaviour",
    "boundary",
    "return_value",
    "ordering",
    "edge_case",
    "precondition",
    "io_format",
    "example",
}

_TYPE_ALIASES = {
    "core": "core_behaviour",
    "behaviour": "core_behaviour",
    "behavior": "core_behaviour",
    "core behaviour": "core_behaviour",
    "core behavior": "core_behaviour",
    "bound": "boundary",
    "return": "return_value",
    "return value": "return_value",
    "order": "ordering",
    "edge": "edge_case",
    "edge case": "edge_case",
    "pre": "precondition",
    "format": "io_format",
    "io": "io_format",
    "i/o": "io_format",
}

_SECTION_OR_TYPE_LABELS = {
    *VALID_TYPES,
    *_TYPE_ALIASES,
    "constraints",
    "input",
    "output",
    "note",
    "notes",
    "sample",
    "samples",
}


@dataclass
class Clause:
    text: str
    quote: str = ""
    type_tag: str = "unknown"
    applicable_ops: list[str] = field(default_factory=lambda: list(OPERATORS))


@dataclass
class _ParsedDecomposition:
    clauses: list[Clause]
    numbered_lines: int = 0
    malformed_lines: list[str] = field(default_factory=list)
    dropped_unanchored_lines: list[str] = field(default_factory=list)
    unsafe_quote_lines: list[str] = field(default_factory=list)
    dropped_duplicate_anchor_lines: list[str] = field(default_factory=list)


_DECOMPOSE_RETRY_MAX = 2
_DECOMPOSE_MAX_TOKENS = 8192


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalised_with_index_map(s: str) -> tuple[str, list[int]]:
    """Return whitespace-normalised text plus original char indices."""
    chars: list[str] = []
    indices: list[int] = []
    in_space = False
    for idx, ch in enumerate(s):
        if ch.isspace():
            if chars and not in_space:
                chars.append(" ")
                indices.append(idx)
            in_space = True
            continue
        chars.append(ch)
        indices.append(idx)
        in_space = False

    while chars and chars[-1] == " ":
        chars.pop()
        indices.pop()
    return "".join(chars), indices


def _resolve_quote_to_exact_span(quote: str, spec: str) -> str:
    """Map a whitespace-normalised quote match back to exact spec text.

    Tooling models sometimes copy a source span with collapsed
    whitespace or stop at a token prefix (e.g. ``close`` vs
    ``closer``). The mutator needs a byte-exact substring, so when the
    normalised quote is accepted we return the corresponding exact
    source slice and expand partial word matches to word boundaries.
    """
    if not quote:
        return ""
    if quote in spec:
        orig_start = spec.find(quote)
        orig_end = orig_start + len(quote)
        while (
            orig_start > 0
            and spec[orig_start - 1].isalnum()
            and spec[orig_start].isalnum()
        ):
            orig_start -= 1
        while (
            orig_end < len(spec)
            and spec[orig_end - 1].isalnum()
            and spec[orig_end].isalnum()
        ):
            orig_end += 1
        return spec[orig_start:orig_end]

    norm_quote = _normalize_whitespace(quote)
    norm_spec, index_map = _normalised_with_index_map(spec)
    start = norm_spec.find(norm_quote)
    if start < 0 or not index_map:
        return ""

    end_norm = start + len(norm_quote) - 1
    if end_norm >= len(index_map):
        return ""

    orig_start = index_map[start]
    orig_end = index_map[end_norm] + 1

    while (
        orig_start > 0
        and spec[orig_start - 1].isalnum()
        and spec[orig_start].isalnum()
    ):
        orig_start -= 1
    while (
        orig_end < len(spec)
        and spec[orig_end - 1].isalnum()
        and spec[orig_end].isalnum()
    ):
        orig_end += 1

    return spec[orig_start:orig_end]


def _parse_ops(raw: str) -> list[str]:
    ops = [
        op.strip().lower().strip(".,;:")
        for op in raw.split(",")
    ]
    ops = [op for op in ops if op in OPERATORS]
    return ops


_SAMPLE_CLAUSE_RE = re.compile(
    r"\b(?:sample\s+(?:input|output)\s*\d+|in\s+sample\s+\d+|"
    r"sample\s+explanation)\b|(?:\bexample\s*:)|(?:^|\n)\s*>>>",
    re.I,
)
_IO_FORMAT_CLAUSE_RE = re.compile(
    r"\b(?:standard\s+input|input\s+is\s+given|given\s+from\s+standard\s+input|"
    r"following\s+format|input\s+consists|output\s+format|"
    r"print\s+[A-Za-z_][A-Za-z0-9_]*\s+lines?|"
    r"[A-Za-z0-9_{}\\-]+-th\s+line|each\s+line|"
    r"separated\s+by\s+a\s+space|output\s+must\s+be\s+printed)\b",
    re.I,
)


def _looks_like_sample_clause(text: str, quote: str) -> bool:
    return _SAMPLE_CLAUSE_RE.search(f"{text or ''}\n{quote or ''}") is not None


def _looks_like_io_format_clause(text: str, quote: str) -> bool:
    return _IO_FORMAT_CLAUSE_RE.search(f"{text or ''}\n{quote or ''}") is not None


def _looks_like_label_only_clause(text: str) -> bool:
    label = _normalize_whitespace(text).lower().replace("-", "_")
    label = re.sub(r"\s+", " ", label)
    return label in _SECTION_OR_TYPE_LABELS


def _quote_branch_budget(quote: str) -> int:
    """Return how many clauses a single exact anchor may own.

    Most source spans should map to one canonical clause. The main
    exception we have seen in LCB is an explicit two-branch instruction
    definition, e.g. one branch for ``H_i is L`` and one for
    ``H_i is R``. Keeping both preserves the useful split while still
    collapsing later restatements of the same sentence.
    """
    q = _normalize_whitespace(quote).lower()
    has_l = re.search(r"\b(?:if|when)\s+h_i\s+is\s+l\b", q) is not None
    has_r = re.search(r"\b(?:if|when)\s+h_i\s+is\s+r\b", q) is not None
    return 2 if has_l and has_r else 1


def _clause_branch_key(clause: Clause) -> str:
    text = clause.text.lower()
    left = re.search(r"\b(?:h_i\s+is\s+l|left\s+hand)\b", text) is not None
    right = re.search(r"\b(?:h_i\s+is\s+r|right\s+hand)\b", text) is not None
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "generic"


def _merge_applicable_ops(target: Clause, source: Clause) -> None:
    """Preserve operator information from a dropped same-anchor clause."""
    merged = list(dict.fromkeys(target.applicable_ops + source.applicable_ops))
    if target.type_tag != "example":
        merged = [op for op in merged if op != "example"]
    if target.type_tag == "example":
        merged = [op for op in merged if op == "example"] or ["example"]
    target.applicable_ops = merged


def _collapse_duplicate_anchor_clauses(
    clauses: list[Clause],
) -> tuple[list[Clause], list[str]]:
    """Collapse paraphrase duplicates that point at the same quote.

    The decomposer sometimes emits a complete pass, then another pass
    that restates the same anchors with slightly different wording.
    Text-only dedup does not catch that. Since the mutator applies
    edits to quoted source spans, repeated clauses with the same exact
    anchor usually generate duplicate mutated specs and noisy
    ``no mutants`` rows. Collapse those at parse time.
    """
    groups: dict[str, list[int]] = {}
    for idx, clause in enumerate(clauses):
        key = _normalize_whitespace(clause.quote).lower()
        groups.setdefault(key, []).append(idx)

    keep: set[int] = set()
    dropped: list[str] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            keep.add(idxs[0])
            continue

        budget = _quote_branch_budget(clauses[idxs[0]].quote)
        chosen: list[int] = []
        if budget == 2:
            by_branch: dict[str, int] = {}
            for idx in idxs:
                branch = _clause_branch_key(clauses[idx])
                by_branch.setdefault(branch, idx)
            if "left" in by_branch and "right" in by_branch:
                chosen = [by_branch["left"], by_branch["right"]]
            else:
                chosen = idxs[:budget]
        else:
            chosen = idxs[:budget]

        keep.update(chosen)
        for idx in idxs:
            if idx in chosen:
                continue
            branch = _clause_branch_key(clauses[idx])
            target_idx = chosen[0]
            if budget == 2 and branch in {"left", "right"}:
                for candidate_idx in chosen:
                    if _clause_branch_key(clauses[candidate_idx]) == branch:
                        target_idx = candidate_idx
                        break
            _merge_applicable_ops(clauses[target_idx], clauses[idx])
            dropped.append(clauses[idx].text)

    return [clause for idx, clause in enumerate(clauses) if idx in keep], dropped


def _parse_decomposition_response(response: str, spec: str) -> _ParsedDecomposition:
    clauses: list[Clause] = []
    seen_keys: set[str] = set()
    malformed_lines: list[str] = []
    dropped_unanchored_lines: list[str] = []
    unsafe_quote_lines: list[str] = []
    numbered_lines = 0
    for line in response.strip().splitlines():
        line = line.strip()
        match = re.match(r"^\d+[.)]\s*(.+)$", line)
        if not match:
            continue
        numbered_lines += 1
        body = match.group(1).strip()

        parts = [p.strip() for p in body.split("|||")]
        if len(parts) < 4:
            malformed_lines.append(line)

        # Accept 1-4 fields. Missing fields fall back to sensible
        # defaults so older decomposer outputs still work.
        # Strip any of the common quoting characters the decomposer
        # LLM may wrap fields with: straight double quote, single
        # quote, markdown backtick, and smart quotes. Without this
        # a quote wrapped in backticks would never match the
        # verbatim spec on the substring check below.
        _WRAPPERS = "\"'`\u201c\u201d\u2018\u2019"
        clause_part = parts[0].strip(_WRAPPERS).strip()
        quote_part = (
            parts[1].strip(_WRAPPERS).strip() if len(parts) > 1 else ""
        )
        type_part = parts[2].strip().lower() if len(parts) > 2 else ""
        ops_part = parts[3] if len(parts) > 3 else ""

        if _looks_like_label_only_clause(clause_part):
            malformed_lines.append(line)
            continue

        # Validate quote: the mutator needs a byte-exact source span.
        # Accept whitespace-normalised matches only after mapping them
        # back to an exact contiguous substring of the original spec.
        if quote_part:
            quote_part = _resolve_quote_to_exact_span(quote_part, spec)

        # Validate type. Map a few common synonyms the LLM may emit.
        type_part = _TYPE_ALIASES.get(type_part, type_part)
        if type_part not in VALID_TYPES:
            type_part = "unknown"

        # Correct common type drift before operator filtering. Tooling
        # models often call sample prose "unknown" or treat "Print Q
        # lines" as core behaviour. Those clauses are useful for audit
        # context but the current mutation policy does not target them.
        if _looks_like_sample_clause(clause_part, quote_part):
            type_part = "example"
        elif _looks_like_io_format_clause(clause_part, quote_part):
            type_part = "io_format"

        norm_quote = _normalize_whitespace(quote_part)
        quote_heading = quote_part.strip()
        quote_looks_like_sample = (
            re.search(r"Sample (Input|Output) \d+", quote_part) is not None
        )
        quote_looks_like_doctest = (
            re.search(r"(?:^|\n)\s*>>>", quote_part) is not None
        )
        if quote_looks_like_sample:
            type_part = "example"
        if quote_part and (
            re.fullmatch(
                r"Sample (Input|Output) \d+", quote_heading
            ) is not None
            or quote_heading in {"Input", "Output", "Constraints"}
            or (
                type_part == "example"
                and len(norm_quote) < 20
                and not quote_looks_like_doctest
            )
            or (spec.count(quote_part) > 1 and len(norm_quote) < 20)
        ):
            unsafe_quote_lines.append(line)
            quote_part = ""

        # Parse applicable operators. Missing or empty field falls
        # back to all operators; the per-operator UNAPPLICABLE escape
        # hatch will discard genuinely inapplicable mutations at run
        # time. Empty was previously interpreted as "decomposer said
        # nothing fits", but in practice some tooling models (notably
        # gpt-5-nano) emit empty trailing fields lazily, which
        # silenced the entire mutation pipeline for some programming
        # contest style tasks.
        applicable = _parse_ops(ops_part) if len(parts) > 3 else []
        if not applicable:
            applicable = list(OPERATORS)

        # Safety rail: the example operator tampers with literal
        # values inside an I/O example block (Sample Input / Sample
        # Output, doctest, "Example:"). It is only meaningful when
        # the clause IS such a block. The decomposer LLM has been
        # observed to list ``example`` as applicable on prose rule
        # clauses (typed return_value, core_behaviour, boundary,
        # etc.), at which point the mutator produces a value tweak
        # inside prose rather than inside an example block, leaving
        # the spec internally inconsistent (prose says X, the
        # sample block still says Y) and the matrix populated with
        # mis-attributed mutations. Filter ``example`` out
        # whenever the clause type is anything other than
        # ``example``. The mutator's "if not actually an example,
        # output UNAPPLICABLE" self-check stays as a backstop for
        # the rare ``example``-typed clause that turns out not to
        # parse cleanly as one.
        if type_part != "example" and "example" in applicable:
            applicable = [op for op in applicable if op != "example"]
        if type_part == "example":
            applicable = [op for op in applicable if op == "example"]
            if not applicable:
                applicable = ["example"]
        if type_part == "io_format":
            applicable = [
                op for op in applicable
                if op in {"partial_edit", "inconsistent_example"}
            ]

        # Hard anti-hallucination guard: every clause MUST have a
        # non-empty quote that actually occurs in the specification.
        # The decomposer prompt forbids unanchored clauses, but we
        # enforce it here defensively because a misbehaving tooling
        # model can still emit them. Clauses without an anchor are
        # almost always hallucinated preconditions, edge cases, or
        # implementation hints that are not in the text.
        if not quote_part:
            dropped_unanchored_lines.append(line)
            continue

        # Dedupe by paraphrased clause text (whitespace-normalised,
        # case-insensitive). The original key was (text, quote), but
        # that was too lenient: a single behavioural rule restated in
        # two places of the spec produced two clause records with the
        # same paraphrase but different anchor substrings (e.g.,
        # "S consisting of lowercase English letters" in the
        # description and "consisting of lowercase English letters"
        # in the constraints block). The mutator could not
        # differentiate the two anchors and emitted byte-identical
        # mutated specs, which inflated the matrix with redundant
        # rows that the per-clause dedup in run.py could not catch
        # because the clause indices differ. Text-only dedup makes
        # one paraphrased rule = one clause regardless of how many
        # times it appears verbatim in the spec.
        dedup_key = _normalize_whitespace(clause_part).lower()
        if dedup_key in seen_keys:
            logger.debug(
                f"Decomposer emitted a duplicate paraphrased clause "
                f"({clause_part[:60]!r}); dropping the second "
                f"occurrence (anchor: {quote_part!r})"
            )
            continue
        seen_keys.add(dedup_key)

        clauses.append(Clause(
            text=clause_part,
            quote=quote_part,
            type_tag=type_part,
            applicable_ops=applicable,
        ))

    clauses, dropped_duplicate_anchor_lines = _collapse_duplicate_anchor_clauses(
        clauses
    )

    return _ParsedDecomposition(
        clauses=clauses,
        numbered_lines=numbered_lines,
        malformed_lines=malformed_lines,
        dropped_unanchored_lines=dropped_unanchored_lines,
        unsafe_quote_lines=unsafe_quote_lines,
        dropped_duplicate_anchor_lines=dropped_duplicate_anchor_lines,
    )


def _has_constraints_section(spec: str) -> bool:
    marker = re.search(r"(?im)^Constraints\s*$", spec)
    if not marker:
        return False
    tail = spec[marker.end():]
    return bool(re.search(r"(\\le|\\ge|<=|>=|<|>|\d)", tail))


def _has_objective_or_return_clause(parsed: _ParsedDecomposition) -> bool:
    for clause in parsed.clauses:
        if clause.type_tag == "return_value":
            return True
        text = clause.text.lower()
        if any(word in text for word in (
            "find ", "print ", "output ", "return ", "answer",
            "minimum", "maximum", "minimize", "maximize",
        )):
            return True
    return False


def _minimum_expected_clause_count(spec: str) -> int:
    spec_len = len(_normalize_whitespace(spec))
    if spec_len >= 2500:
        return 8
    if spec_len >= 1500:
        return 5
    return 1


def _decomposition_rejection_reason(
    spec: str, parsed: _ParsedDecomposition
) -> str | None:
    """Return a retry reason when the decomposition looks incomplete."""
    if not parsed.clauses:
        return "no_valid_clauses"

    min_expected = _minimum_expected_clause_count(spec)

    # A few malformed or dropped sample/input-format rows are not worth
    # a retry when the usable decomposition is already large enough.
    # Retrying those cases was dominating batch time without improving
    # the clauses that mutation actually targets. Still retry severe
    # truncation, because that is how long specs become artificially
    # tiny in the visualizer and downstream matrix.
    if parsed.malformed_lines:
        severe_malformed = (
            len(parsed.malformed_lines)
            > max(2, parsed.numbered_lines // 5)
        )
        if len(parsed.clauses) < min_expected or severe_malformed:
            return "malformed_numbered_line"

    if parsed.unsafe_quote_lines:
        severe_unsafe = (
            len(parsed.unsafe_quote_lines)
            > max(3, parsed.numbered_lines // 3)
        )
        if len(parsed.clauses) < min_expected or severe_unsafe:
            return "unsafe_short_or_ambiguous_quote"

    spec_len = len(_normalize_whitespace(spec))
    n_clauses = len(parsed.clauses)
    if spec_len >= 2500 and n_clauses < 8:
        return "too_few_clauses_for_very_long_spec"
    if spec_len >= 1500 and n_clauses < 5:
        return "too_few_clauses_for_long_spec"

    if _has_constraints_section(spec) and not any(
        c.type_tag in {"boundary", "precondition", "ordering"}
        for c in parsed.clauses
    ):
        return "missing_constraints_clause"

    if (
        re.search(r"\b(Find|Print|Output|Return)\b", spec)
        and not _has_objective_or_return_clause(parsed)
    ):
        return "missing_objective_or_return_clause"

    return None


def _decomposition_score(parsed: _ParsedDecomposition) -> int:
    return (
        len(parsed.clauses) * 10
        - len(parsed.malformed_lines) * 3
        - len(parsed.dropped_unanchored_lines)
        - len(parsed.unsafe_quote_lines) * 2
        - len(parsed.dropped_duplicate_anchor_lines)
    )


def _retry_nudge(reason: str, parsed: _ParsedDecomposition) -> str:
    return (
        "\n\nIMPORTANT: the previous decomposition was rejected as "
        f"{reason}. It produced {len(parsed.clauses)} usable clauses "
        f"from {parsed.numbered_lines} numbered lines. Start over "
        "from 1 and emit a COMPLETE numbered list. Every numbered "
        "line must contain exactly four fields separated by `|||`: "
        "paraphrased clause, exact source quote, type, applicable "
        "operators. Do not stop mid-clause. For long programming "
        "problem statements, cover every explicitly written "
        "observable behaviour: core rules, objective/answer, scoring "
        "or cost formulas, input/output layout, numeric bounds, "
        "ordering constraints, and explicitly stated edge cases. "
        "Do not infer algorithmic strategy."
    )


def decompose(
    client: LLMClient, spec: str, model: str | None = None
) -> list[Clause]:
    """Decompose a specification into atomic, type-tagged clauses."""
    config = Config()
    template = (config.prompts_dir / "decompose.txt").read_text()
    base_user_message = template.replace("{spec}", spec)

    best: _ParsedDecomposition | None = None
    best_response = ""
    last_reason = ""
    for attempt in range(_DECOMPOSE_RETRY_MAX + 1):
        if attempt == 0:
            user_message = base_user_message
            salt = ""
        else:
            assert best is not None
            user_message = base_user_message + _retry_nudge(last_reason, best)
            salt = f"decompose-retry-{attempt}-{last_reason}"

        response = client.call(
            [{"role": "user", "content": user_message}],
            model=model,
            max_tokens=_DECOMPOSE_MAX_TOKENS,
            cache_salt=salt,
        )

        parsed = _parse_decomposition_response(response, spec)
        if best is None or _decomposition_score(parsed) > _decomposition_score(best):
            best = parsed
            best_response = response

        reason = _decomposition_rejection_reason(spec, parsed)
        if reason is None:
            return parsed.clauses

        last_reason = reason
        logger.info(
            "      Decomposition rejected (%s) on attempt %d/%d "
            "[usable=%d, numbered=%d, malformed=%d, "
            "dropped_unanchored=%d, unsafe_quote=%d, "
            "duplicate_anchor=%d]",
            reason,
            attempt + 1,
            _DECOMPOSE_RETRY_MAX + 1,
            len(parsed.clauses),
            parsed.numbered_lines,
            len(parsed.malformed_lines),
            len(parsed.dropped_unanchored_lines),
            len(parsed.unsafe_quote_lines),
            len(parsed.dropped_duplicate_anchor_lines),
        )

    if best and best.clauses:
        logger.warning(
            "      Using best incomplete decomposition after %d retries "
            "[usable=%d, numbered=%d, malformed=%d, "
            "dropped_unanchored=%d, unsafe_quote=%d, "
            "duplicate_anchor=%d]",
            _DECOMPOSE_RETRY_MAX,
            len(best.clauses),
            best.numbered_lines,
            len(best.malformed_lines),
            len(best.dropped_unanchored_lines),
            len(best.unsafe_quote_lines),
            len(best.dropped_duplicate_anchor_lines),
        )
        return best.clauses

    raise ValueError(f"Failed to parse clauses from response:\n{best_response}")
