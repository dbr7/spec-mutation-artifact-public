"""Spec mutation operators.

Each operator can be realised either by deterministic slot-style
rules or by an LLM-driven rewrite of the specification. The
deterministic rule tables live in ``classic_mutation_rules.py``;
the LLM path is parameterised by a fault category that names the
kind of edit the LLM should apply. The operators correspond to
documented categories of NL specification fault:

* ``weaken``     -- vagueness fault
* ``negate``     -- polarity fault
* ``perturb``    -- numeric off-by-one fault
* ``example``    -- I/O example tampering
* ``substitute`` -- concept substitution (replaces the older
                    mechanical ``swap`` operator and its hand-curated
                    slot vocabularies)

The single LLM dispatch path is ``_apply``, which fills the
operator's prompt template, calls the tooling LLM, runs
chain-of-thought recovery on the response, and returns the rewritten
specification. The operator's prompt is responsible for:

* Forcing a specific fault category so RQ3 can attribute kill
  rates to fault types.
* Demanding internal consistency: if the seed edit conflicts with
  another clause, an I/O example, or a stated constraint, the LLM
  is told to propagate the change to those parts as well.
* Surfacing an UNAPPLICABLE escape hatch when no meaningful edit
  is possible, so the calling pipeline can skip the candidate
  rather than treat a degenerate response as a real mutation.
"""

import difflib
import json
import logging
import re

from specmut.mutation.classic_mutation_rules import (
    CLASSIC_ANSWER_OBJECT_RE as _CLASSIC_ANSWER_OBJECT_RE,
    CLASSIC_ANSWER_TRANSFORM_PHRASES as _CLASSIC_ANSWER_TRANSFORM_PHRASES,
    CLASSIC_NUMERIC_LITERAL_RE as _CLASSIC_NUMERIC_LITERAL_RE,
    CLASSIC_PATH_ADJACENCY_FORMULA_RE as _CLASSIC_PATH_ADJACENCY_FORMULA_RE,
    CLASSIC_PATH_ADJACENCY_PHRASE as _CLASSIC_PATH_ADJACENCY_PHRASE,
    CLASSIC_PATH_ADJACENCY_RULES_BY_OPERATOR as _CLASSIC_PATH_ADJACENCY_RULES_BY_OPERATOR,
    CLASSIC_SEMANTIC_HEAD_RE as _CLASSIC_SEMANTIC_HEAD_RE,
    CLASSIC_SYMBOL_RULES as _CLASSIC_SYMBOL_RULES,
    CLASSIC_TWO_VALUE_DOMAIN_RE as _CLASSIC_TWO_VALUE_DOMAIN_RE,
    CLASSIC_WORD_RULES as _CLASSIC_WORD_RULES,
)
from specmut.config import Config
from specmut.llm import LLMClient

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"\S+|\s+")


def _short_diff_summary(
    original: str, mutated: str, max_chars: int = 120
) -> str:
    """Return a single ``"before" -> "after"`` fragment summarising
    the biggest changed token region between ``original`` and
    ``mutated``. Used to surface prior operators' edits compactly in
    the priors block of a subsequent mutation prompt, instead of
    pasting the whole mutated_spec.
    """
    a = _TOKEN_RE.findall(original)
    b = _TOKEN_RE.findall(mutated)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    best: tuple[str, str] | None = None
    best_size = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a_text = "".join(a[i1:i2]).strip()
        b_text = "".join(b[j1:j2]).strip()
        size = len(a_text) + len(b_text)
        if size > best_size:
            best_size = size
            best = (a_text, b_text)
    if best is None:
        return "<no textual change>"
    a_text, b_text = best
    if len(a_text) > max_chars:
        a_text = a_text[: max_chars - 3].rstrip() + "..."
    if len(b_text) > max_chars:
        b_text = b_text[: max_chars - 3].rstrip() + "..."
    return f"\"{a_text}\" -> \"{b_text}\""


def _format_priors_block(
    priors: list[tuple[str, str]], original_spec: str
) -> str:
    """Render an in-prompt nudge listing earlier operators' mutations
    of the SAME clause as compact ``"before" -> "after"`` fragments.
    The block is empty when ``priors`` is empty so the prompt is
    unchanged on the first operator call for a clause.
    """
    if not priors:
        return ""
    lines = [
        "",
        "Earlier attempts have already produced these mutations of",
        "the SAME clause. Your output's mutated specification MUST be",
        "byte-different from all of them. If your only natural",
        "candidate edit matches one of these prior edits, output",
        "UNAPPLICABLE on a single line instead.",
        "",
    ]
    for op, ms in priors:
        lines.append(f"  - {op}: {_short_diff_summary(original_spec, ms)}")
    lines.append("")
    return "\n".join(lines)

# Tokens that almost always indicate the model is reasoning out loud
# rather than producing a clean spec. Used by the chain-of-thought
# recovery heuristic in _recover_from_chain_of_thought.
_META_LANGUAGE_MARKERS = (
    "wait", "however,", "therefore,", "thus,", "hence,",
    "note:", "final answer", "let me think", "let's think",
    "actually,", "but to ", "originally,", "original substring",
    "clause to remove", "we remove", "we must ", "we need to",
    "thus, the corrected", "thus, the final", "the corrected",
    "the final answer", "to clarify",
)

# Concision system prompt prepended to every mutator call. Borrowed
# in spirit from LLMorpheus's SystemPrompt-Generic. The role framing
# narrows the model into "produce the spec, nothing else" mode and
# repeats the most-violated parts of the per-operator contract so the
# instructions land before the operator-specific prompt is processed.
_SYSTEM_PROMPT = (
    "You are a careful specification-mutation tool. You output ONLY "
    "the requested artefact. Do NOT include commentary, drafts, "
    "examples of your own, chain-of-thought, parentheticals, "
    "self-corrections, or any meta-discussion. If the prompt asks "
    "for a fenced code block, output exactly one fenced code block "
    "and nothing else. Prefer a clean, concrete behaviour-changing "
    "edit whenever the target clause supports one. Output "
    "UNAPPLICABLE on a single line only when the requested mutation "
    "cannot be expressed as a clean specification edit."
)

_MUTATION_STRENGTH_CONTRACT = (
    "\n\nMUTATION TESTING STRENGTH CONTRACT:\n"
    "- This is not a paraphrasing task. The mutation must change the "
    "observable input-output relation, not merely wording.\n"
    "- Prefer edits that should make the original intended solution "
    "wrong for at least one ordinary input under the mutated spec.\n"
    "- Prefer semantic levers: answer condition, objective direction, "
    "operation rule, transition rule, ordering/adjacency, aggregation, "
    "counting target, modulo/base, tie-breaking, or edge-case rule.\n"
    "- Stated numeric bounds, type-domain constraints, allowed-value "
    "sets, and preconditions are valid mutation targets. Do not treat "
    "them as benchmark metadata.\n"
    "- Do NOT mutate only generic I/O layout, sample prose, or "
    "explanation text when the same original solution would still "
    "compute the right relation.\n"
    "- If the targeted clause has no reasonable behaviour-changing lever, "
    "output UNAPPLICABLE.\n"
)

_MUTATION_VARIANT_CONTRACT = (
    "\n\nMUTATION VARIANT CONTRACT:\n"
    "- You are producing candidate {variant_idx} of {variant_total} "
    "for the same clause and operator.\n"
    "- Choose a DIFFERENT semantic fault from earlier attempts on "
    "this clause. Do not repeat the same replacement with different "
    "wording.\n"
    "- Prefer another concrete fault point inside the anchor, or a "
    "different target value/relation/order/objective when the anchor "
    "contains several choices.\n"
    "- If no distinct behaviour-changing variant is available, output "
    "UNAPPLICABLE.\n"
)

# Maximum retry budget on contract-violation rejections. Each retry
# bypasses the LLM cache (via a salt) and appends a corrective nudge
# to the user message, so the LLM has both a fresh sample and a
# clearer instruction. Two retries is enough to recover from sampling
# noise without inflating cost on persistent contract violations.
_RETRY_ON_META_LEAK_MAX = 2

# Tighter marker set used for the post-fence rejection check. These
# are specifically LLM-reasoning artefacts that should never appear
# in a clean rewritten specification. Markers in the broader
# _META_LANGUAGE_MARKERS set above include words like "Note:" and
# "Thus," that LiveCodeBench problem statements use naturally; if
# we used the broader list here, every mutation of an LCB problem
# would be falsely rejected. The tighter list focuses on phrases
# that are LLM-only.
_INTRODUCED_META_MARKERS = (
    "Wait,",
    "Let me think",
    "Let's think",
    "let me reconsider",
    "Let me reconsider",
    "Correction:",
    "(Correction:",
    "(Note:",
    "Final answer",
    "Final spec:",
    "let's just leave",
    "Let's just leave",
    "the corrected spec",
    "the corrected version",
    "to clarify,",
)

_QUALITY_MAX_LENGTH_RATIO = 1.35
_QUALITY_MAX_LENGTH_GROWTH = 600
_QUALITY_MIN_LENGTH_RATIO = 0.55

_SPEC_SECTION_HEADINGS = (
    "Input",
    "Output",
    "Constraints",
)

_PROMPT_LEAK_MARKERS = (
    "Original specification:",
    "Clause to ",
    "Anchor quote:",
    "HARD CONTRACT",
    "Return your output",
    "UNAPPLICABLE on a single line",
    "Original:",
    "New:",
    "Propagation:",
    "If I change",
    "No, usually",
    "Update:",
)

_PATCH_JSON_LEAK_RE = re.compile(
    r'\{\s*"edits"\s*:|"\s*(?:target_text|replacement_text|'
    r'unapplicable)\s*"\s*:'
)

_PATCH_RESPONSE_INSTRUCTIONS = (
    "\nPATCH OUTPUT PROTOCOL — THIS OVERRIDES ANY EARLIER OUTPUT "
    "FORMAT INSTRUCTION IN THE OPERATOR PROMPT.\n"
    "Do NOT rewrite the whole specification. Return a JSON object "
    "inside one fenced json block and nothing else. The JSON object "
    "must have this form:\n"
    "```json\n"
    "{\"edits\":[{\"target_text\":\"EXACT text copied from the "
    "original specification\",\"replacement_text\":\"mutated "
    "replacement text\"}]}\n"
    "```\n"
    "Rules:\n"
    " - The first edit's target_text MUST be exactly the targeted "
    "substring shown above.\n"
    " - Additional edits are allowed only when needed to keep the "
    "specification coherent. Each additional target_text MUST also "
    "be an exact substring copied from the original specification.\n"
    " - Each replacement_text must be only replacement specification "
    "text. Do not include labels such as Original:, New:, "
    "Propagation:, commentary, markdown fences, or explanations.\n"
    " - Preserve all text outside the listed replacements exactly. "
    "The Python harness will apply the replacements.\n"
    " - If no meaningful mutation is possible, output exactly:\n"
    "```json\n"
    "{\"edits\":[],\"unapplicable\":true}\n"
    "```\n"
)

_PATCH_RESPONSE_INSTRUCTIONS_INCONSISTENT = (
    "\nPATCH OUTPUT PROTOCOL — THIS OVERRIDES ANY EARLIER OUTPUT "
    "FORMAT INSTRUCTION IN THE OPERATOR PROMPT.\n"
    "Do NOT rewrite the whole specification. Return a JSON object "
    "inside one fenced json block and nothing else. The JSON object "
    "must have this form:\n"
    "```json\n"
    "{\"edits\":[{\"target_text\":\"EXACT targeted substring copied "
    "from the original specification\",\"replacement_text\":\"mutated "
    "replacement text\"}]}\n"
    "```\n"
    "Rules:\n"
    " - The first edit's target_text MUST be exactly the targeted "
    "substring shown above.\n"
    " - For this deliberate-inconsistency operator, normally emit "
    "ONLY that first edit. Do not propagate dependent prose or "
    "sample updates unless this operator prompt explicitly asked "
    "you to mutate an example instead of prose.\n"
    " - Each replacement_text must be only replacement specification "
    "text. Do not include labels such as Original:, New:, "
    "Propagation:, commentary, markdown fences, or explanations.\n"
    " - Preserve all text outside the listed replacements exactly. "
    "The Python harness will apply the replacements.\n"
    " - If no meaningful mutation is possible, output exactly:\n"
    "```json\n"
    "{\"edits\":[],\"unapplicable\":true}\n"
    "```\n"
)


def _introduced_meta_marker(orig: str, mutated: str) -> str | None:
    """Return the first chain-of-thought marker that appears in
    ``mutated`` but not in ``orig``, or None when no introduced
    marker is found. Comparing against ``orig`` is essential
    because problem statements (especially competitive-programming
    benchmarks) use words like "Note that" and "Thus," in their
    natural prose; only markers that the LLM injected count as
    contract violations.
    """
    if not mutated:
        return None
    for marker in _INTRODUCED_META_MARKERS:
        if marker in mutated and marker not in orig:
            return marker
    return None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _line_count(text: str, needle: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() == needle)


def _sample_heading_count(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if re.fullmatch(r"Sample (Input|Output) \d+", line.strip())
    )


def _first_meaningful_lines(text: str, limit: int = 3) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 24:
            lines.append(stripped)
            if len(lines) >= limit:
                break
    return lines


def mutation_quality_rejection_reason(orig: str, mutated: str) -> str | None:
    """Return a reason when a rewrite looks like a malformed LLM artefact."""
    if not mutated or not mutated.strip():
        return "empty"

    if mutated.count("```") != orig.count("```"):
        return "code_fence_count_changed"
    if mutated.count("`") != orig.count("`"):
        return "backtick_count_changed"

    for marker in _PROMPT_LEAK_MARKERS:
        if marker in mutated and marker not in orig:
            return "prompt_leak"

    if (
        _PATCH_JSON_LEAK_RE.search(mutated)
        and not _PATCH_JSON_LEAK_RE.search(orig)
    ):
        return "patch_json_leak"

    norm_orig = _normalize_ws(orig)
    norm_mutated = _normalize_ws(mutated)
    if not norm_orig or not norm_mutated:
        return "empty_after_normalization"

    orig_len = len(norm_orig)
    mutated_len = len(norm_mutated)
    if (
        mutated_len > orig_len + _QUALITY_MAX_LENGTH_GROWTH
        and mutated_len / orig_len > _QUALITY_MAX_LENGTH_RATIO
    ):
        return "too_long"
    if mutated_len / orig_len < _QUALITY_MIN_LENGTH_RATIO:
        return "too_short"

    for line in _first_meaningful_lines(orig):
        if _line_count(mutated, line) > _line_count(orig, line):
            return "repeated_opening_line"

    for heading in _SPEC_SECTION_HEADINGS:
        if _line_count(mutated, heading) > _line_count(orig, heading):
            return f"duplicated_heading:{heading}"

    if _sample_heading_count(mutated) > _sample_heading_count(orig):
        return "duplicated_sample_heading"

    return None


# Operators in this set are rejected before code generation. The main
# experiment now enables every implemented operator; keep the hook for
# future ablations, but do not disable any operator by default.
_STRENGTH_DISABLED_OPERATORS: set[str] = set()
_LOW_IMPACT_CLAUSE_TYPES = {"example", "io_format"}
_HIGH_IMPACT_CLAUSE_TYPES = {
    "boundary",
    "core_behaviour",
    "edge_case",
    "ordering",
    "precondition",
    "return_value",
}
_GENERIC_OUTPUT_RE = re.compile(
    r"(?is)^\s*(?:[-*]\s*)?"
    r"(?:print|output|return)\s+(?:the\s+)?"
    r"(?:answer|result|value)\.?\s*$"
)
_SEMANTIC_LEVER_RE = re.compile(
    r"(?i)\b("
    r"yes|no|true|false|minimum|maximum|minimize|maximize|min|max|"
    r"sum|count|number|product|modulo|mod|remainder|xor|and|or|"
    r"exactly|at\s+least|at\s+most|less|greater|equal|different|"
    r"distinct|same|first|last|left|right|row|column|adjacent|"
    r"edge|corner|before|after|ascending|descending|sorted|"
    r"reachable|connected|path|distance|cost|score|operation|"
    r"choose|remove|insert|delete|replace|swap|reverse|rotate|"
    r"integer|string|length|uppercase|lowercase|achievable|instruction"
    r")\b|<=|>=|<|>|=|!=|\\le|\\ge"
)
_WEAKEN_MODAL_SOFTENING_RE = re.compile(
    r"(?i)\b("
    r"may|might|could|possibly|probably|typically|generally|usually|"
    r"not\s+necessarily|optionally|may\s+mean|may\s+indicate|"
    r"may\s+represent|may\s+correspond"
    r")\b"
)
_DEFINITIONAL_MAPPING_RE = re.compile(
    r"(?i)\b("
    r"means?|indicates?|denotes?|represents?|represented\s+by|"
    r"corresponds?\s+to|is\s+equal\s+to|are\s+equal\s+to|"
    r"is\s+called|symbol|character|cell|state|healthy|cavit(?:y|ies)|"
    r"does\s+not|do\s+not|cannot|never"
    r")\b"
)
_SAMPLE_PROSE_RE = re.compile(
    r"(?i)\b("
    r"sample\s+(?:input|output)\s+\d+|"
    r"in\s+sample\s+(?:input|output)\s+\d+|"
    r"sample\s+\d+|worked\s+example"
    r")\b"
)
_EXISTENCE_EDGE_CASE_RE = re.compile(
    r"(?i)\b("
    r"there\s+may\s+be\s+no|may\s+be\s+no|may\s+be\s+empty|"
    r"can\s+be\s+empty|possibly\s+empty"
    r")\b"
)
_NUMERIC_OR_COMPARISON_RE = re.compile(
    r"(?ix)"
    r"(?:<=|>=|!=|==|<|>|=|\\leq?|\\geq?|\\neq|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"no|at\s+least|at\s+most|less\s+than|greater\s+than|"
    r"equal(?:s)?|non-negative|positive|negative)\b|"
    r"\b\d+(?:\.\d+)?\b)"
)
_ARITHMETIC_OPERATOR_RE = re.compile(
    r"(?x)"
    r"(?:\\times|\\cdot|[*/]|"
    r"(?<=[A-Za-z0-9_}\]\)|])\s*\+\s*(?=[A-Za-z0-9_\\{\(\[|])|"
    r"(?:(?<=[0-9_}\]\)|])\s*-\s*|(?<=[A-Za-z])\s+-\s+)"
    r"(?=[A-Za-z0-9_\\{\(\[|]))"
)
_ARITHMETIC_CHANGE_FRAGMENT_RE = re.compile(
    r"(?x)(?:\\times|\\cdot|\\sum|\\prod|[*/]|(?:^|\s)[+\-](?:\s|$))"
)
_NUMERIC_LITERAL_RE = re.compile(
    r"(?i)\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten)\b"
)
_OUTPUT_AFFECTING_RE = re.compile(
    r"(?i)\b("
    r"print|output|return|answer|result|find|determine|count|number|"
    r"maximum|minimum|maximize|minimize|max|min|sum|product|"
    r"mod|modulo|remainder|quotient|divide|divided|"
    r"integer|formed|arrang\w*|order|"
    r"hundreds|tens|ones|digit|denotes?|means?|represents?|"
    r"yes|no|true|false|possible|impossible|"
    r"exactly|at\s+least|at\s+most|less\s+than|greater\s+than|"
    r"same|different|ascending|descending|first|last|left|right|"
    r"healthy|cavit(?:y|ies)|empty|blocked|reachable|connected|"
    r"adjacent|distance|operation"
    r")\b|<=|>=|<|>|=|!=|\\le|\\ge"
)
_WEAKEN_UNDERSPECIFIED_ANSWER_RE = re.compile(
    r"(?i)\b("
    r"a|any|some|one\s+of|arbitrary|valid|acceptable|possible"
    r")\b"
)
_OBJECTIVE_OR_UNIQUE_ANSWER_RE = re.compile(
    r"(?i)\b("
    r"maximum|minimum|maximize|minimize|max|min|exactly|the\s+answer|"
    r"this\s+order|the\s+number|the\s+value|unique"
    r")\b"
)
_WEAKEN_PARAPHRASE_VERB_RE = re.compile(
    r"(?i)\b(process|handle|consider|deal\s+with)\b"
)
_ANSWER_VERB_RE = re.compile(r"(?i)\b(answer|determine|find|print|output)\b")
_REPRESENTATION_OR_FORMAT_RE = re.compile(
    r"(?i)\b("
    r"input\s+format|output\s+format|following\s+format|"
    r"standard\s+input|input\s+is\s+given"
    r")\b"
)
_REPRESENTATION_SUBSTITUTE_RE = re.compile(
    r"(?is)\b("
    r"input\s+is\s+given|standard\s+input|input\s+format|"
    r"output\s+format|following\s+format"
    r")\b"
)
_MODAL_OR_POLARITY_CHANGE_RE = re.compile(
    r"(?i)\b("
    r"must|must\s+not|may|may\s+not|can|cannot|can't|"
    r"should|should\s+not|is\s+not|are\s+not|not"
    r")\b"
)
_INTRODUCED_CASE_TRANSFORM_RE = re.compile(
    r"(?i)\b("
    r"invert(?:ing)?\s+case|case[-\s]?inversion|toggle\s+case|"
    r"swap\s+case|uppercase\s+to\s+lowercase|lowercase\s+to\s+uppercase"
    r")\b"
)
_CASE_CONSTRAINT_RE = re.compile(r"(?i)\b(?:uppercase|lowercase)\b")
_STATE_DOMAIN_ONLY_RE = re.compile(
    r"(?i)\b("
    r"(?:each|every)\s+(?:square|cell)\s+is\s+either\s+empty\s+or\s+"
    r"(?:has|contains)\s+(?:a\s+)?(?:piece|token|marker)|"
    r"(?:each|every)\s+(?:square|cell)\s+(?:has|contains)\s+"
    r"(?:a\s+)?(?:piece|token|marker)|"
    r"multiple\s+(?:pieces|tokens|markers)\s+placed"
    r")\b"
)
_SYMBOL_DEFINITION_RE = re.compile(
    r"(?is)\bif\s+(?:(?:the\s+)?(?:i-?th\s+)?character\s+of\s+\w+|it)\s+is\s+"
    r"([A-Z.#])\s*,?\s+it\s+means\s+that\s+([^.\n]+)"
)
_MODULO_SEMANTICS_RE = re.compile(
    r"(?i)\bmod(?:ulo)?\b|\\mathrm\{mod\}|\\mathbin\{\\mathrm\{mod\}\}"
)
_ANSWER_TRANSFORM_TARGET_RE = re.compile(
    r"(?is)\b(?:count|find|compute|determine|calculate|output|print|return)\b"
    r".{0,160}\b(?:answer|result|value|score|number|count|sum|total|"
    r"minimum|maximum|largest|smallest)\b|"
    r"\b(?:the\s+)?(?:number|count|sum|total|answer|result|value|score)\s+"
    r"(?:of|to|for)\b"
)
_ANSWER_TRANSFORM_CHANGE_RE = re.compile(
    r"(?i)\b(?:one\s+more\s+than|one\s+less\s+than|twice|double)\b"
)


_MULTI_VARIANT_OPERATORS = {
    "weaken",
    "negate",
    "perturb",
    "substitute",
}
_VARIANT_SEMANTIC_LEVER_RE = re.compile(
    r"(?i)\b("
    r"answer|result|find|determine|count|number|maximum|minimum|"
    r"maximize|minimize|max|min|sum|product|mod|modulo|remainder|"
    r"yes|no|possible|impossible|exactly|at\s+least|at\s+most|"
    r"less\s+than|greater\s+than|same|different|first|last|"
    r"left|right|ascending|descending|adjacent|reachable|connected|"
    r"distance|cost|score|operation|choose|remove|insert|delete|"
    r"replace|swap|reverse|rotate|integer|string|length|uppercase|"
    r"lowercase|letter|digit|either|or|and|achievable|instruction"
    r")\b|<=|>=|<|>|=|!=|\\le|\\ge"
)


def _has_perturbable_token(text: str) -> bool:
    return (
        _NUMERIC_OR_COMPARISON_RE.search(text or "") is not None
        or _ARITHMETIC_OPERATOR_RE.search(text or "") is not None
    )


def _has_perturbable_change(text: str) -> bool:
    return (
        _has_perturbable_token(text)
        or _ARITHMETIC_CHANGE_FRAGMENT_RE.search(text or "") is not None
    )


def mutation_variant_budget(
    clause: str,
    quote: str,
    clause_type: str | None,
    operator: str,
    requested: int,
) -> int:
    """Return how many variants are worth attempting for this target.

    Multiplicity is useful only when the anchor contains several
    plausible behavioural fault points. I/O format, examples, and
    prompt-layout clauses mostly inflate cost and tend to surface
    survivor-prone junk. Keep those at one candidate and spend the
    additional budget on semantic anchors, including stated input
    domain constraints.
    """
    requested = max(1, int(requested or 1))
    if requested == 1:
        return 1
    if operator not in _MULTI_VARIANT_OPERATORS:
        return 1

    clause_type = clause_type or "unknown"
    target_text = quote or clause
    if clause_type in _LOW_IMPACT_CLAUSE_TYPES:
        return 1
    if clause_type not in _HIGH_IMPACT_CLAUSE_TYPES:
        return 1
    if _SAMPLE_PROSE_RE.search(target_text) or _SAMPLE_PROSE_RE.search(clause):
        return 1
    if _GENERIC_OUTPUT_RE.match(target_text):
        return 1
    if _EXISTENCE_EDGE_CASE_RE.search(target_text):
        return 1
    if _STATE_DOMAIN_ONLY_RE.search(target_text):
        return 1
    if _REPRESENTATION_OR_FORMAT_RE.search(target_text):
        return 1

    if operator == "perturb" and not _has_perturbable_token(target_text):
        return 1
    if operator == "substitute" and _REPRESENTATION_SUBSTITUTE_RE.search(target_text):
        return 1
    if not _VARIANT_SEMANTIC_LEVER_RE.search(target_text):
        return 1
    return requested




def _case_preserve(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _classic_apply_replacement(
    spec: str, quote: str, replacement: str, operator: str
) -> str | None:
    if not quote or quote not in spec:
        return None
    if _normalize_ws(quote) == _normalize_ws(replacement):
        return None
    if _should_propagate_duplicate_anchor(
        operator, quote, quote, replacement, spec
    ):
        mutated = spec.replace(quote, replacement)
    else:
        mutated = spec.replace(quote, replacement, 1)
    if _normalize_ws(mutated) == _normalize_ws(spec):
        return None
    return mutated


def _classic_resolve_quote_anchor(spec: str, quote: str) -> str | None:
    """Return the exact spec slice for a decomposed quote.

    The decomposer often normalises display math whitespace, turning
    a multi-line formula into a single-line quote. Classic mutations
    need an exact slice so replacements preserve the rest of the spec
    byte-for-byte. Use a unique whitespace-tolerant match when the
    quote is not already an exact substring.
    """
    if not quote:
        return None
    if quote in spec:
        return quote
    stripped = quote.strip()
    if stripped and stripped in spec:
        return stripped
    parts = [part for part in re.split(r"\s+", stripped) if part]
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(part) for part in parts)
    matches = list(re.finditer(pattern, spec or ""))
    if len(matches) != 1:
        return None
    start, end = matches[0].span()
    return spec[start:end]


def _classic_match_is_leading_scaffold(text: str, start: int) -> bool:
    """True for numeric/comparison tokens before the behavioural head.

    Clauses often start with coordinate/index domains such as
    ``Square (i,j) (1<=i<=8,1<=j<=8) is empty if ...``. Classic
    comparison/numeric operators should mutate the behavioural part
    (``is empty if``), not the leading domain scaffold.
    """
    match = _CLASSIC_SEMANTIC_HEAD_RE.search(text or "")
    return match is not None and start < match.start()


def _classic_match_is_quantifier_scaffold(text: str, start: int) -> bool:
    """True for comparison/numeric tokens inside local quantifier ranges.

    Mutation-testing style operators should hit the behaviour formula,
    not bookkeeping ranges such as ``for each 0 <= k < l <= K``. Those
    edits tend to produce vacuous or malformed specifications while
    inflating the mutant count.
    """
    prefix = (text or "")[:start].lower()
    last_quantifier = max(
        prefix.rfind("for each"),
        prefix.rfind("for every"),
        prefix.rfind("for all"),
        prefix.rfind("for some"),
    )
    if last_quantifier < 0:
        return False
    last_boundary = max(
        prefix.rfind("."),
        prefix.rfind(";"),
        prefix.rfind("\n"),
        prefix.rfind(","),
        prefix.rfind(":"),
    )
    return last_quantifier > last_boundary


def _classic_match_is_latex_subscript(text: str, start: int) -> bool:
    """True when a token is inside ``_{...}`` subscript syntax.

    Edits such as ``i_{k+1}`` -> ``i_{k+2}`` often create references
    outside the defined index range. The behaviour-changing formula
    token is usually outside the subscript, so skip subscript internals.
    """
    prefix = (text or "")[:start]
    last_subscript = prefix.rfind("_{")
    if last_subscript < 0:
        return False
    last_close = prefix.rfind("}")
    return last_subscript > last_close


def _classic_add_candidate(
    candidates: list[dict[str, str]],
    seen: set[str],
    spec: str,
    quote: str,
    replacement: str,
    operator: str,
    slot_type: str,
    original_token: str,
    target_token: str,
    limit: int,
) -> None:
    if len(candidates) >= limit:
        return
    mutated = _classic_apply_replacement(spec, quote, replacement, operator)
    if mutated is None:
        return
    key = _normalize_ws(mutated)
    if key in seen:
        return
    seen.add(key)
    candidates.append({
        "mutated_spec": mutated,
        "slot_type": f"classic_{slot_type}",
        "slot_original": original_token,
        "slot_target": target_token,
    })


def _classic_add_raw_candidate(
    candidates: list[dict[str, str]],
    seen: set[str],
    mutated: str | None,
    slot_type: str,
    original_token: str,
    target_token: str,
    limit: int,
) -> None:
    if len(candidates) >= limit or mutated is None:
        return
    key = _normalize_ws(mutated)
    if key in seen:
        return
    seen.add(key)
    candidates.append({
        "mutated_spec": mutated,
        "slot_type": f"classic_{slot_type}",
        "slot_original": original_token,
        "slot_target": target_token,
    })


def _classic_path_formula_replacement(text: str, relation: str) -> str:
    return re.sub(r"=\s*1\s*$", lambda _m: relation, text)


def _classic_add_path_adjacency_candidates(
    candidates: list[dict[str, str]],
    seen: set[str],
    spec: str,
    quote: str,
    target_text: str,
    operator: str,
    limit: int,
) -> None:
    """Emit coherent path-distance mutants for redundant path specs.

    Many grid-walk statements define adjacency twice: informal prose
    says "adjacent cells (up, down, left, or right)" and a later bullet
    gives the Manhattan-distance formula. A one-location edit leaves a
    contradiction that generators resolve back to the original rule.
    These bundled candidates mutate both mentions together.
    """
    if len(candidates) >= limit:
        return
    if (
        _CLASSIC_PATH_ADJACENCY_PHRASE not in spec
        or _CLASSIC_PATH_ADJACENCY_FORMULA_RE.search(spec) is None
    ):
        return

    for relation, phrase_replacement, original_token, target_token in (
        _CLASSIC_PATH_ADJACENCY_RULES_BY_OPERATOR.get(operator, ())
    ):
        formula_match = _CLASSIC_PATH_ADJACENCY_FORMULA_RE.search(target_text)
        phrase_in_target = _CLASSIC_PATH_ADJACENCY_PHRASE in target_text
        if formula_match is None and not phrase_in_target:
            continue

        replacement = target_text
        if formula_match is not None:
            replacement = (
                replacement[:formula_match.start()]
                + _classic_path_formula_replacement(
                    formula_match.group(0), relation
                )
                + replacement[formula_match.end():]
            )
        if phrase_in_target:
            replacement = replacement.replace(
                _CLASSIC_PATH_ADJACENCY_PHRASE, phrase_replacement, 1
            )

        mutated = _classic_apply_replacement(
            spec, quote, replacement, operator
        )
        if mutated is None:
            continue

        dependent_edits = 0
        if _CLASSIC_PATH_ADJACENCY_PHRASE in mutated:
            mutated = mutated.replace(
                _CLASSIC_PATH_ADJACENCY_PHRASE, phrase_replacement, 1
            )
            dependent_edits += 1

        def repl(match: re.Match[str]) -> str:
            return _classic_path_formula_replacement(match.group(0), relation)

        mutated, n_formula = _CLASSIC_PATH_ADJACENCY_FORMULA_RE.subn(
            repl, mutated, count=1
        )
        dependent_edits += n_formula
        if dependent_edits == 0:
            continue

        _classic_add_raw_candidate(
            candidates, seen, mutated, "path_adjacency",
            original_token, target_token, limit,
        )


def _classic_add_answer_transform_candidates(
    candidates: list[dict[str, str]],
    seen: set[str],
    spec: str,
    quote: str,
    target_text: str,
    operator: str,
    limit: int,
) -> None:
    """Emit return-value style mutants for numeric/count objectives.

    These deliberately mutate the required answer instead of a
    precondition or wording detail. They are closer to classic return
    value replacement than prompt-level weakening and tend to be
    destructive when the generator follows the changed spec.
    """
    if operator != "perturb" or len(candidates) >= limit:
        return
    if not _ANSWER_TRANSFORM_TARGET_RE.search(target_text or ""):
        return
    if _ANSWER_TRANSFORM_CHANGE_RE.search(target_text or ""):
        return

    match = _CLASSIC_ANSWER_OBJECT_RE.search(target_text or "")
    if match is None:
        return

    for prefix, target_token in _CLASSIC_ANSWER_TRANSFORM_PHRASES:
        if len(candidates) >= limit:
            return
        replacement = (
            target_text[:match.start()]
            + prefix
            + target_text[match.start():]
        )
        mutated = _classic_apply_replacement(
            spec, quote, replacement, operator
        )
        _classic_add_raw_candidate(
            candidates, seen, mutated, "answer_transform",
            match.group(0), target_token, limit,
        )


def _classic_add_two_value_domain_candidates(
    candidates: list[dict[str, str]],
    seen: set[str],
    spec: str,
    quote: str,
    target_text: str,
    operator: str,
    limit: int,
) -> None:
    """Emit mutations for explicit two-value input domains.

    Clauses such as ``H_i is L or R`` are ordinary specification
    constraints in the SMT fault model. Mutating them is useful even
    when they came from a contest-style Constraints section: if no
    test detects the changed domain, that is a real coverage signal.
    """
    if operator not in {"negate", "substitute", "weaken"}:
        return
    for match in _CLASSIC_TWO_VALUE_DOMAIN_RE.finditer(target_text or ""):
        prefix, first, second = match.group(1), match.group(2), match.group(3)
        replacements: list[tuple[str, str]] = []
        if operator == "substitute":
            replacements = [
                (f"{prefix} {first}", first),
                (f"{prefix} {second}", second),
            ]
        elif operator == "negate":
            replacements = [
                (f"{prefix} neither {first} nor {second}", f"neither {first} nor {second}"),
            ]
        elif operator == "weaken":
            replacements = [
                (
                    f"{prefix} {first}, {second}, or another value",
                    f"{first}/{second}/other",
                ),
            ]

        for replacement_text, target_token in replacements:
            replacement = (
                target_text[:match.start()]
                + replacement_text
                + target_text[match.end():]
            )
            _classic_add_candidate(
                candidates, seen, spec, quote, replacement, operator,
                "allowed_domain", match.group(0), target_token, limit,
            )
            if len(candidates) >= limit:
                return


def _classic_add_numeric_perturb_candidates(
    candidates: list[dict[str, str]],
    seen: set[str],
    spec: str,
    quote: str,
    target_text: str,
    limit: int,
) -> None:
    for match in _CLASSIC_NUMERIC_LITERAL_RE.finditer(target_text):
        if _classic_match_is_leading_scaffold(target_text, match.start()):
            continue
        if _classic_match_is_quantifier_scaffold(target_text, match.start()):
            continue
        if _classic_match_is_latex_subscript(target_text, match.start()):
            continue
        value = int(match.group(0))
        replacement_values = [value + 1, value - 1]
        replacement_values = list(dict.fromkeys(replacement_values))
        for new_value in replacement_values:
            replacement_token = str(new_value)
            replacement = (
                target_text[:match.start()]
                + replacement_token
                + target_text[match.end():]
            )
            _classic_add_candidate(
                candidates, seen, spec, quote, replacement, "perturb",
                "numeric", match.group(0), replacement_token, limit,
            )
            if len(candidates) >= limit:
                return


def _classic_has_path_adjacency_dependency(spec: str, target_text: str) -> bool:
    if _CLASSIC_PATH_ADJACENCY_PHRASE not in spec:
        return False
    if _CLASSIC_PATH_ADJACENCY_FORMULA_RE.search(spec) is None:
        return False
    return (
        _CLASSIC_PATH_ADJACENCY_PHRASE in target_text
        or _CLASSIC_PATH_ADJACENCY_FORMULA_RE.search(target_text) is not None
    )


def _classic_match_is_direction_enumeration(text: str, start: int) -> bool:
    window = (text or "")[max(0, start - 24): start + 48].lower()
    return "up, down, left, or right" in window


def classic_mutation_candidates(
    spec: str,
    clause: str,
    quote: str,
    clause_type: str | None,
    operator: str,
    limit: int,
) -> list[dict[str, str]]:
    """Generate deterministic mutation candidates for common NL slots.

    This layer is deliberately closer to classic mutation-testing
    tooling than the LLM mutator: enumerate local syntactic
    replacements, then let the existing quality, impact, sample, and
    duplicate gates decide which candidates are usable. It increases
    mutant density without asking the tooling model to invent every
    variant from scratch.
    """
    limit = max(0, int(limit or 0))
    if limit <= 0:
        return []

    exact_quote = _classic_resolve_quote_anchor(spec, quote)
    if exact_quote is None:
        return []

    clause_type = clause_type or "unknown"
    if clause_type == "boundary":
        limit = min(limit, 8)
    elif clause_type == "precondition":
        limit = min(limit, 6)
    target_text = exact_quote
    if clause_type in _LOW_IMPACT_CLAUSE_TYPES:
        return []
    if _SAMPLE_PROSE_RE.search(target_text) or _SAMPLE_PROSE_RE.search(clause):
        return []
    generic_output = _GENERIC_OUTPUT_RE.match(target_text) is not None
    if generic_output and operator != "perturb":
        return []
    if _REPRESENTATION_OR_FORMAT_RE.search(target_text):
        return []
    if operator not in _MULTI_VARIANT_OPERATORS:
        return []
    if not _VARIANT_SEMANTIC_LEVER_RE.search(target_text):
        return []

    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    if operator == "perturb":
        _classic_add_answer_transform_candidates(
            candidates, seen, spec, exact_quote, target_text, operator, limit
        )
        if len(candidates) >= limit:
            return candidates
        if not _has_perturbable_token(target_text):
            return candidates
        if clause_type == "boundary":
            _classic_add_numeric_perturb_candidates(
                candidates, seen, spec, exact_quote, target_text, limit
            )
            return candidates

    if _classic_has_path_adjacency_dependency(spec, target_text):
        _classic_add_path_adjacency_candidates(
            candidates, seen, spec, exact_quote, target_text, operator, limit
        )
        return candidates

    _classic_add_two_value_domain_candidates(
        candidates, seen, spec, exact_quote, target_text, operator, limit
    )
    if len(candidates) >= limit:
        return candidates

    for slot_type, pattern, replacement_text in _CLASSIC_WORD_RULES.get(
        operator, ()
    ):
        for match in re.finditer(pattern, target_text, flags=re.IGNORECASE):
            if (
                slot_type == "direction"
                and _classic_match_is_direction_enumeration(
                    target_text, match.start()
                )
            ):
                continue
            replacement_token = _case_preserve(match.group(0), replacement_text)
            replacement = (
                target_text[:match.start()]
                + replacement_token
                + target_text[match.end():]
            )
            _classic_add_candidate(
                candidates, seen, spec, exact_quote, replacement, operator,
                slot_type, match.group(0), replacement_token, limit,
            )
            if len(candidates) >= limit:
                return candidates

    for slot_type, pattern, replacement_token in _CLASSIC_SYMBOL_RULES.get(
        operator, ()
    ):
        for match in re.finditer(pattern, target_text):
            if (
                slot_type in {"comparison", "arithmetic"}
                and _classic_match_is_leading_scaffold(
                    target_text, match.start()
                )
            ):
                continue
            if (
                slot_type in {"comparison", "arithmetic"}
                and _classic_match_is_quantifier_scaffold(
                    target_text, match.start()
                )
            ):
                continue
            if (
                slot_type in {"arithmetic", "numeric"}
                and _classic_match_is_latex_subscript(
                    target_text, match.start()
                )
            ):
                continue
            replacement = (
                target_text[:match.start()]
                + replacement_token
                + target_text[match.end():]
            )
            _classic_add_candidate(
                candidates, seen, spec, exact_quote, replacement, operator,
                slot_type, match.group(0), replacement_token, limit,
            )
            if len(candidates) >= limit:
                return candidates

    if operator == "perturb":
        _classic_add_numeric_perturb_candidates(
            candidates, seen, spec, exact_quote, target_text, limit
        )
        if len(candidates) >= limit:
            return candidates

    return candidates


def _regex_match_count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text or ""))


def _changed_text_parts(orig: str, mutated: str) -> tuple[str, str]:
    """Return deleted and inserted text fragments from a token diff."""
    deleted: list[str] = []
    inserted: list[str] = []
    a = _TOKEN_RE.findall(orig or "")
    b = _TOKEN_RE.findall(mutated or "")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"delete", "replace"}:
            deleted.append("".join(a[i1:i2]))
        if tag in {"insert", "replace"}:
            inserted.append("".join(b[j1:j2]))
    return " ".join(deleted), " ".join(inserted)


def _changed_text(orig: str, mutated: str) -> str:
    deleted, inserted = _changed_text_parts(orig, mutated)
    return f"{deleted} {inserted}"


def _target_token_count_changed(
    pattern: re.Pattern[str], target_text: str, orig: str, mutated: str
) -> bool:
    """True if a target token's full-spec occurrence count changes."""
    tokens = {m.group(0) for m in pattern.finditer(target_text or "")}
    return any((mutated or "").count(token) != (orig or "").count(token)
               for token in tokens)


def _has_duplicate_symbol_definitions(text: str) -> bool:
    """Detect one symbol being assigned two different prose meanings."""
    meanings_by_symbol: dict[str, set[str]] = {}
    for symbol, meaning in _SYMBOL_DEFINITION_RE.findall(text or ""):
        normalized = _normalize_ws(meaning).lower()
        meanings_by_symbol.setdefault(symbol, set()).add(normalized)
    return any(len(meanings) > 1 for meanings in meanings_by_symbol.values())


def mutation_impact_rejection_reason(
    orig: str,
    mutated: str,
    clause: str,
    quote: str,
    operator: str,
    clause_type: str | None,
) -> str | None:
    """Return a reason when a clean mutation is too weak for MT.

    The quality gate above asks whether the rewrite is well-formed.
    This gate asks whether it is likely to force a different program.
    It is deliberately conservative and syntactic: it rejects the
    survivor-prone classes we observed in LCB while avoiding any
    expensive semantic judgement or extra LLM call.
    """
    if operator in _STRENGTH_DISABLED_OPERATORS:
        return f"disabled_operator:{operator}"

    clause_type = clause_type or "unknown"
    if clause_type in _LOW_IMPACT_CLAUSE_TYPES and not (
        clause_type == "example" and operator == "example"
    ):
        return f"low_impact_clause_type:{clause_type}"

    target_text = quote or clause
    original_sample_region = _sample_region(orig)
    if (
        operator != "example"
        and (
            _SAMPLE_PROSE_RE.search(target_text)
            or _SAMPLE_PROSE_RE.search(clause)
        )
    ):
        return "sample_prose_clause"
    if (
        operator != "example"
        and
        quote
        and len(quote.strip()) > _ANCHOR_MIN_LEN
        and orig.count(quote) == 1
        and quote in original_sample_region
    ):
        return "sample_prose_clause"

    changed_text = _changed_text(orig, mutated)
    answer_transform = (
        operator == "perturb"
        and _ANSWER_TRANSFORM_TARGET_RE.search(target_text or "") is not None
        and _ANSWER_TRANSFORM_CHANGE_RE.search(changed_text or "") is not None
    )

    if _GENERIC_OUTPUT_RE.match(target_text) and not answer_transform:
        return "generic_output_rule"

    if _has_duplicate_symbol_definitions(mutated):
        return "contradictory_symbol_definitions"

    if operator == "perturb":
        if not _has_perturbable_token(target_text) and not answer_transform:
            return "invalid_perturb:no_numeric_or_comparison_target"
        if not _has_perturbable_change(changed_text) and not answer_transform:
            return "invalid_perturb:no_numeric_or_comparison_change"
        if _REPRESENTATION_OR_FORMAT_RE.search(target_text):
            return "weak_perturb:representation_or_format"

    if operator == "substitute":
        if _target_token_count_changed(
            _NUMERIC_LITERAL_RE, target_text, orig, mutated
        ):
            return "invalid_substitute:numeric_change"
        if (
            _REPRESENTATION_OR_FORMAT_RE.search(target_text)
            or _REPRESENTATION_SUBSTITUTE_RE.search(target_text)
        ):
            return "weak_substitute:representation_or_format"
        if _MODAL_OR_POLARITY_CHANGE_RE.search(changed_text):
            return "invalid_substitute:modal_or_polarity_change"
        if (
            _INTRODUCED_CASE_TRANSFORM_RE.search(mutated or "")
            and not _INTRODUCED_CASE_TRANSFORM_RE.search(orig or "")
            and _CASE_CONSTRAINT_RE.search(orig or "")
        ):
            return "invalid_substitute:unpropagated_case_constraint"

    if operator == "negate":
        if _target_token_count_changed(
            _NUMERIC_LITERAL_RE, target_text, orig, mutated
        ):
            return "invalid_negate:numeric_change"
        if _REPRESENTATION_OR_FORMAT_RE.search(target_text):
            return "weak_negate:representation_or_format"

    # Weakening a definition by adding uncertainty ("means" ->
    # "may indicate") is a prose ambiguity, not a useful mutation.
    # It rarely forces a different implementation because generators
    # continue to use the concrete mapping stated elsewhere.
    if (
        operator == "weaken"
        and _DEFINITIONAL_MAPPING_RE.search(target_text)
        and _regex_match_count(_WEAKEN_MODAL_SOFTENING_RE, mutated)
        > _regex_match_count(_WEAKEN_MODAL_SOFTENING_RE, orig)
    ):
        return "weak_weaken:modal_softening"

    if (
        operator == "weaken"
        and _DEFINITIONAL_MAPPING_RE.search(target_text)
        and not _NUMERIC_OR_COMPARISON_RE.search(changed_text)
    ):
        return "weak_weaken:definitional_ambiguity"

    if (
        operator == "weaken"
        and _OBJECTIVE_OR_UNIQUE_ANSWER_RE.search(target_text)
        and _WEAKEN_UNDERSPECIFIED_ANSWER_RE.search(changed_text)
    ):
        return "weak_weaken:underspecified_answer"

    if (
        operator == "weaken"
        and _ANSWER_VERB_RE.search(target_text)
        and _WEAKEN_PARAPHRASE_VERB_RE.search(changed_text)
        and not _NUMERIC_OR_COMPARISON_RE.search(changed_text)
    ):
        return "weak_weaken:paraphrase"

    if (
        operator == "weaken"
        and _REPRESENTATION_OR_FORMAT_RE.search(target_text)
    ):
        return "weak_weaken:representation_or_format"

    if (
        operator in {"weaken", "negate", "perturb", "substitute"}
        and original_sample_region
        and _MODULO_SEMANTICS_RE.search(target_text)
        and _MODULO_SEMANTICS_RE.search(original_sample_region)
        and _MODULO_SEMANTICS_RE.search(_sample_region(mutated))
    ):
        return "stale_samples:modulo_example_dependency"

    if (
        operator in {"weaken", "negate", "perturb", "substitute"}
        and original_sample_region
        and _sample_output_payloads(orig)
        and _sample_output_payloads(mutated) == _sample_output_payloads(orig)
        and _OUTPUT_AFFECTING_RE.search(target_text)
    ):
        return "stale_sample_outputs:output_affecting_change"

    if (
        operator in {"weaken", "negate", "perturb", "substitute"}
        and original_sample_region
        and _sample_region(mutated) == original_sample_region
        and _OUTPUT_AFFECTING_RE.search(target_text)
    ):
        return "stale_samples:output_affecting_change"

    if (
        operator == "example"
        and clause_type == "example"
        and _normalize_ws(orig) != _normalize_ws(mutated)
    ):
        return None

    if (
        clause_type not in _HIGH_IMPACT_CLAUSE_TYPES
        and not _SEMANTIC_LEVER_RE.search(target_text)
    ):
        return f"no_obvious_semantic_lever:{clause_type}"

    if _normalize_ws(orig) == _normalize_ws(mutated):
        return "noop"
    return None


# Anchors shorter than this are skipped by the anchor-modification
# check because short tokens may match incidentally elsewhere in the
# spec. Twelve characters comfortably excludes single-word identifiers
# while still catching short anchored phrases like
# ``"K=1"`` and ``"|S|<i\\leq|T|"`` once whitespace is normalised.
_ANCHOR_MIN_LEN = 12


def _anchor_unchanged(quote: str, orig: str, mutated: str) -> bool:
    """True iff the anchor count in ``mutated`` is greater than or
    equal to the count in ``orig``, meaning no occurrence of the
    anchor was edited. Excludes short anchors that may match
    incidentally; excludes the trivial empty-anchor case so the
    caller does not have to.
    """
    if not quote or len(quote) <= _ANCHOR_MIN_LEN:
        return False
    return mutated.count(quote) >= orig.count(quote)


def _looks_like_reasoning(text: str) -> bool:
    """True if ``text`` contains markers that strongly suggest
    chain-of-thought rather than a clean spec rewrite.
    """
    lower = text.lower()
    return any(marker in lower for marker in _META_LANGUAGE_MARKERS)


def _recover_from_chain_of_thought(
    raw: str, original_spec: str
) -> tuple[str, bool]:
    """Detect and recover from a mutator response that contains
    chain-of-thought commentary.

    The heuristic: if the response is more than 2x the original
    spec length OR contains meta-language markers, split the
    response into paragraphs and pick the LAST paragraph that
    (a) does not contain meta-language markers and (b) is at
    least half the length of the original spec.

    Returns ``(recovered_text, was_recovered)`` where
    ``was_recovered`` is True iff the heuristic actually picked a
    different block from the raw response.
    """
    if not raw or not original_spec:
        return raw, False

    too_long = len(raw) > 2.0 * len(original_spec)
    has_meta = _looks_like_reasoning(raw)
    if not too_long and not has_meta:
        return raw, False

    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    if not chunks:
        return raw, False

    min_len = max(20, len(original_spec) // 2)
    for chunk in reversed(chunks):
        if _looks_like_reasoning(chunk):
            continue
        if len(chunk) < min_len:
            continue
        if chunk.startswith(">>>") and chunk.count("\n") < 2:
            continue
        return chunk, True

    return raw, False


# All six operators. Each is a fault-category prompt; the pipeline
# treats them uniformly. The historical ``swap`` operator has been
# replaced by ``substitute``, which is also LLM-driven and avoids the
# brittleness of hand-curated slot vocabularies.
OPERATORS = [
    "weaken", "negate", "perturb", "example", "substitute",
    "partial_edit", "inconsistent_example",
]

# All operators dispatch through the LLM. The mechanical-vs-LLM
# distinction the older pipeline carried is gone.
LLM_OPERATORS = set(OPERATORS)
MECHANICAL_OPERATORS: set[str] = set()

# Operators whose intent is to create internal inconsistency in the
# mutated spec (mimicking real developer mistakes: incomplete edits
# that leave one clause out of sync, or rule changes whose examples
# weren't refreshed). These operators must NOT propagate coherence;
# the rest of the spec is deliberately left contradicting the
# mutated clause. They use a different anchor block in ``_apply``.
INCONSISTENCY_OPERATORS = {"partial_edit", "inconsistent_example"}

# Operators that are supposed to leave a coherent rewritten
# specification. The example operator deliberately tampers with sample
# I/O, so it is excluded along with the explicit inconsistency faults.
CONSISTENCY_REQUIRED_OPERATORS = (
    set(OPERATORS) - INCONSISTENCY_OPERATORS - {"example"}
)

_PROMPT_FILES = {
    "weaken": "weaken.txt",
    "negate": "negate.txt",
    "perturb": "perturb.txt",
    "example": "example.txt",
    "substitute": "substitute.txt",
    "partial_edit": "partial_edit.txt",
    "inconsistent_example": "inconsistent_example.txt",
}


# Sentinel string the LLM emits when no meaningful mutation is
# possible. Detected case-insensitively and with surrounding
# whitespace tolerance so a response of just ``"UNAPPLICABLE"`` (or
# ``"UNAPPLICABLE.\n"``) is recognised.
_UNAPPLICABLE_TOKEN = "UNAPPLICABLE"


def is_unapplicable(response: str) -> bool:
    """Return True if the LLM signalled that no meaningful mutation
    could be produced for this clause-operator pair."""
    if not response:
        return False
    s = response.strip().rstrip(".").strip().upper()
    return s == _UNAPPLICABLE_TOKEN


def _strip_response_wrappers(raw: str) -> str:
    """Return the model payload after removing common outer wrappers."""
    result = raw.strip()
    fence_matches = re.findall(
        r"```[a-zA-Z0-9_-]*\n?(.*?)```", result, re.DOTALL
    )
    if fence_matches:
        result = fence_matches[-1].strip("\n")

    if result.startswith('"""') and result.endswith('"""'):
        result = result[3:-3].strip()
    if result.startswith('"') and result.endswith('"'):
        try:
            decoded = json.loads(result)
        except Exception:
            decoded = result[1:-1]
        if isinstance(decoded, str):
            result = decoded.strip()
    return result


def _json_payload_candidates(raw: str) -> list[str]:
    """Return plausible JSON snippets from a mutator response."""
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(raw)
    for block in reversed(re.findall(
        r"```[a-zA-Z0-9_-]*\n?(.*?)```", raw, re.DOTALL
    )):
        add(block)
    for source in list(candidates) + [raw]:
        start = source.find("{")
        end = source.rfind("}")
        if 0 <= start < end:
            add(source[start:end + 1])
    return candidates


def _loads_json_prefix(candidate: str) -> object | None:
    """Parse JSON, tolerating harmless trailing whitespace/prose."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        pass
    try:
        obj, _idx = json.JSONDecoder().raw_decode(candidate)
        return obj
    except Exception:
        return None


def _first_string_value(obj: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return None


def _coerce_patch_edits(
    obj: object, quote: str
) -> tuple[list[dict[str, str]], bool]:
    """Coerce common patch JSON shapes into an edit list.

    Returns ``(edits, unapplicable)``. The tolerated aliases are
    intentionally narrow: enough to salvage slightly off-schema
    replies without treating arbitrary model prose as instructions.
    """
    if isinstance(obj, str) and is_unapplicable(obj):
        return [], True
    if isinstance(obj, list):
        raw_edits = obj
        explicit_unapplicable = False
    elif isinstance(obj, dict):
        explicit_unapplicable = obj.get("unapplicable") is True
        if isinstance(obj.get("edits"), list):
            raw_edits = obj["edits"]
        elif explicit_unapplicable:
            raw_edits = []
        else:
            raw_edits = [obj]
    else:
        return [], False

    edits: list[dict[str, str]] = []
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, dict):
            continue
        target = _first_string_value(raw_edit, (
            "target_text", "target", "original_text", "original",
            "find", "old_text",
        ))
        replacement = _first_string_value(raw_edit, (
            "replacement_text", "replacement", "new_text", "new",
            "replace_with", "mutated_text", "edited_text",
        ))
        if target is None and quote:
            target = quote
        if target is None or replacement is None:
            continue
        edits.append({
            "target_text": target,
            "replacement_text": replacement,
        })
    return edits, explicit_unapplicable


_PRINT_LITERAL_RE = re.compile(
    r"(?i)\bprint\s+("
    r"`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|-?\d+(?:\.\d+)?|yes|no|true|false"
    r")\b"
)


def _print_literals(text: str) -> set[str]:
    """Return simple literal outputs mentioned as ``print X``."""
    literals: set[str] = set()
    for match in _PRINT_LITERAL_RE.finditer(text):
        raw = match.group(1).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'`":
            raw = raw[1:-1]
        literals.add(raw)
    return literals


def _sample_region(text: str) -> str:
    """Return the sample/example region of a competitive-programming spec."""
    match = re.search(r"(?im)^Sample\s+(?:Input|Output)\s+\d+\b", text)
    if not match:
        return ""
    return text[match.start():]


def _sample_output_payloads(text: str) -> list[str]:
    """Return Sample Output payloads, excluding explanation prose."""
    payloads: list[str] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        if not re.fullmatch(r"\s*Sample\s+Output\s+\d+\s*", lines[i]):
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        payload: list[str] = []
        while i < len(lines) and lines[i].strip():
            payload.append(lines[i].strip())
            i += 1
        payloads.append("\n".join(payload))
    return payloads


def _sample_mentions_print_literal(sample_text: str, literal: str) -> bool:
    if not sample_text:
        return False
    escaped = re.escape(literal)
    return re.search(rf"(?i)\bprint\s+{escaped}\b", sample_text) is not None


def _sample_output_line_is_literal(sample_text: str, literal: str) -> bool:
    if not sample_text:
        return False
    escaped = re.escape(literal)
    pattern = (
        r"(?im)^Sample\s+Output\s+\d+\s*$"
        r"(?:\n[ \t]*)*"
        rf"\n?[ \t]*{escaped}[ \t]*(?:\n|$)"
    )
    return re.search(pattern, sample_text) is not None


def _coherence_dependency_failure(
    spec: str,
    mutated: str,
    quote: str,
    replacement: str,
    operator: str,
) -> str | None:
    """Return a consistency failure for obvious stale dependent text.

    This is intentionally conservative and deterministic. It catches
    the common LCB failure mode where a coherent operator changes a
    literal output rule such as ``print 0`` but leaves worked examples
    saying ``Sample Output ... 0`` or ``so print 0``.
    """
    if operator not in CONSISTENCY_REQUIRED_OPERATORS:
        return None
    if not quote or not replacement:
        return None

    samples = _sample_region(mutated)
    if (
        samples
        and re.search(r"(?i)\bprint\b", quote)
        and _normalize_ws(quote) != _normalize_ws(replacement)
        and samples == _sample_region(spec)
    ):
        return "coherence_unpropagated_sample_example"

    old_literals = _print_literals(quote)
    if not old_literals:
        return None
    new_literals = _print_literals(replacement)
    changed_literals = old_literals - new_literals
    if not changed_literals:
        return None

    for literal in changed_literals:
        if (
            _sample_mentions_print_literal(samples, literal)
            or _sample_output_line_is_literal(samples, literal)
        ):
            return "coherence_unpropagated_sample_example"
    return None


def _should_propagate_duplicate_anchor(
    operator: str, quote: str, target: str, replacement: str, current: str
) -> bool:
    """True when the first coherent patch should update duplicate anchors."""
    return (
        operator in CONSISTENCY_REQUIRED_OPERATORS
        and quote
        and _normalize_ws(target) == _normalize_ws(quote)
        and current.count(target) > 1
        and target not in replacement
    )


def _apply_patch_edits(
    spec: str, quote: str, edits: list[dict[str, str]], operator: str
) -> tuple[str | None, str | None]:
    """Apply exact JSON edits to ``spec`` and return a full mutation."""
    if not edits:
        return None, "patch_no_edits"

    mutated = spec
    auto_propagated: set[tuple[str, str]] = set()
    first_target = ""
    first_replacement = ""
    for idx, edit in enumerate(edits):
        target = edit["target_text"]
        replacement = edit["replacement_text"]
        if not target or not target.strip():
            return None, "patch_empty_target"
        if replacement is None or not replacement.strip():
            return None, "patch_empty_replacement"

        if quote and idx == 0 and target not in mutated:
            # The model sometimes normalises whitespace in JSON.
            # For the first edit, where the target must be the anchor,
            # use the exact anchor if the normalised texts match.
            if _normalize_ws(target) == _normalize_ws(quote):
                target = quote

        if target not in mutated:
            if (target, replacement) in auto_propagated:
                continue
            return None, "patch_target_not_found"
        if _normalize_ws(target) == _normalize_ws(replacement):
            return None, "patch_noop"
        if idx == 0:
            first_target = target
            first_replacement = replacement

        if _should_propagate_duplicate_anchor(
            operator, quote, target, replacement, mutated
        ):
            mutated = mutated.replace(target, replacement)
            auto_propagated.add((target, replacement))
        else:
            mutated = mutated.replace(target, replacement, 1)

    if quote and _anchor_unchanged(quote, spec, mutated):
        return None, "patch_anchor_unchanged"
    if _normalize_ws(mutated) == _normalize_ws(spec):
        return None, "patch_noop"
    dependency_failure = _coherence_dependency_failure(
        spec, mutated, first_target or quote, first_replacement, operator
    )
    if dependency_failure is not None:
        return None, dependency_failure
    return mutated, None


def _parse_and_apply_patch_response(
    raw: str, spec: str, quote: str, operator: str
) -> tuple[str | None, str | None]:
    """Parse a patch-protocol response and apply it to the original spec."""
    last_failure: str | None = None
    for candidate in _json_payload_candidates(raw):
        obj = _loads_json_prefix(candidate)
        if obj is None:
            continue
        edits, unapplicable = _coerce_patch_edits(obj, quote)
        if unapplicable and not edits:
            return _UNAPPLICABLE_TOKEN, None
        mutated, failure = _apply_patch_edits(spec, quote, edits, operator)
        if mutated is not None:
            return mutated, None
        last_failure = failure
    return None, last_failure


_LABELED_REPLACEMENT_RE = re.compile(
    r"(?is)(?:^|\n)\s*"
    r"(?:New|Replacement(?:\s+text)?|replacement_text|"
    r"Mutated(?:\s+clause)?|Edited(?:\s+clause)?)"
    r"\s*:\s*"
    r"(?P<value>.*?)"
    r"(?=\n\s*(?:Original|Target(?:\s+text)?|Propagation|"
    r"Explanation|Reasoning|Rationale|Update|Notes?)\s*:|\Z)"
)


def _salvage_labeled_replacement_response(
    raw: str, spec: str, quote: str, operator: str
) -> str | None:
    """Recover common off-schema ``Original:/New:`` patch replies."""
    if not quote:
        return None
    for match in _LABELED_REPLACEMENT_RE.finditer(raw):
        replacement = _strip_response_wrappers(match.group("value"))
        if is_unapplicable(replacement):
            return _UNAPPLICABLE_TOKEN
        if not replacement or _normalize_ws(replacement) == _normalize_ws(quote):
            continue

        # If the labeled value is a short clause, splice it into the
        # original; if it is already a full statement, let the normal
        # downstream quality gate decide whether it is usable.
        if len(_normalize_ws(replacement)) < 0.75 * len(_normalize_ws(spec)):
            if quote not in spec:
                continue
            if _should_propagate_duplicate_anchor(
                operator, quote, quote, replacement, spec
            ):
                mutated = spec.replace(quote, replacement)
            else:
                mutated = spec.replace(quote, replacement, 1)
            if _coherence_dependency_failure(
                spec, mutated, quote, replacement, operator
            ) is not None:
                continue
        else:
            mutated = replacement
        if quote and _anchor_unchanged(quote, spec, mutated):
            continue
        return mutated
    return None


def _apply(
    client: LLMClient,
    spec: str,
    clause: str,
    operator: str,
    model: str | None = None,
    quote: str = "",
    prior_mutated_specs: list[tuple[str, str]] | None = None,
    variant_idx: int = 1,
    variant_total: int = 1,
) -> str:
    """Run one operator on one clause and return the rewritten
    specification text. The operator's prompt owns the fault-category
    framing; this function only handles prompt assembly, the LLM
    call, and chain-of-thought recovery.

    The returned string may be the literal sentinel
    :data:`_UNAPPLICABLE_TOKEN` (caller checks via
    :func:`is_unapplicable`) when the LLM determined that no
    meaningful mutation was possible.
    """
    config = Config()
    template = (config.prompts_dir / _PROMPT_FILES[operator]).read_text()

    # If we have a quote span, inject an explicit anchor block so the
    # model knows which exact substring of the spec to edit. Otherwise
    # the anchor block is replaced with an empty string and the prompt
    # degrades to the operator's bare instruction. We also reiterate
    # the consistency expectation here: when we have an anchor, we
    # still want the LLM to consider the rest of the spec when
    # propagating the change.
    if quote and operator in INCONSISTENCY_OPERATORS:
        # Inverted contract: the mutated spec should be deliberately
        # internally inconsistent. We mimic real developer mistakes
        # (forgot to propagate an edit, didn't refresh an example),
        # so coherence repair is forbidden.
        if operator == "partial_edit":
            inconsistency_target = (
                "leave at least one OTHER prose clause (a constraint, "
                "return-value description, or rule statement elsewhere "
                "in the spec) UNCHANGED so that it now contradicts "
                "your edited clause"
            )
        else:  # inconsistent_example
            inconsistency_target = (
                "leave the sample I/O examples UNCHANGED so that they "
                "now reflect the OLD rule and contradict your edited "
                "clause"
            )
        anchor = (
            "\nThe clause corresponds to this EXACT substring of the "
            "original specification:\n\"\"\"\n" + quote + "\n\"\"\"\n"
            "\nHARD CONTRACT for this mutation (deliberate "
            "inconsistency):\n"
            " - The targeted substring above MUST be modified in your "
            "output to a semantically different statement (not a "
            "synonym, not cosmetic).\n"
            " - You MUST " + inconsistency_target + ". The whole "
            "point of this mutation is that the resulting spec is "
            "internally inconsistent — that is the developer mistake "
            "we are simulating.\n"
            " - Do NOT add notes, parentheticals, or any text "
            "acknowledging the inconsistency. The spec must read as "
            "if a developer simply forgot to propagate the edit.\n"
            " - If the targeted clause has no propagation surface "
            "(no other clause/example depends on it, so leaving them "
            "fixed produces no real inconsistency), return "
            "UNAPPLICABLE on a single line.\n"
            "\nReturn the COMPLETE rewritten specification, or "
            "UNAPPLICABLE.\n"
        ) + _PATCH_RESPONSE_INSTRUCTIONS_INCONSISTENT
    elif quote:
        anchor = (
            "\nThe clause corresponds to this EXACT substring of the "
            "original specification:\n\"\"\"\n" + quote + "\n\"\"\"\n"
            "\nHARD CONTRACT for this mutation:\n"
            " - The targeted substring above MUST be modified in your "
            "output. If it would appear verbatim in your output, no "
            "meaningful mutation was made — return UNAPPLICABLE on a "
            "single line.\n"
            " - Cosmetic-only changes (whitespace, punctuation, LaTeX "
            "delimiters such as \"$...$\", capitalisation, quoting "
            "style) do NOT count as a mutation. If your only candidate "
            "changes to the targeted substring are cosmetic, return "
            "UNAPPLICABLE.\n"
            " - Coherence edits to other clauses or examples are "
            "allowed ONLY IN ADDITION TO modifying the targeted "
            "substring, never instead of it.\n"
            " - The output must read as a clean specification: no "
            "parentheticals about your choices "
            "(\"(Correction: ...)\", \"(Note: ...)\"), no first-person "
            "reasoning (\"we treat\", \"let's use\"), no "
            "justifications for ambiguous values. If you would "
            "naturally explain a value, you have introduced "
            "reasoning — pick the simpler choice and stop.\n"
            " - If updating an example or constraint to remain "
            "consistent with the mutated clause requires inventing "
            "values whose meaning is genuinely underdetermined, "
            "prefer leaving that example unchanged over fabricating "
            "outputs with explanatory text. Do not give up merely "
            "because a small direct recomputation is needed.\n"
            " - Step-by-step explanation traces inside sample I/O "
            "blocks (per-iteration analyses, arithmetic "
            "walk-throughs, \"Therefore X\" summaries, "
            "dash-prefixed enumerations of cases) MUST NOT be "
            "regenerated. If your mutation changes the underlying "
            "rule, you may either DELETE the explanation block or "
            "LEAVE IT UNCHANGED. Never recompute it.\n"
            " - If you find yourself uncertain mid-output (any "
            "phrase like \"no, wait\", \"let me reconsider\", "
            "\"Correction:\", \"however...\"), STOP. Return "
            "UNAPPLICABLE on a single line. The presence of "
            "uncertainty in your output is a signal that the "
            "mutation is too disruptive for clean propagation.\n"
            "\nCRITICAL — PRESERVE EXACT WHITESPACE. The output must "
            "contain the same newlines, blank lines, and paragraph "
            "breaks as the original specification. Edit ONLY the "
            "substring your mutation requires; keep every other "
            "character byte-identical (newlines, indentation, blank "
            "lines).\n"
            "\nReturn your output as the LITERAL rewritten "
            "specification inside a fenced code block, exactly like "
            "this:\n"
            "```text\n"
            "<rewritten specification with original line breaks "
            "preserved>\n"
            "```\n"
            "\nThe code fence is not optional — output nothing before "
            "or after it. Treat the contents as verbatim text: every "
            "newline, blank line, and section break from the "
            "original must appear inside the fence at the same "
            "positions. If you cannot meaningfully edit, output "
            "exactly:\n"
            "```text\n"
            "UNAPPLICABLE\n"
            "```\n"
        ) + _PATCH_RESPONSE_INSTRUCTIONS
    else:
        anchor = ""

    priors_block = _format_priors_block(prior_mutated_specs or [], spec)

    base_user_message = (
        template
        .replace("{spec}", spec)
        .replace("{clause}", clause)
        .replace("{quote_anchor}", anchor + priors_block)
        + _MUTATION_STRENGTH_CONTRACT
    )
    if variant_total > 1:
        base_user_message += _MUTATION_VARIANT_CONTRACT.format(
            variant_idx=variant_idx,
            variant_total=variant_total,
        )

    # Retry loop on contract violations. The first iteration uses
    # the base user message and the cache as normal. Each subsequent
    # iteration appends a corrective nudge to the user message and
    # passes a per-attempt cache_salt so the cache lookup misses and
    # the LLM is sampled fresh. The loop exits as soon as a clean
    # rewrite is obtained or as soon as UNAPPLICABLE is returned.
    #
    # ``last_failure`` records the reason the most recent attempt
    # failed: ``None`` after a clean attempt, ``("meta", marker)``
    # when chain-of-thought leaked into the output, ``("anchor",
    # quote)`` when the targeted anchor substring was not modified,
    # or ``("quality", reason)`` when the output looked like a
    # malformed wrapper/paste artefact. The corrective nudge varies
    # by reason so the LLM is told what specifically went wrong.
    last_result: str | None = None
    last_failure: tuple[str, str] | None = None
    for attempt in range(_RETRY_ON_META_LEAK_MAX + 1):
        if attempt == 0:
            user_message = base_user_message
            salt = "" if variant_idx == 1 else f"variant-{variant_idx}"
        else:
            assert last_failure is not None
            reason, detail = last_failure
            if quote:
                retry_contract = (
                    "Output ONLY one fenced json block containing a "
                    "patch object of the form "
                    "{\"edits\":[{\"target_text\":\"...\","
                    "\"replacement_text\":\"...\"}]}. Do not output "
                    "the full rewritten specification."
                )
                retry_unapplicable = (
                    "```json\n"
                    "{\"edits\":[],\"unapplicable\":true}\n"
                    "```\n"
                )
            else:
                retry_contract = (
                    "Output ONLY the rewritten specification inside "
                    "the fenced block."
                )
                retry_unapplicable = "```text\nUNAPPLICABLE\n```\n"
            if reason == "meta":
                corrective = (
                    "\n\nIMPORTANT: a previous attempt at this "
                    "mutation leaked chain-of-thought into the "
                    f"output (a phrase like {detail!r}). "
                    f"{retry_contract} Do not include any meta-"
                    "commentary, drafts, or self-corrections. If "
                    "you cannot do that, output exactly:\n"
                    f"{retry_unapplicable}"
                )
            elif reason == "anchor":
                corrective = (
                    "\n\nIMPORTANT: a previous attempt at this "
                    "mutation did not modify the targeted anchor "
                    "substring (the anchor still appears verbatim "
                    "in your output). The anchor MUST be edited; "
                    "editing a different part of the spec instead "
                    "does not count as a mutation. "
                    f"{retry_contract} Either rewrite the anchor "
                    "substring or output exactly:\n"
                    f"{retry_unapplicable}"
                )
            elif reason == "coherence":
                corrective = (
                    "\n\nIMPORTANT: a previous attempt changed only "
                    f"part of the specification and left stale "
                    f"dependent text behind ({detail}). "
                    "For coherent operators, every exact duplicate "
                    "of the targeted rule and every dependent Output/"
                    "Sample Output/example explanation that follows "
                    "from the changed rule must be updated too. "
                    f"{retry_contract} Include those extra edits in "
                    "the same patch object. If you cannot recompute "
                    "the dependent sample text confidently, output "
                    "exactly:\n"
                    f"{retry_unapplicable}"
                )
            else:  # reason == "quality"
                corrective = (
                    "\n\nIMPORTANT: a previous attempt produced a "
                    f"malformed rewritten specification ({detail}). "
                    f"{retry_contract} Replacement text must not "
                    "contain ``` fences, duplicated Input/Output/"
                    "Constraints sections, duplicated Sample Input/"
                    "Output headings, prompt text, raw JSON patch "
                    "objects, or pasted copies of the original "
                    "statement. If you cannot "
                    "produce a single coherent mutation, output "
                    "exactly:\n"
                    f"{retry_unapplicable}"
                )
            user_message = base_user_message + corrective
            salt_prefix = f"variant-{variant_idx}-" if variant_idx != 1 else ""
            salt = f"{salt_prefix}contract-retry-{attempt}-{reason}"

        response = client.call(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=model,
            cache_salt=salt,
        )

        raw_result = response.strip()

        # Anchored clauses use a patch protocol: the model returns
        # exact target/replacement snippets and Python applies them
        # to the original spec. This sharply limits formatting
        # damage. For backwards compatibility and non-anchored
        # clauses, we still accept a full rewritten specification.
        result: str | None = None
        if quote:
            patched_result, patch_failure = _parse_and_apply_patch_response(
                raw_result, spec, quote, operator
            )
            if patched_result is not None:
                result = patched_result
                if not is_unapplicable(result):
                    logger.info("      Applied JSON patch response")
            elif patch_failure and patch_failure.startswith("coherence_"):
                last_result = raw_result
                last_failure = ("coherence", patch_failure)
                logger.info(
                    f"      Coherence contract failure {patch_failure!r} "
                    f"on attempt {attempt+1}/"
                    f"{_RETRY_ON_META_LEAK_MAX + 1}"
                )
                continue
            else:
                salvaged_result = _salvage_labeled_replacement_response(
                    raw_result, spec, quote, operator
                )
                if salvaged_result is not None:
                    result = salvaged_result
                    if not is_unapplicable(result):
                        logger.info("      Salvaged labeled replacement")
                elif patch_failure is not None:
                    logger.info(
                        f"      Patch response unusable ({patch_failure}); "
                        "falling back to full-spec parsing"
                    )

        if result is None:
            # The mutation prompt asks the model to wrap a
            # full rewritten spec in a fenced block to keep the
            # model in literal-content mode. Strip common wrappers
            # so downstream code sees the raw specification.
            result = _strip_response_wrappers(raw_result)

        # Detect the UNAPPLICABLE sentinel before chain-of-thought
        # recovery (which would otherwise discard a one-word
        # response as too short).
        if is_unapplicable(result):
            return _UNAPPLICABLE_TOKEN

        recovered, was_recovered = _recover_from_chain_of_thought(result, spec)
        if was_recovered:
            logger.info(
                f"      Recovered from chain-of-thought response "
                f"(attempt {attempt+1}): {len(result)} -> "
                f"{len(recovered)} chars"
            )
            result = recovered

        # Contract check 1: chain-of-thought marker introduced.
        introduced = _introduced_meta_marker(spec, result)
        if introduced is not None:
            last_result = result
            last_failure = ("meta", introduced)
            logger.info(
                f"      Chain-of-thought marker {introduced!r} "
                f"detected on attempt "
                f"{attempt+1}/{_RETRY_ON_META_LEAK_MAX + 1}"
            )
            continue

        # Contract check 2: anchor substring not modified. The
        # operator-specific prompts already include "the anchor
        # MUST be modified" in the contract block; this enforces
        # it in Python so a violation triggers a retry instead of
        # silently producing a same-as-original mutation. Note we
        # only check when there is a quote anchor of meaningful
        # length; clauses that decomposed without a verbatim
        # anchor pass through this check unchanged.
        if quote and _anchor_unchanged(quote, spec, result):
            last_result = result
            last_failure = ("anchor", quote[:50])
            logger.info(
                f"      Anchor not modified on attempt "
                f"{attempt+1}/{_RETRY_ON_META_LEAK_MAX + 1} "
                f"(anchor: {quote[:60]!r})"
            )
            continue

        # Contract check 3: coherent operators must not edit only one
        # copy of an exact duplicate anchor. The patch path
        # auto-propagates this when the replacement is known; this
        # catches full-spec replies and other fallback paths.
        if (
            operator in CONSISTENCY_REQUIRED_OPERATORS
            and quote
            and spec.count(quote) > 1
            and quote in result
        ):
            last_result = result
            last_failure = ("coherence", "coherence_unpropagated_duplicate_anchor")
            logger.info(
                f"      Coherence contract failure "
                f"'coherence_unpropagated_duplicate_anchor' on attempt "
                f"{attempt+1}/{_RETRY_ON_META_LEAK_MAX + 1}"
            )
            continue

        # Contract check 4: reject wrapper/paste artefacts before the
        # candidate reaches codegen. This catches failure modes that
        # still edit the anchor but produce an unusable spec, such as
        # leaking the requested code fence into the problem statement
        # or pasting an entire duplicate copy into the anchor span.
        quality_failure = mutation_quality_rejection_reason(spec, result)
        # Short fragment returns are repairable by run.py when a
        # quote anchor is available, so do not burn retry budget here
        # for that specific shape. The final run-level gate validates
        # the repaired full spec before codegen.
        if quality_failure == "too_short" and quote:
            quality_failure = None
        if quality_failure is not None:
            last_result = result
            last_failure = ("quality", quality_failure)
            logger.info(
                f"      Structural quality failure {quality_failure!r} "
                f"on attempt {attempt+1}/"
                f"{_RETRY_ON_META_LEAK_MAX + 1}"
            )
            continue

        # All checks passed: the rewrite is clean.
        last_result = result
        last_failure = None
        break

    if last_failure is not None:
        reason = last_failure[0]
        logger.info(
            f"      UNAPPLICABLE ({reason} contract violation "
            f"persisted after {_RETRY_ON_META_LEAK_MAX} retries)"
        )
        return _UNAPPLICABLE_TOKEN

    result = last_result or ""

    return result


def weaken(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
           variant_idx=1, variant_total=1):
    return _apply(
        client, spec, clause, "weaken", model, quote, prior_mutated_specs,
        variant_idx, variant_total,
    )


def negate(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
           variant_idx=1, variant_total=1):
    return _apply(
        client, spec, clause, "negate", model, quote, prior_mutated_specs,
        variant_idx, variant_total,
    )


def perturb(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
            variant_idx=1, variant_total=1):
    return _apply(
        client, spec, clause, "perturb", model, quote, prior_mutated_specs,
        variant_idx, variant_total,
    )


def example(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
            variant_idx=1, variant_total=1):
    return _apply(
        client, spec, clause, "example", model, quote, prior_mutated_specs,
        variant_idx, variant_total,
    )


def substitute(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
               variant_idx=1, variant_total=1):
    """Concept-substitution mutation. Replaces the historical
    mechanical ``swap`` operator. The LLM is asked to identify a
    domain concept inside the clause that has a semantically
    distinct alternative within the same conceptual category, apply
    the substitution, and propagate any consistency-breaking
    consequences across the rest of the spec."""
    return _apply(
        client, spec, clause, "substitute", model, quote, prior_mutated_specs,
        variant_idx, variant_total,
    )


def partial_edit(client, spec, clause, model=None, quote="", prior_mutated_specs=None,
                 variant_idx=1, variant_total=1):
    """Partial-edit fault. Mimics the developer mistake of editing
    one prose clause but forgetting to update other prose clauses
    that depend on it, leaving the spec internally inconsistent at
    the prose level. Uses the inverted anchor block in ``_apply``
    that forbids consistency repair."""
    return _apply(
        client, spec, clause, "partial_edit", model, quote,
        prior_mutated_specs, variant_idx, variant_total,
    )


def inconsistent_example(
    client, spec, clause, model=None, quote="", prior_mutated_specs=None,
    variant_idx=1, variant_total=1,
):
    """Inconsistent-example fault. Mimics the developer mistake of
    revising a rule in the description but never re-running or
    re-editing the worked examples, so the examples now reflect the
    OLD rule and contradict the new prose. Uses the inverted anchor
    block in ``_apply`` that forbids example refresh."""
    return _apply(
        client, spec, clause, "inconsistent_example", model, quote,
        prior_mutated_specs, variant_idx, variant_total,
    )
