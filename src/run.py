"""CLI entry point for spec mutation testing."""

import argparse
import difflib
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from specmut.benchmarks import load_benchmark
from specmut.codegen import generate, replace_docstring
from specmut.config import Config
from specmut.mutation.decompose import decompose
from specmut.execute import run_outputs, run_tests
from specmut.llm import LLMClient
from specmut.ref_cache import get_ref, save_ref
from specmut.matrix import MutationResult, build_matrix
from specmut.mutation.mutate import (
    weaken, negate, perturb, example, substitute,
    partial_edit, inconsistent_example,
    is_unapplicable,
    mutation_quality_rejection_reason,
    mutation_impact_rejection_reason,
    mutation_variant_budget,
    classic_mutation_candidates,
    _sample_output_payloads,
)

logger = logging.getLogger(__name__)

_OPERATOR_FNS = {
    "weaken": weaken,
    "negate": negate,
    "perturb": perturb,
    "example": example,
    "substitute": substitute,
    "partial_edit": partial_edit,
    "inconsistent_example": inconsistent_example,
}

_DEFAULT_OPERATOR_PRIORITY = (
    "perturb",
    "substitute",
    "negate",
    "weaken",
    "partial_edit",
    "inconsistent_example",
    "example",
)
_CLAUSE_TYPE_OPERATOR_PRIORITY = {
    "boundary": ("perturb", "weaken", "negate", "substitute"),
    "precondition": ("substitute", "negate", "weaken", "perturb"),
    "core_behaviour": ("perturb", "substitute", "negate", "weaken"),
    "return_value": ("perturb", "substitute", "negate", "weaken"),
    "ordering": ("substitute", "negate", "weaken", "perturb"),
    "edge_case": ("negate", "weaken", "substitute", "perturb"),
}

def _ordered_clause_operators(
    operators: list[str], clause_type: str | None
) -> list[str]:
    """Return operators in the preferred order for one clause type.

    The caller still applies structural applicability and candidate
    gates. This only decides which operators get first claim on a
    clause-level budget.
    """
    priority = _CLAUSE_TYPE_OPERATOR_PRIORITY.get(
        (clause_type or "unknown").lower(),
        _DEFAULT_OPERATOR_PRIORITY,
    )
    rank = {op: idx for idx, op in enumerate(priority)}
    fallback_rank = len(priority)
    original_order = {op: idx for idx, op in enumerate(operators)}
    return sorted(
        operators,
        key=lambda op: (
            rank.get(op, fallback_rank),
            original_order.get(op, len(original_order)),
        ),
    )


def _normalize_spec(text: str) -> str:
    """Collapse whitespace for comparing two specs for textual equality."""
    return re.sub(r"\s+", " ", text).strip()


# A mutation is considered a "fragment return" when the mutator's
# output is substantially shorter than the original spec, strongly
# suggesting that the LLM replied with just the edited phrase instead
# of the full rewritten specification. This constant is the length
# ratio below which repair is attempted.
_FRAGMENT_REPAIR_THRESHOLD = 0.6


_STALE_SAMPLE_REPAIR_REASONS = {
    "stale_samples:modulo_example_dependency",
    "stale_sample_outputs:output_affecting_change",
    "stale_samples:output_affecting_change",
}
_SAMPLE_OUTPUTS_ALREADY_CONSISTENT = "sample_outputs_already_consistent"

_SAMPLE_OUTPUT_HEADING_RE = re.compile(r"^\s*Sample\s+Output\s+\d+\s*$", re.I)
_BEHAVIOURAL_MUTATION_OPERATORS = {
    "weaken", "negate", "perturb", "substitute",
}
_BEHAVIOURAL_CLAUSE_TYPES = {
    "core_behaviour", "return_value", "ordering", "edge_case",
}
_MUTANT_CODEGEN_PREAMBLE = (
    "Mutation-testing instruction: this problem statement has been "
    "intentionally mutated. Implement the mutated statement exactly as "
    "written, even if it differs from a familiar contest problem. If a "
    "remaining sample, explanation, title-like summary, or high-level "
    "paraphrase conflicts with a formal bullet, equation, output rule, "
    "or changed condition, follow the formal/changed rule."
)


def _repair_fragment_mutation(
    original: str, mutated: str, quote: str
) -> tuple[str, bool]:
    """Detect and repair a fragment-return from the mutator.

    If the mutated spec looks like a truncated reply (substantially
    shorter than the original) and an anchor quote is available,
    rebuild the full mutated spec by substituting the fragment for
    the anchor in the original. Returns ``(repaired_spec, repaired)``
    where ``repaired`` is True if a substitution actually happened.
    """
    if not quote:
        return mutated, False
    norm_original = _normalize_spec(original)
    norm_mutated = _normalize_spec(mutated)
    if not norm_original or not norm_mutated:
        return mutated, False
    ratio = len(norm_mutated) / len(norm_original)
    if ratio >= _FRAGMENT_REPAIR_THRESHOLD:
        return mutated, False  # looks like a full rewrite, trust it

    # Only repair when we can cleanly splice.
    if quote in original:
        return original.replace(quote, mutated.strip(), 1), True
    stripped = quote.strip()
    if stripped and stripped in original:
        return original.replace(stripped, mutated.strip(), 1), True
    # Whitespace-tolerant fallback: find the quote in the
    # normalised original and splice using regex. Conservative: if
    # we cannot find a unique match, give up and return the raw
    # fragment so the bug is visible rather than silently patched.
    pattern = re.escape(stripped).replace(r"\ ", r"\s+")
    matches = list(re.finditer(pattern, original))
    if len(matches) == 1:
        start, end = matches[0].span()
        return original[:start] + mutated.strip() + original[end:], True
    return mutated, False


def _normalise_sample_output_payload(output) -> str:
    """Render an execution output as a Sample Output payload."""
    if output is None:
        return ""
    if not isinstance(output, str):
        output = str(output)
    return "\n".join(line.rstrip() for line in output.strip().splitlines())


def _replace_sample_output_payloads(spec: str, payloads: list[str]) -> str | None:
    """Replace Sample Output payload blocks with ``payloads``.

    The parser intentionally mirrors ``mutate._sample_output_payloads``:
    a Sample Output payload starts after the heading and any blank
    separator lines, then continues until the next blank line. If the
    number of headings does not exactly match ``payloads``, return
    ``None`` rather than guessing.
    """
    lines = (spec or "").splitlines(keepends=True)
    heading_count = sum(
        1 for line in lines
        if _SAMPLE_OUTPUT_HEADING_RE.fullmatch(line.rstrip("\r\n"))
    )
    if heading_count != len(payloads):
        return None

    out: list[str] = []
    i = 0
    payload_idx = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if not _SAMPLE_OUTPUT_HEADING_RE.fullmatch(line.rstrip("\r\n")):
            i += 1
            continue

        i += 1
        while i < len(lines) and not lines[i].strip():
            out.append(lines[i])
            i += 1

        while i < len(lines) and lines[i].strip():
            i += 1

        payload = payloads[payload_idx]
        payload_idx += 1
        if payload:
            for payload_line in payload.splitlines():
                out.append(payload_line.rstrip() + "\n")
        # Leave the blank line after the original payload for the
        # outer loop to append; it preserves explanation prose layout.

    if payload_idx != len(payloads):
        return None
    return "".join(out)


def _strip_sample_explanations(spec: str) -> str:
    """Drop worked-example prose while preserving sample I/O blocks."""
    lines = (spec or "").splitlines(keepends=True)
    out: list[str] = []
    i = 0
    sample_heading_re = re.compile(
        r"^\s*Sample\s+(?:Input|Output)\s+\d+\s*$", re.I
    )
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if not _SAMPLE_OUTPUT_HEADING_RE.fullmatch(line.rstrip("\r\n")):
            i += 1
            continue

        i += 1
        while i < len(lines) and not lines[i].strip():
            out.append(lines[i])
            i += 1
        while i < len(lines) and lines[i].strip():
            out.append(lines[i])
            i += 1
        if i < len(lines) and not lines[i].strip():
            out.append(lines[i])
            i += 1
        while i < len(lines) and not sample_heading_re.fullmatch(
            lines[i].rstrip("\r\n")
        ):
            i += 1
    return "".join(out)


def _strip_sample_region_for_codegen(spec: str) -> str:
    """Remove public samples before generating code for sample repair."""
    match = re.search(r"(?im)^Sample\s+Input\s+\d+\b", spec or "")
    if not match:
        return spec
    return spec[:match.start()].rstrip() + "\n"


def _should_strip_samples_for_mutant_codegen(
    operator: str, clause_type: str | None
) -> bool:
    return (
        operator in _BEHAVIOURAL_MUTATION_OPERATORS
        and (clause_type or "unknown") in _BEHAVIOURAL_CLAUSE_TYPES
    )


def _mutant_codegen_problem_text(spec: str) -> str:
    if (spec or "").startswith(_MUTANT_CODEGEN_PREAMBLE):
        return spec
    return _MUTANT_CODEGEN_PREAMBLE + "\n\n" + (spec or "")


def run_problem(
    client, problem, config, test_inputs=None, expected_outputs=None,
    run_id: str = "", generator_temperature: float | None = None,
):
    """Run the full pipeline for a single problem.

    If test_inputs and expected_outputs are provided, they override
    the problem's benchmark tests (used for LLM-generated test suite mode).

    ``run_id`` is mixed into the LLM cache key for the mutant
    codegen call only (not for ref generation, decompose, or
    mutate), so different run-ids on the same command produce
    different stochastic samples of mutant code while reusing all
    deterministic upstream work. ``generator_temperature``, when
    set, overrides the sampling temperature for the same call.
    """
    # Use custom tests if provided, otherwise use benchmark tests
    t_inputs = test_inputs if test_inputs is not None else problem.test_inputs
    t_expected = expected_outputs if expected_outputs is not None else problem.expected_outputs

    logger.info(f"=== {problem.task_id} ({len(t_inputs)} tests) ===")

    # Step 1: Generate reference program (or load from cache)
    # Stage 1: reference generation — Group G (generator).
    cached = get_ref(
        config,
        problem.task_id,
        config.model_generator,
        include_failed=not config.retry_failed_refs,
    )
    freshly_generated = False
    if cached:
        ref_code = cached["code"]
    else:
        logger.info("  Generating reference program...")
        ref_code = generate(
            client, problem.spec, problem.prompt, problem.entry_point,
            config.model_generator,
            mode=getattr(problem, "mode", "function"),
        )
        freshly_generated = True

    # Step 2: Verify reference passes the current test suite
    ref_results = run_tests(
        ref_code, problem.entry_point, t_inputs,
        t_expected, config.execution_timeout,
        mode=getattr(problem, "mode", "function"),
    )
    ref_passes = all(ref_results)

    # Persist the fresh ref with its actual pass/fail verdict so a
    # later run does not pay to regenerate a known-failing ref and
    # does not silently treat a failing ref as passing.
    if freshly_generated:
        save_ref(
            config, problem.task_id, config.model_generator,
            ref_code, ref_passes,
        )

    if not ref_passes:
        fail_count = sum(1 for r in ref_results if not r)
        logger.warning(f"  Reference program fails {fail_count}/{len(ref_results)} tests, skipping")
        return None

    # Stage 2: decomposition — Group T (tooling).
    logger.info("  Decomposing specification...")
    clause_records = decompose(client, problem.spec, config.model_tooling)
    num_anchored = sum(1 for c in clause_records if c.quote)
    logger.info(
        f"  Found {len(clause_records)} clauses ({num_anchored} quote-anchored)"
    )
    # Flat views kept for backward-compatible fields in the result dict.
    clauses = [c.text for c in clause_records]
    quotes = [c.quote for c in clause_records]
    clause_types = [c.type_tag for c in clause_records]
    clause_applicable_ops = [c.applicable_ops for c in clause_records]

    # Step 4: For each clause x operator: mutate, generate, test
    mutations = []
    num_skipped_structural = 0
    # Duplicates dropped at the dedup stage. Reported separately
    # from structural skips so the audit can see how often the
    # pipeline collapses operator collisions.
    num_skipped_duplicate = 0
    # Malformed LLM rewrites rejected before codegen. These are also
    # included in num_skipped_structural because they never become
    # matrix rows, but the separate breakdown makes quality failures
    # auditable.
    num_rejected_quality = 0
    quality_rejection_counts: dict[str, int] = {}
    # Well-formed but low-impact rewrites rejected before codegen.
    # These are also structural skips because they do not become
    # matrix rows, but are tracked separately from malformed-output
    # quality failures.
    num_rejected_impact = 0
    impact_rejection_counts: dict[str, int] = {}
    # Stale sample-output repairs that successfully propagated a
    # behaviour-changing mutation into the public Sample Output blocks
    # before the impact gate made its final decision.
    num_repaired_samples = 0
    sample_repair_counts: dict[str, int] = {}
    # Mutations whose text passed the gates but whose generated code
    # did not respond to the changed spec. These are not useful live
    # mutants; counting them as survivors hides generator noncompliance
    # as test-suite weakness.
    num_rejected_codegen = 0
    codegen_rejection_counts: dict[str, int] = {}
    # Deterministic "classic" candidates are generated without a
    # tooling-model call and then pass through the same gates as LLM
    # mutations. Track them separately so density gains are visible in
    # summaries and audits.
    num_classic_candidates = 0
    num_classic_accepted = 0
    # Clause-level budget controls. Operator-budget skips count
    # whole operators that would otherwise have work; candidate-budget
    # skips count concrete classic/LLM candidates not attempted after
    # a clause has already reached its mutant cap.
    num_operators_skipped_budget = 0
    num_mutation_candidates_skipped_budget = 0

    # Per-problem dedup index keyed on normalised mutated_spec.
    # The value records which (clause, operator) first produced that
    # mutated_spec and the raw text. Two roles:
    #
    # 1. Post-hoc dedup, both same-clause and cross-clause. Earlier
    #    iterations only deduped within a single clause_idx, but a
    #    common decomposer pathology emits the same paraphrased
    #    rule under two different anchors, leading two distinct
    #    clause_idxs to receive identical mutated specs. Keying the
    #    dedup on the mutated_spec rather than on the clause_idx
    #    catches both same-clause collisions (e.g., negate and
    #    substitute both reaching for "Yes" -> "No") and cross-
    #    clause collisions.
    #
    # 2. Prompt-side nudge. Before each subsequent mutator call on
    #    a given clause, the priors list is filtered to entries from
    #    the SAME clause and passed to the mutator so the LLM is
    #    told what earlier operators produced and instructed to
    #    differ. We deliberately do NOT show cross-clause priors in
    #    the nudge: those came from a different clause and are
    #    irrelevant to the current operator's targeting decision.
    seen_mutated_in_problem: dict[str, dict] = {}
    # Inner schema:
    #   ms_key -> {"clause_idx": int, "operator": str, "raw_ms": str}
    prior_candidate_specs_by_clause: dict[int, list[tuple[str, str]]] = {}
    mutants_per_operator = max(
        1, int(getattr(config, "mutants_per_operator", 1) or 1)
    )
    max_mutants_per_clause = max(
        0, int(getattr(config, "max_mutants_per_clause", 0) or 0)
    )
    max_operators_per_clause = max(
        0, int(getattr(config, "max_operators_per_clause", 0) or 0)
    )
    max_codegen_failures_per_clause = max(
        0, int(getattr(config, "max_codegen_failures_per_clause", 0) or 0)
    )
    clause_budget_stats = [
        {
            "max_mutants": max_mutants_per_clause,
            "max_operators": max_operators_per_clause,
            "max_codegen_failures": max_codegen_failures_per_clause,
            "mutants_recorded": 0,
            "codegen_failures": 0,
            "operators_used": [],
            "operators_skipped_budget": [],
            "candidate_skips_budget": 0,
            "mutant_budget_reached": False,
            "codegen_failure_budget_reached": False,
        }
        for _ in clause_records
    ]
    clause_mutation_diagnostics = [
        {
            "candidate_attempts": 0,
            "classic_candidates": 0,
            "llm_candidates": 0,
            "mutation_failures": 0,
            "evaluable": 0,
            "killed": 0,
            "survived": 0,
            "noop": 0,
            "skipped_structural": 0,
            "skipped_applicability": 0,
            "skipped_no_anchor": 0,
            "duplicates": 0,
            "rejected_quality": {},
            "rejected_impact": {},
            "rejected_codegen": {},
            "sample_repairs": {},
        }
        for _ in clause_records
    ]

    def _diag_inc(i: int, key: str, amount: int = 1) -> None:
        clause_mutation_diagnostics[i][key] = (
            clause_mutation_diagnostics[i].get(key, 0) + amount
        )

    def _diag_reason(i: int, group: str, reason: str) -> None:
        counts = clause_mutation_diagnostics[i].setdefault(group, {})
        counts[reason] = counts.get(reason, 0) + 1

    def _clause_mutant_budget_reached(i: int) -> bool:
        return (
            max_mutants_per_clause > 0
            and clause_budget_stats[i]["mutants_recorded"]
            >= max_mutants_per_clause
        )

    def _clause_codegen_failure_budget_reached(i: int) -> bool:
        return (
            max_codegen_failures_per_clause > 0
            and clause_budget_stats[i]["codegen_failures"]
            >= max_codegen_failures_per_clause
        )

    def _record_budgeted_mutation(i: int, status: str) -> None:
        if status == "evaluable":
            clause_budget_stats[i]["mutants_recorded"] += 1
            if _clause_mutant_budget_reached(i):
                clause_budget_stats[i]["mutant_budget_reached"] = True

    def _record_codegen_failure(i: int) -> None:
        clause_budget_stats[i]["codegen_failures"] += 1
        if _clause_codegen_failure_budget_reached(i):
            clause_budget_stats[i]["codegen_failure_budget_reached"] = True

    def _record_candidate_budget_skip(i: int, count: int = 1) -> None:
        nonlocal num_mutation_candidates_skipped_budget
        count = max(0, int(count or 0))
        if count <= 0:
            return
        num_mutation_candidates_skipped_budget += count
        clause_budget_stats[i]["candidate_skips_budget"] += count
        _diag_inc(i, "candidate_skips_budget", count)

    problem_mode = getattr(problem, "mode", "function")

    def _mutant_prompt_for(mutated_spec: str) -> str:
        if problem_mode == "stdin":
            # Stdin-mode problems carry the spec outside the prompt,
            # so the mutated spec flows directly into the codegen
            # template rather than being spliced into a function
            # docstring.
            return ""
        return replace_docstring(problem.prompt, mutated_spec)

    def _generate_code_from_spec(codegen_spec: str) -> str:
        codegen_spec = _mutant_codegen_problem_text(codegen_spec)
        return generate(
            client, codegen_spec, _mutant_prompt_for(codegen_spec),
            problem.entry_point, config.model_generator,
            mode=problem_mode,
            cache_salt=run_id,
            temperature=generator_temperature,
        )

    def _mutant_codegen_spec(
        mutated_spec: str, operator: str, clause_type: str | None
    ) -> tuple[str, str]:
        if _should_strip_samples_for_mutant_codegen(operator, clause_type):
            stripped = _strip_sample_region_for_codegen(mutated_spec)
            if stripped != mutated_spec:
                return stripped, "sample_stripped"
        return mutated_spec, "full"

    def _generate_mutant_code(
        mutated_spec: str, operator: str, clause_type: str | None
    ) -> tuple[str, str]:
        codegen_spec, source = _mutant_codegen_spec(
            mutated_spec, operator, clause_type
        )
        return _generate_code_from_spec(codegen_spec), source

    def _sample_repair_inputs(sample_count: int) -> list | None:
        pairs = problem.example_pairs or []
        if len(pairs) != sample_count:
            return None
        inputs = []
        for args, _expected in pairs:
            if problem_mode == "stdin":
                if isinstance(args, (list, tuple)) and len(args) == 1:
                    inputs.append(args[0])
                else:
                    inputs.append(args)
                continue
            if isinstance(args, (list, tuple)):
                inputs.append(list(args))
            else:
                inputs.append([args])
        return inputs

    def _try_repair_stale_sample_outputs(
        mutated_spec: str,
        impact_reason: str,
    ) -> tuple[str | None, str]:
        """Propagate a mutation into Sample Output payloads.

        This is intentionally narrow: it only runs when the impact gate
        already found stale outputs, the loader extracted exactly the
        same number of public examples as Sample Output blocks, and the
        generated mutant code produces different public-example outputs.
        """
        strip_explanations = impact_reason == (
            "stale_samples:modulo_example_dependency"
        )
        repair_base = (
            _strip_sample_explanations(mutated_spec)
            if strip_explanations
            else mutated_spec
        )
        original_payloads = _sample_output_payloads(problem.spec)
        mutated_payloads = _sample_output_payloads(repair_base)
        if not original_payloads or not mutated_payloads:
            return None, "no_sample_outputs"
        if len(original_payloads) != len(mutated_payloads):
            return None, "sample_output_count_changed"

        example_inputs = _sample_repair_inputs(len(mutated_payloads))
        if example_inputs is None:
            return None, "example_count_mismatch"

        try:
            sample_codegen_spec = _strip_sample_region_for_codegen(
                repair_base
            )
            sample_code = _generate_code_from_spec(sample_codegen_spec)
        except Exception as e:
            return None, f"codegen_failed:{type(e).__name__}"

        results = run_outputs(
            sample_code, problem.entry_point, example_inputs,
            config.execution_timeout, mode=problem_mode,
        )
        if len(results) != len(mutated_payloads):
            return None, "execution_count_mismatch"

        repaired_payloads: list[str] = []
        for status, output in results:
            if status != "ok":
                return None, f"sample_execution_{status}"
            repaired_payloads.append(_normalise_sample_output_payload(output))

        if repaired_payloads == mutated_payloads:
            if strip_explanations and repair_base != mutated_spec:
                return repair_base, "sample_explanations"
            return mutated_spec, _SAMPLE_OUTPUTS_ALREADY_CONSISTENT
        if repaired_payloads == original_payloads:
            return None, "sample_outputs_match_original"

        repaired = _replace_sample_output_payloads(
            repair_base, repaired_payloads
        )
        if repaired is None:
            return None, "replace_failed"
        if _sample_output_payloads(repaired) != repaired_payloads:
            return None, "replace_verification_failed"
        return repaired, "sample_outputs"
    # Mutations often prompt the generator into the same non-reference
    # program. Cache benchmark test results by normalised mutant code
    # so duplicate code is executed once per problem.
    test_results_by_code: dict[str, list[bool]] = {}

    def _process_mutated_spec(
        i: int, clause: str, op_name: str, mutated_spec: str,
        stale_samples_matched_original: bool = False,
        slot_type: str | None = None,
        slot_original: str | None = None,
        slot_target: str | None = None,
        variant_idx: int = 1,
        variant_total: int = 1,
    ) -> str:
        """Run codegen and benchmark-test execution on one mutated spec.

        Return ``"evaluable"`` for killed/survived mutants and
        ``"rejected"`` for failures that never become result rows.
        We intentionally do not special-case AST-identical or
        stale-sample survivors here: if the generated mutant program
        passes the benchmark test suite, it is a survived mutant.
        """
        nonlocal num_skipped_structural
        nonlocal num_rejected_codegen

        clause_type = clause_records[i].type_tag if i < len(clause_records) else None

        try:
            mut_code, codegen_source = _generate_mutant_code(
                mutated_spec, op_name, clause_type
            )
        except Exception as e:
            logger.warning(f"      Code generation failed: {e}")
            num_skipped_structural += 1
            num_rejected_codegen += 1
            _record_codegen_failure(i)
            _diag_inc(i, "skipped_structural")
            _diag_reason(i, "rejected_codegen", "codegen_failed")
            codegen_rejection_counts["codegen_failed"] = (
                codegen_rejection_counts.get("codegen_failed", 0) + 1
            )
            return "rejected"

        if codegen_source == "sample_stripped":
            logger.info(
                "      CODEGEN-SPEC stripped public samples for "
                "behavioural mutant"
            )

        # The reference program was already verified against
        # ``t_expected`` at the start of the problem, so running both
        # reference and mutant here is redundant. Execute the mutant
        # once, keep the full per-test vector for the matrix, and use
        # ``all(test_results)`` as the survival verdict.
        test_cache_key = mut_code
        cached_results = test_results_by_code.get(test_cache_key)
        if cached_results is None:
            test_results = run_tests(
                mut_code, problem.entry_point, t_inputs,
                t_expected, config.execution_timeout,
                mode=problem_mode,
            )
            test_results_by_code[test_cache_key] = test_results
        else:
            test_results = list(cached_results)
        survived_against_tests = all(test_results)

        if survived_against_tests:
            if stale_samples_matched_original:
                logger.info(
                    "      SURVIVED "
                    "(stale public samples matched generated behaviour)"
                )
            else:
                logger.info("      SURVIVED")
            _diag_inc(i, "evaluable")
            _diag_inc(i, "survived")
            mutations.append(MutationResult(
                clause_idx=i, clause_text=clause, operator=op_name,
                survived=True, mutated_spec=mutated_spec,
                mutant_code=mut_code,
                test_results=test_results,
                slot_type=slot_type,
                slot_original=slot_original,
                slot_target=slot_target,
                variant_idx=variant_idx,
                variant_total=variant_total,
            ))
            return "evaluable"

        fail_count = sum(1 for r in test_results if not r)
        logger.info(f"      KILLED ({fail_count}/{len(test_results)} tests failed)")
        _diag_inc(i, "evaluable")
        _diag_inc(i, "killed")
        mutations.append(MutationResult(
            clause_idx=i, clause_text=clause, operator=op_name,
            survived=False, mutated_spec=mutated_spec,
            mutant_code=mut_code,
            test_results=test_results,
            slot_type=slot_type,
            slot_original=slot_original,
            slot_target=slot_target,
            variant_idx=variant_idx,
            variant_total=variant_total,
        ))
        return "evaluable"

    def _handle_mutation_candidate(
        i: int,
        rec,
        clause: str,
        quote: str,
        op_name: str,
        candidate_label: str,
        mutated_spec: str,
        slot_type: str | None = None,
        slot_original: str | None = None,
        slot_target: str | None = None,
        variant_idx: int = 1,
        variant_total: int = 1,
    ) -> str:
        """Validate, deduplicate, codegen, and test one candidate.

        Returns ``"evaluable"`` for killed/survived/noop records,
        and ``"rejected"`` for candidates that never become result rows.
        """
        nonlocal num_skipped_structural
        nonlocal num_skipped_duplicate
        nonlocal num_rejected_quality
        nonlocal num_rejected_impact
        nonlocal num_repaired_samples

        if is_unapplicable(mutated_spec):
            logger.info(
                f"      UNAPPLICABLE (mutator could not produce "
                f"a meaningful {candidate_label} mutation for "
                f"this clause)"
            )
            num_skipped_structural += 1
            _diag_inc(i, "skipped_structural")
            _diag_reason(i, "rejected_quality", "unapplicable")
            return "rejected"

        # Detect fragment-return: when the mutator replies with just
        # the edited phrase instead of the full spec. When this happens
        # and we have an anchor, splice the fragment back into the
        # original spec so downstream codegen sees a complete docstring
        # rather than a one-line stub.
        mutated_spec, was_repaired = _repair_fragment_mutation(
            problem.spec, mutated_spec, quote
        )
        if was_repaired:
            logger.info(
                "      Repaired fragment-return: spliced "
                "fragment into spec"
            )

        quality_reason = mutation_quality_rejection_reason(
            problem.spec, mutated_spec
        )
        if quality_reason is not None:
            logger.info(f"      REJECTED-QUALITY ({quality_reason})")
            num_skipped_structural += 1
            num_rejected_quality += 1
            _diag_inc(i, "skipped_structural")
            _diag_reason(i, "rejected_quality", quality_reason)
            quality_rejection_counts[quality_reason] = (
                quality_rejection_counts.get(quality_reason, 0) + 1
            )
            return "rejected"

        impact_reason = mutation_impact_rejection_reason(
            problem.spec, mutated_spec, clause, quote, op_name, rec.type_tag,
        )
        stale_samples_matched_original = False
        if impact_reason in _STALE_SAMPLE_REPAIR_REASONS:
            repaired_spec, repair_note = (
                _try_repair_stale_sample_outputs(mutated_spec, impact_reason)
            )
            if repair_note == _SAMPLE_OUTPUTS_ALREADY_CONSISTENT:
                stale_samples_matched_original = True
                logger.info(
                    "      SAMPLES-OK "
                    f"({impact_reason}; public samples already "
                    "match generated mutant behaviour)"
                )
                impact_reason = None
            elif repaired_spec is not None:
                repaired_quality_reason = (
                    mutation_quality_rejection_reason(
                        problem.spec, repaired_spec
                    )
                )
                if repaired_quality_reason is not None:
                    logger.info(
                        "      REJECTED-QUALITY after sample "
                        f"repair ({repaired_quality_reason})"
                    )
                    num_skipped_structural += 1
                    num_rejected_quality += 1
                    _diag_inc(i, "skipped_structural")
                    _diag_reason(
                        i, "rejected_quality", repaired_quality_reason
                    )
                    quality_rejection_counts[repaired_quality_reason] = (
                        quality_rejection_counts.get(
                            repaired_quality_reason, 0
                        ) + 1
                    )
                    return "rejected"

                repaired_impact_reason = (
                    mutation_impact_rejection_reason(
                        problem.spec, repaired_spec, clause, quote,
                        op_name, rec.type_tag,
                    )
                )
                if repaired_impact_reason is None:
                    mutated_spec = repaired_spec
                    num_repaired_samples += 1
                    _diag_reason(i, "sample_repairs", impact_reason)
                    sample_repair_counts[impact_reason] = (
                        sample_repair_counts.get(impact_reason, 0) + 1
                    )
                    logger.info(
                        "      REPAIRED-SAMPLES "
                        f"({impact_reason} -> {repair_note})"
                    )
                    impact_reason = None
                else:
                    logger.info(
                        "      Sample repair did not satisfy impact gate "
                        f"({repair_note}; {repaired_impact_reason})"
                    )
                    impact_reason = repaired_impact_reason
            else:
                logger.info(
                    "      Sample repair unavailable "
                    f"({repair_note})"
                )
        if impact_reason is not None:
            prior_candidate_specs_by_clause.setdefault(i, []).append(
                (candidate_label, mutated_spec)
            )
            logger.info(f"      REJECTED-IMPACT ({impact_reason})")
            num_skipped_structural += 1
            num_rejected_impact += 1
            _diag_inc(i, "skipped_structural")
            _diag_reason(i, "rejected_impact", impact_reason)
            impact_rejection_counts[impact_reason] = (
                impact_rejection_counts.get(impact_reason, 0) + 1
            )
            return "rejected"

        # Detect noops: the mutator returned the spec unchanged (after
        # whitespace normalization). Record it as a survived-by-
        # construction mutant and skip codegen to save API calls.
        if _normalize_spec(mutated_spec) == _normalize_spec(problem.spec):
            logger.info("      NOOP (mutator returned unchanged spec)")
            _diag_inc(i, "evaluable")
            _diag_inc(i, "noop")
            mutations.append(MutationResult(
                clause_idx=i, clause_text=clause, operator=op_name,
                survived=True, mutated_spec=mutated_spec,
                mutant_code=ref_code,
                test_results=[True] * len(t_inputs),
                noop=True,
                slot_type=slot_type,
                slot_original=slot_original,
                slot_target=slot_target,
                variant_idx=variant_idx,
                variant_total=variant_total,
            ))
            return "evaluable"

        # Post-hoc dedup keyed on mutated_spec. Catches both same-
        # clause cross-operator/variant collisions and cross-clause
        # collisions from duplicate decomposed rules under different
        # anchors.
        ms_key = _normalize_spec(mutated_spec)
        prior = seen_mutated_in_problem.get(ms_key)
        if prior is not None:
            prior_ci = prior["clause_idx"]
            prior_op = prior["operator"]
            if prior_ci == i:
                logger.info(
                    f"      DROPPED-DUPLICATE (same mutated_spec "
                    f"as prior {prior_op} on clause {i})"
                )
            else:
                logger.info(
                    f"      DROPPED-DUPLICATE-CROSS-CLAUSE "
                    f"(same mutated_spec as prior {prior_op} on "
                    f"clause {prior_ci}, attempted on clause {i})"
            )
            num_skipped_duplicate += 1
            _diag_inc(i, "duplicates")
            prior_candidate_specs_by_clause.setdefault(i, []).append(
                (candidate_label, mutated_spec)
            )
            return "rejected"
        seen_mutated_in_problem[ms_key] = {
            "clause_idx": i,
            "operator": candidate_label,
            "raw_ms": mutated_spec,
        }
        prior_candidate_specs_by_clause.setdefault(i, []).append(
            (candidate_label, mutated_spec)
        )

        process_status = _process_mutated_spec(
            i, clause, op_name, mutated_spec,
            stale_samples_matched_original=stale_samples_matched_original,
            slot_type=slot_type,
            slot_original=slot_original,
            slot_target=slot_target,
            variant_idx=variant_idx,
            variant_total=variant_total,
        )
        if process_status == "rejected":
            seen_mutated_in_problem.pop(ms_key, None)
        return process_status

    for i, rec in enumerate(clause_records):
        clause = rec.text
        quote = rec.quote
        for op_name in _ordered_clause_operators(config.operators, rec.type_tag):
            if _clause_mutant_budget_reached(i):
                continue

            classic_candidates: list[dict[str, str]] = []
            if getattr(config, "classic_mutants", True):
                classic_candidates = classic_mutation_candidates(
                    problem.spec, clause, quote, rec.type_tag, op_name,
                    getattr(config, "classic_mutants_per_operator", 64),
                )

            llm_enabled = bool(getattr(config, "llm_mutants", True))
            llm_has_work = (
                llm_enabled
                and op_name in rec.applicable_ops
            )
            operator_has_work = bool(classic_candidates) or llm_has_work
            if operator_has_work and _clause_codegen_failure_budget_reached(i):
                logger.info(
                    f"    Clause {i} [{op_name}]: skipped by codegen "
                    f"failure budget "
                    f"({max_codegen_failures_per_clause}/clause)"
                )
                num_operators_skipped_budget += 1
                clause_budget_stats[i]["operators_skipped_budget"].append(
                    f"{op_name}:codegen_failure_budget"
                )
                _diag_inc(i, "operators_skipped_budget")
                continue
            if (
                operator_has_work
                and max_operators_per_clause > 0
                and len(clause_budget_stats[i]["operators_used"])
                >= max_operators_per_clause
            ):
                logger.info(
                    f"    Clause {i} [{op_name}]: skipped by operator "
                    f"budget ({max_operators_per_clause}/clause)"
                )
                num_operators_skipped_budget += 1
                clause_budget_stats[i]["operators_skipped_budget"].append(
                    op_name
                )
                _diag_inc(i, "operators_skipped_budget")
                continue

            if operator_has_work:
                clause_budget_stats[i]["operators_used"].append(op_name)

            if classic_candidates:
                for classic_idx, candidate in enumerate(
                    classic_candidates, start=1
                ):
                    if _clause_mutant_budget_reached(i):
                        _record_candidate_budget_skip(
                            i, len(classic_candidates) - classic_idx + 1
                        )
                        break
                    classic_label = f"{op_name}@classic{classic_idx}"
                    logger.info(
                        f"    Clause {i} [{classic_label}]: "
                        f"{clause[:50]}..."
                    )
                    num_classic_candidates += 1
                    _diag_inc(i, "candidate_attempts")
                    _diag_inc(i, "classic_candidates")
                    status = _handle_mutation_candidate(
                        i, rec, clause, quote, op_name, classic_label,
                        candidate["mutated_spec"],
                        slot_type=candidate.get("slot_type"),
                        slot_original=candidate.get("slot_original"),
                        slot_target=candidate.get("slot_target"),
                    )
                    _record_budgeted_mutation(i, status)
                    if status == "evaluable":
                        num_classic_accepted += 1
                    if _clause_codegen_failure_budget_reached(i):
                        _record_candidate_budget_skip(
                            i, len(classic_candidates) - classic_idx
                        )
                        break

            if not llm_enabled:
                continue

            # Structural skip A: the decomposer tagged this operator as
            # not meaningful for this clause type (for example,
            # "perturb" on a non-numeric clause). That tag still gates
            # expensive LLM attempts, but deterministic classic
            # candidates above are driven by concrete syntactic slots
            # so decomposition sparsity does not starve mutant volume.
            if op_name not in rec.applicable_ops:
                logger.info(
                    f"    Clause {i} [{op_name}]: skipped LLM "
                    f"(not applicable to type={rec.type_tag})"
                )
                if not classic_candidates:
                    num_skipped_structural += 1
                    _diag_inc(i, "skipped_structural")
                    _diag_inc(i, "skipped_applicability")
                continue

            variant_total = mutation_variant_budget(
                clause, quote, rec.type_tag, op_name, mutants_per_operator
            )
            for variant_idx in range(1, variant_total + 1):
                if _clause_mutant_budget_reached(i):
                    _record_candidate_budget_skip(
                        i, variant_total - variant_idx + 1
                    )
                    break
                variant_label = (
                    f"{op_name}#{variant_idx}"
                    if variant_total > 1 else op_name
                )
                logger.info(
                    f"    Clause {i} [{variant_label}]: "
                    f"{clause[:50]}..."
                )
                _diag_inc(i, "candidate_attempts")
                _diag_inc(i, "llm_candidates")

                # Stage 3: spec mutation — Group T (tooling).
                # Prior mutated specs from earlier attempts on the
                # SAME clause are passed to the mutator so the prompt
                # can nudge the LLM away from emitting a byte-identical
                # output. With --mutants-per-operator > 1 this includes
                # rejected-but-clean attempts too, which is exactly the
                # pressure needed to get several distinct mutants from
                # one mutation point.
                priors_for_clause = prior_candidate_specs_by_clause.get(i, [])
                mutate_fn = _OPERATOR_FNS[op_name]
                try:
                    mutated_spec = mutate_fn(
                        client, problem.spec, clause, config.model_tooling,
                        quote=quote,
                        prior_mutated_specs=priors_for_clause,
                        variant_idx=variant_idx,
                        variant_total=variant_total,
                    )
                except Exception as e:
                    logger.warning(f"      Mutation failed: {e}")
                    _diag_inc(i, "mutation_failures")
                    _diag_inc(i, "skipped_structural")
                    continue

                status = _handle_mutation_candidate(
                    i, rec, clause, quote, op_name, variant_label,
                    mutated_spec,
                    variant_idx=variant_idx,
                    variant_total=variant_total,
                )
                _record_budgeted_mutation(i, status)
                if _clause_codegen_failure_budget_reached(i):
                    _record_candidate_budget_skip(
                        i, variant_total - variant_idx
                    )
                    break

    # Step 5: Build matrix (clause-level + mutant-level)
    mat = build_matrix(clauses, mutations, len(t_inputs))
    kr_str = f"{mat.kill_rate:.2f}" if mat.kill_rate is not None else "None"
    logger.info(
        f"  SC={mat.sc}, GR={mat.gr}, kill_rate={kr_str}, "
        f"evaluable_clauses={mat.num_evaluable_clauses}/{mat.num_total_clauses}, "
        f"survival_rate={mat.survival_rate:.2f}, noop_rate={mat.noop_rate:.2f}"
    )
    # Serialise example pairs for audit/debugging. Each pair is
    # (args_list, expected) and the loader populated it via the
    # benchmark-specific parser. We convert any non-JSON-friendly
    # values to repr() so they round-trip cleanly through the
    # results.jsonl file.
    example_pairs_serialised = [
        {"args": list(args), "expected": expected}
        for args, expected in (problem.example_pairs or [])
    ]

    return {
        "task_id": problem.task_id,
        "spec": problem.spec,
        "reference_program": ref_code,
        "clauses": clauses,
        "quotes": quotes,
        "clause_types": clause_types,
        "clause_applicable_ops": clause_applicable_ops,
        "num_quote_anchored_clauses": num_anchored,
        "mutants_per_operator": mutants_per_operator,
        "num_mutations_skipped_structural": num_skipped_structural,
        "num_mutations_skipped_duplicate": num_skipped_duplicate,
        "num_mutations_rejected_quality": num_rejected_quality,
        "mutation_quality_rejection_counts": quality_rejection_counts,
        "num_mutations_rejected_impact": num_rejected_impact,
        "mutation_impact_rejection_counts": impact_rejection_counts,
        "num_mutations_rejected_codegen": num_rejected_codegen,
        "mutation_codegen_rejection_counts": codegen_rejection_counts,
        "num_mutations_repaired_samples": num_repaired_samples,
        "mutation_sample_repair_counts": sample_repair_counts,
        "num_classic_mutation_candidates": num_classic_candidates,
        "num_classic_mutations_accepted": num_classic_accepted,
        "classic_mutants": getattr(config, "classic_mutants", True),
        "llm_mutants": getattr(config, "llm_mutants", True),
        "max_mutants_per_clause": max_mutants_per_clause,
        "max_operators_per_clause": max_operators_per_clause,
        "max_codegen_failures_per_clause": max_codegen_failures_per_clause,
        "num_operators_skipped_budget": num_operators_skipped_budget,
        "num_mutation_candidates_skipped_budget": (
            num_mutation_candidates_skipped_budget
        ),
        "clause_mutation_budgets": clause_budget_stats,
        "clause_mutation_diagnostics": clause_mutation_diagnostics,
        "example_pairs": example_pairs_serialised,
        "matrix": mat.data,
        "mutations": [
            {
                "clause_idx": m.clause_idx,
                "operator": m.operator,
                "survived": m.survived,
                "noop": m.noop,
                "slot_type": m.slot_type,
                "slot_original": m.slot_original,
                "slot_target": m.slot_target,
                "variant_idx": m.variant_idx,
                "variant_total": m.variant_total,
                "duplicate_of": m.duplicate_of,
                "mutated_spec": m.mutated_spec,
                "mutant_code": m.mutant_code,
                "test_results": m.test_results,
            }
            for m in mutations
        ],
        "sc": mat.sc,
        "sc_evaluable": mat.sc_evaluable,
        "num_clause_covered": mat.num_killed_clauses,
        "num_clause_evaluable": mat.num_evaluable_clauses,
        "num_clause_survived_only": mat.num_survived_only_clauses,
        "num_clause_no_evaluable": mat.num_no_evaluable_clauses,
        "gr": mat.gr,
        "survival_rate": mat.survival_rate,
        "noop_rate": mat.noop_rate,
        # Mutant-level rollup. ``mutant_data`` is intentionally
        # omitted from the JSON because it is fully reconstructable
        # from ``mutations[*].test_results`` and would otherwise
        # unnecessarily enlarge the JSONL files.
        "kill_rate": mat.kill_rate,
        "per_operator_kill_rate": mat.per_operator_kill_rate,
        "per_operator_count": mat.per_operator_count,
        "per_slot_kill_rate": mat.per_slot_kill_rate,
        "per_slot_count": mat.per_slot_count,
        "num_tests": len(t_inputs),
        "num_clauses": len(clauses),
    }


def load_completed(results_file: Path) -> dict[str, dict]:
    completed = {}
    if results_file.exists():
        for line in results_file.read_text().strip().splitlines():
            if line.strip():
                data = json.loads(line)
                completed[data["task_id"]] = data
    return completed


def main():
    parser = argparse.ArgumentParser(description="Spec Mutation Testing")
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "livecodebench"],
        default="humaneval",
    )
    parser.add_argument(
        "--lcb-split-file", type=str, default="test.jsonl",
        help="LiveCodeBench JSONL shard to load (test.jsonl..test6.jsonl). "
             "Ignored for other benchmarks.",
    )
    parser.add_argument(
        "--lcb-cutoff", type=str, default="2024-01-01",
        help="LiveCodeBench contest_date filter (ISO date). Problems "
             "before this date are skipped. Ignored for other benchmarks.",
    )
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--operators", type=str,
        default=(
            "weaken,negate,perturb,example,substitute,"
            "partial_edit,inconsistent_example"
        ),
    )
    parser.add_argument(
        "--mutants-per-operator", type=int, default=1,
        help="How many distinct mutation candidates to request for each "
             "applicable (clause, operator) pair. Values above 1 increase "
             "mutant count and cost; rejected/duplicate candidates are "
             "still filtered before codegen.",
    )
    parser.add_argument(
        "--classic-mutants", dest="classic_mutants", action="store_true",
        default=True,
        help="Add deterministic, slot-style candidates before the LLM "
             "mutator. This increases mutant density in the style of "
             "classic mutation-testing tools.",
    )
    parser.add_argument(
        "--no-classic-mutants", dest="classic_mutants", action="store_false",
        help="Disable deterministic slot-style candidates and use only "
             "LLM-generated mutations.",
    )
    parser.add_argument(
        "--llm-mutants", dest="llm_mutants", action="store_true",
        default=True,
        help="Enable LLM-generated mutations after deterministic slot-style "
             "candidates.",
    )
    parser.add_argument(
        "--no-llm-mutants", "--classic-only-mutants",
        dest="llm_mutants", action="store_false",
        help="Disable LLM-generated mutations. With --classic-mutants, this "
             "runs the rule-inspired deterministic-only ablation.",
    )
    parser.add_argument(
        "--classic-mutants-per-operator", type=int, default=64,
        help="Maximum deterministic slot-style candidates to add for each "
             "syntactically eligible (clause, operator) pair before quality/impact "
             "filtering.",
    )
    parser.add_argument(
        "--max-mutants-per-clause", type=int, default=4,
        help="Maximum recorded mutants per clause after quality/impact/"
             "duplicate gates. Counts killed, survived, and noop "
             "records. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-operators-per-clause", type=int, default=8,
        help="Maximum mutation operators to attempt per clause, after "
             "clause-type priority ordering and ignoring operators with "
             "no applicable LLM path or deterministic candidates. Set 0 "
             "to disable.",
    )
    parser.add_argument(
        "--max-codegen-failures-per-clause", type=int, default=8,
        help="Stop attempting additional mutants for a clause after this "
             "many candidates pass mutation gates but fail code generation. "
             "This bounds cost when the generator repeatedly returns empty "
             "or invalid completions. Set 0 to disable.",
    )
    parser.add_argument(
        "--llm-timeout-seconds", type=float, default=None,
        help="Override the per-attempt LLM API timeout. Defaults to the "
             "model-sensitive client timeout.",
    )
    parser.add_argument(
        "--llm-max-attempts", type=int, default=5,
        help="Maximum LLM API attempts for retryable transient failures.",
    )
    parser.add_argument(
        "--retry-failed-refs",
        action="store_true",
        help=(
            "Ignore cached reference programs recorded with passes=false "
            "and attempt to regenerate them. Useful for final runs after "
            "reference-generation prompts or model behavior have changed."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Override the root directory for results.",
    )
    parser.add_argument(
        "--model-generator", type=str, default=None,
        help="Group G: model used for reference generation, mutant "
             "code generation, and RQ4 action experiments. This is the model whose SC/GR is "
             "characterised by the run.",
    )
    parser.add_argument(
        "--model-tooling", type=str, default=None,
        help="Group T: model used for clause decomposition and spec "
             "mutation. Defaults to the generator model when omitted.",
    )
    parser.add_argument(
        "--generator-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning_effort for the generator model when the "
             "model supports OpenAI-style reasoning controls.",
    )
    parser.add_argument(
        "--tooling-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning_effort for the tooling model when the "
             "model supports OpenAI-style reasoning controls.",
    )
    parser.add_argument("--start", type=int, default=None, help="Start index (0-based, inclusive) in sorted problem list")
    parser.add_argument("--end", type=int, default=None, help="End index (0-based, exclusive) in sorted problem list")
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Split the [start, end) range into independent batches of "
             "this size, each written to its own results directory. "
             "Auto-resume picks up incomplete batches on re-run. "
             "Set to 0 to disable batching and write a single directory.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--precompute-refs", action="store_true",
                        help="Only generate and cache reference programs, skip mutations")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for --precompute-refs. "
                             "Each worker holds one in-flight LLM request and one "
                             "subprocess executing the validation tests. The ref "
                             "cache is serialised across workers via a lock.")
    parser.add_argument(
        "--run-id", type=str, default="",
        help="Tag for stochastic-variability runs. Suffixes the "
             "result directory (e.g. livecodebench_0-20_run1) and is "
             "mixed into the GENERATOR's LLM cache key so re-running "
             "the same command with a different --run-id yields a "
             "fresh sample. Tooling cache (decompose/mutate) is "
             "unaffected. Empty (default) disables this behaviour and "
             "keeps the historical layout.",
    )
    parser.add_argument(
        "--generator-temperature", type=float, default=None,
        help="Sampling temperature for the GENERATOR's mutant code "
             "generation. Defaults to None (model default, usually 0). "
             "Set to 0.6 for variability runs paired with --run-id.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.problem and not args.all and args.start is None and args.end is None and not args.precompute_refs:
        parser.error("Specify --problem TASK_ID, --all, --start/--end range, or --precompute-refs")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config()
    # Resolve the two-model split. Precedence:
    #   1. --model-generator / --model-tooling take highest priority.
    #   2. Otherwise fall back to the Config defaults.
    if args.model_generator:
        config.model_generator = args.model_generator

    if args.model_tooling:
        config.model_tooling = args.model_tooling
    else:
        # Keep tooling in sync with generator if the user did not
        # override it explicitly (Config.__post_init__ already does
        # this, but re-apply in case --model-generator was set above).
        config.model_tooling = (
            args.model_tooling or config.model_tooling or config.model_generator
        )

    logger.info(
        f"Models: generator={config.model_generator}, "
        f"tooling={config.model_tooling}"
    )
    config.generator_reasoning_effort = args.generator_reasoning_effort
    config.tooling_reasoning_effort = args.tooling_reasoning_effort
    if (
        config.generator_reasoning_effort
        or config.tooling_reasoning_effort
    ):
        logger.info(
            "Reasoning effort: generator=%s, tooling=%s",
            config.generator_reasoning_effort or "default",
            config.tooling_reasoning_effort or "default",
        )
    config.operators = [
        op.strip() for op in args.operators.split(",") if op.strip()
    ]
    unknown_ops = [
        op for op in config.operators if op not in _OPERATOR_FNS
    ]
    if unknown_ops:
        parser.error(
            "unknown mutation operator(s): "
            + ", ".join(unknown_ops)
            + ". Active operators are: "
            + ", ".join(_OPERATOR_FNS)
        )
    config.mutants_per_operator = max(1, args.mutants_per_operator)
    config.classic_mutants = bool(args.classic_mutants)
    config.llm_mutants = bool(args.llm_mutants)
    config.classic_mutants_per_operator = max(
        0, args.classic_mutants_per_operator
    )
    config.max_mutants_per_clause = max(0, args.max_mutants_per_clause)
    config.max_operators_per_clause = max(0, args.max_operators_per_clause)
    config.max_codegen_failures_per_clause = max(
        0, args.max_codegen_failures_per_clause
    )
    config.llm_timeout_seconds = (
        max(1.0, args.llm_timeout_seconds)
        if args.llm_timeout_seconds is not None else None
    )
    config.llm_max_attempts = max(1, args.llm_max_attempts)
    config.retry_failed_refs = bool(args.retry_failed_refs)
    if args.results_root:
        config.results_dir = Path(args.results_root).expanduser()

    logger.info(f"Loading {args.benchmark}...")
    if args.benchmark == "livecodebench":
        from specmut.benchmarks import load_livecodebench
        problems = load_livecodebench(
            cutoff_date=args.lcb_cutoff,
            split_file=args.lcb_split_file,
        )
    else:
        problems = load_benchmark(args.benchmark)
    logger.info(f"Loaded {len(problems)} problems")

    if args.problem:
        problems = [p for p in problems if p.task_id == args.problem]
        if not problems:
            logger.error(f"Problem {args.problem} not found")
            sys.exit(1)

    # Sort by task_id, using numeric order when the id ends with digits
    # (e.g. HumanEval/0, HumanEval/1, ..., HumanEval/10 instead of
    # lexicographic 0, 1, 10, 100). Falls back to string order otherwise.
    def _task_sort_key(p):
        m = re.search(r"(\d+)$", p.task_id)
        num = int(m.group(1)) if m else -1
        prefix = p.task_id[: m.start()] if m else p.task_id
        return (prefix, num, p.task_id)

    problems.sort(key=_task_sort_key)

    # Preserve the original effective slice boundaries before we mutate
    # the `problems` list, because batch directory names use the
    # original indices into the sorted benchmark (not into the slice).
    effective_start = args.start if args.start is not None else 0
    effective_end = args.end if args.end is not None else len(problems)
    if args.start is not None or args.end is not None:
        logger.info(
            f"Slicing problems [{effective_start}:{effective_end}] of {len(problems)}"
        )
        problems = problems[effective_start:effective_end]

    # Run pipeline
    client = LLMClient(config)

    # Precompute-refs mode: only generate and cache reference
    # programs. Always uses the generator model (Group G).
    #
    # Parallelised across `args.workers` threads. Each worker runs one
    # LLM request followed by one validation subprocess; both are
    # naturally I/O bound so threads (not processes) are the right
    # tool. The ref cache is a single JSON file, so we serialise
    # writes across workers with a lock to avoid lost updates.
    if args.precompute_refs:
        passed = 0
        failed = 0
        cached = 0

        # Pre-pass: split problems into already-cached and to-process
        # so we don't enqueue work that has nothing to do.
        to_process = []
        for problem in problems:
            ref = get_ref(
                config,
                problem.task_id,
                config.model_generator,
                include_failed=not config.retry_failed_refs,
            )
            if ref:
                cached += 1
                if ref["passes"]:
                    passed += 1
                else:
                    failed += 1
            else:
                to_process.append(problem)
        logger.info(
            f"Precompute-refs: {cached} already cached, {len(to_process)} to "
            f"generate with {args.workers} workers"
        )

        cache_lock = threading.Lock()

        def _process_one(problem):
            """Generate the reference for one problem and validate it.

            Returns one of the strings ``"pass"``, ``"fail"``, or
            ``"error:<message>"`` so the caller can update counters
            without inspecting exception types.
            """
            try:
                ref_code = generate(
                    client, problem.spec, problem.prompt, problem.entry_point,
                    config.model_generator,
                    mode=getattr(problem, "mode", "function"),
                )
            except Exception as e:
                # Persist the failed attempt so the next run does not
                # retry it. The cache lock guards the read-modify-
                # write cycle inside save_ref against concurrent
                # callers.
                with cache_lock:
                    save_ref(
                        config, problem.task_id, config.model_generator,
                        "", False,
                    )
                return f"error:{type(e).__name__}: {str(e)[:120]}"

            ref_results = run_tests(
                ref_code, problem.entry_point, problem.test_inputs,
                problem.expected_outputs, config.execution_timeout,
                mode=getattr(problem, "mode", "function"),
            )
            passes_b = all(ref_results)
            with cache_lock:
                save_ref(
                    config, problem.task_id, config.model_generator,
                    ref_code, passes_b,
                )
            if passes_b:
                return "pass"
            fail_count = sum(1 for r in ref_results if not r)
            return f"fail:{fail_count}/{len(ref_results)}"

        completed_so_far = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(_process_one, p): p for p in to_process
            }
            for fut in as_completed(futures):
                problem = futures[fut]
                completed_so_far += 1
                try:
                    outcome = fut.result()
                except Exception as e:
                    # Defensive: _process_one should already swallow
                    # exceptions, but if anything escapes (e.g. a
                    # KeyboardInterrupt or a programming error), don't
                    # let it kill the whole loop.
                    logger.error(
                        f"  [{completed_so_far}/{len(to_process)}] "
                        f"{problem.task_id}: unhandled {e}"
                    )
                    failed += 1
                    continue

                progress = f"[{completed_so_far}/{len(to_process)}]"
                if outcome == "pass":
                    passed += 1
                    logger.info(f"  {progress} {problem.task_id}: pass")
                elif outcome.startswith("fail"):
                    failed += 1
                    _, ratio = outcome.split(":", 1)
                    logger.warning(
                        f"  {progress} {problem.task_id}: ref fails "
                        f"benchmark ({ratio} tests)"
                    )
                else:
                    failed += 1
                    _, msg = outcome.split(":", 1)
                    logger.warning(
                        f"  {progress} {problem.task_id}: generation "
                        f"failed ({msg})"
                    )

        logger.info(
            f"Precompute done: {passed} pass, {failed} fail, "
            f"{cached} already cached"
        )
        return

    # Build the list of batches. Each batch is a (batch_start,
    # batch_end) tuple in original-index space, meaning the indices
    # into the sorted benchmark (not into the current sliced list).
    #
    # Precedence:
    #   1. --resume: a single batch whose directory is the path's
    #      parent. Preserves backward compatibility.
    #   2. --batch-size > 0 and slice larger than batch_size: split
    #      into fixed-size chunks.
    #   3. Otherwise: one batch covering the whole slice.
    if args.resume:
        batches = [(effective_start, effective_end, Path(args.resume).parent)]
    elif args.batch_size > 0 and len(problems) > args.batch_size:
        batches = []
        for i in range(0, len(problems), args.batch_size):
            bs = effective_start + i
            be = min(bs + args.batch_size, effective_end)
            batches.append((bs, be, None))
    else:
        batches = [(effective_start, effective_end, None)]

    logger.info(f"Processing {len(batches)} batch(es)")

    for batch_idx, (batch_start, batch_end, override_dir) in enumerate(batches):
        local_lo = batch_start - effective_start
        local_hi = batch_end - effective_start
        batch_problems = problems[local_lo:local_hi]
        logger.info(
            f"=== Batch {batch_idx + 1}/{len(batches)}: "
            f"[{batch_start}, {batch_end}) ({len(batch_problems)} problems) ==="
        )
        _run_batch(
            config=config,
            client=client,
            args=args,
            batch_problems=batch_problems,
            batch_start=batch_start,
            batch_end=batch_end,
            override_run_dir=override_dir,
        )


def _run_batch(
    config,
    client,
    args,
    batch_problems,
    batch_start: int,
    batch_end: int,
    override_run_dir: Path | None = None,
):
    """Execute the pipeline over one batch and write its own
    results.jsonl, config.json, summary.json, and simplified/.

    If ``override_run_dir`` is set, the batch uses that directory
    instead of computing one from ``batch_start``/``batch_end``. This
    preserves ``--resume`` semantics where the user names a specific
    existing directory to continue.
    """
    # Determine the output directory for this batch.
    if override_run_dir is not None:
        run_dir = override_run_dir
    else:
        run_tag = f"_run{args.run_id}" if args.run_id else ""
        # Sanitise model id for filesystem use. Directory is keyed by
        # the generator (Group G) only; see comment at the original
        # site in main() for rationale.
        model_dir = (
            config.model_generator.replace("/", "_").replace(":", "_")
        )
        run_dir = (
            config.results_dir
            / model_dir
            / f"{args.benchmark}_{batch_start}-{batch_end}{run_tag}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "results.jsonl"

    # Auto-resume: if this batch's results file already exists, load
    # the completed task ids so a re-run of the same command picks up
    # where the previous one stopped. This is what makes re-running
    # the same command idempotent, without the user needing --resume.
    completed_records = load_completed(results_file) if results_file.exists() else {}
    completed = set(completed_records)
    if completed:
        logger.info(
            f"  Auto-resume: {len(completed)} problems already completed in "
            f"{run_dir.name}"
        )

    # Short-circuit: if every problem in this batch is already done,
    # skip without even writing a fresh config.json so the original
    # timestamp is preserved.
    remaining = [p for p in batch_problems if p.task_id not in completed]
    if not remaining and completed:
        logger.info(
            f"  Batch already complete ({len(completed)} problems), skipping"
        )
        return

    # Save config snapshot for this batch. Both models are recorded,
    # together with the batch bounds, so each batch directory is a
    # fully-self-describing experimental artifact.
    config_snapshot = {
        "model_generator": config.model_generator,
        "model_tooling": config.model_tooling,
        "generator_reasoning_effort": config.generator_reasoning_effort,
        "tooling_reasoning_effort": config.tooling_reasoning_effort,
        "benchmark": args.benchmark,
        "operators": config.operators,
        "mutants_per_operator": config.mutants_per_operator,
        "classic_mutants": config.classic_mutants,
        "llm_mutants": config.llm_mutants,
        "classic_mutants_per_operator": config.classic_mutants_per_operator,
        "max_mutants_per_clause": config.max_mutants_per_clause,
        "max_operators_per_clause": config.max_operators_per_clause,
        "max_codegen_failures_per_clause": (
            config.max_codegen_failures_per_clause
        ),
        "llm_timeout_seconds": config.llm_timeout_seconds,
        "llm_max_attempts": config.llm_max_attempts,
        "retry_failed_refs": config.retry_failed_refs,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Loader args used to build this batch. Persisted so the
        # post-hoc analysis can re-load the exact
        # same problem set without guessing defaults.
        "lcb_split_file": args.lcb_split_file,
        "lcb_cutoff": args.lcb_cutoff,
        "run_id": args.run_id,
        "generator_temperature": args.generator_temperature,
        "results_root": str(config.results_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))

    all_results = list(completed_records.values())
    skipped = []

    # File-write lock. results.jsonl is append-only when workers > 1,
    # so writes must be serialised. Per-problem simplified files live
    # under simplified/<task_id>/, so they do not collide between
    # workers and need no extra lock; only the parent dir's mkdir is
    # shared, which Path.mkdir(exist_ok=True) handles safely.
    write_lock = threading.Lock()

    def _process_one(problem):
        """Run the full pipeline for one problem and write its
        results. Returns one of:

            ("done", result_dict)
            ("skipped", reason_string)
            ("error", error_message)
        """
        try:
            result = run_problem(
                client, problem, config,
                run_id=args.run_id,
                generator_temperature=args.generator_temperature,
            )
            if result is None:
                return ("skipped", "ref-program-fails-benchmark")
            result["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Persist the result line and the per-problem simplified
            # tree. Both happen under the lock so a concurrent writer
            # can never see a partial line in results.jsonl.
            with write_lock:
                with open(results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                _write_simplified(run_dir, result)

            return ("done", result)
        except Exception as e:
            return ("error", f"{type(e).__name__}: {e}")

    # Schedule the work. Auto-resume already filters out completed
    # problems before we get here via the `completed` set, but we
    # double-check inside the worker dispatch as well.
    pending = [p for p in batch_problems if p.task_id not in completed]
    for tid in (p.task_id for p in batch_problems if p.task_id in completed):
        logger.info(f"Skipping {tid} (already completed)")

    completed_so_far = 0
    workers = max(1, args.workers)
    logger.info(
        f"  Processing {len(pending)} problems with {workers} worker(s)"
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, p): p for p in pending}
        for fut in as_completed(futures):
            problem = futures[fut]
            completed_so_far += 1
            try:
                status, payload = fut.result()
            except Exception as e:
                logger.error(
                    f"  [{completed_so_far}/{len(pending)}] "
                    f"{problem.task_id}: unhandled {type(e).__name__}: {e}"
                )
                skipped.append(problem.task_id)
                continue

            progress = f"[{completed_so_far}/{len(pending)}]"
            if status == "done":
                all_results.append(payload)
                logger.info(
                    f"  {progress} {problem.task_id}: done"
                )
            elif status == "skipped":
                skipped.append(problem.task_id)
                logger.warning(
                    f"  {progress} {problem.task_id}: skipped ({payload})"
                )
            else:  # error
                logger.error(
                    f"  {progress} {problem.task_id}: error ({payload})"
                )
                skipped.append(problem.task_id)

    # Write summary
    sc_values = [
        r["sc"]
        for r in all_results
        if r.get("sc") is not None
    ]
    sc_evaluable_values = [
        r["sc_evaluable"]
        for r in all_results
        if r.get("sc_evaluable") is not None
    ]
    gr_values = [r["gr"] for r in all_results if r["gr"] is not None]
    surv_values = [
        r["survival_rate"]
        for r in all_results
        if r.get("survival_rate") is not None
    ]
    noop_values = [r["noop_rate"] for r in all_results if r.get("noop_rate") is not None]
    # Mutant-level kill rate aggregated across problems. Reported
    # alongside mean_sc because the two answer different questions
    # (clause coverage vs classical mutation kill rate).
    kill_values = [r["kill_rate"] for r in all_results if r.get("kill_rate") is not None]
    # Per-problem applicable-ops counts (how many operators the
    # decomposer declared meaningful, averaged across clauses).
    mean_applicable = []
    total_structural_skips = 0
    total_duplicate_skips = 0
    total_quality_rejections = 0
    quality_rejection_counts: dict[str, int] = {}
    total_impact_rejections = 0
    impact_rejection_counts: dict[str, int] = {}
    total_codegen_rejections = 0
    codegen_rejection_counts: dict[str, int] = {}
    total_sample_repairs = 0
    sample_repair_counts: dict[str, int] = {}
    total_classic_candidates = 0
    total_classic_accepted = 0
    total_operator_budget_skips = 0
    total_candidate_budget_skips = 0
    total_clause_covered = 0
    total_clause_evaluable = 0
    total_clause_survived_only = 0
    total_clause_no_evaluable = 0
    # Per-operator and per-slot kill aggregation across the batch.
    # We re-derive the per-batch kill rate as killed/total rather
    # than averaging the per-problem rates, because per-problem
    # averaging double-weights problems with few mutants.
    op_total: dict[str, int] = {}
    op_killed: dict[str, int] = {}
    slot_total: dict[str, int] = {}
    slot_killed: dict[str, int] = {}
    for r in all_results:
        ops_lists = r.get("clause_applicable_ops", [])
        if ops_lists:
            mean_applicable.append(
                sum(len(ops) for ops in ops_lists) / len(ops_lists)
            )
        total_structural_skips += r.get("num_mutations_skipped_structural", 0)
        total_duplicate_skips += r.get("num_mutations_skipped_duplicate", 0)
        total_quality_rejections += r.get("num_mutations_rejected_quality", 0)
        total_impact_rejections += r.get("num_mutations_rejected_impact", 0)
        total_codegen_rejections += r.get("num_mutations_rejected_codegen", 0)
        total_sample_repairs += r.get("num_mutations_repaired_samples", 0)
        total_classic_candidates += r.get(
            "num_classic_mutation_candidates", 0
        )
        total_classic_accepted += r.get(
            "num_classic_mutations_accepted", 0
        )
        total_operator_budget_skips += r.get(
            "num_operators_skipped_budget", 0
        )
        total_candidate_budget_skips += r.get(
            "num_mutation_candidates_skipped_budget", 0
        )
        total_clause_covered += r.get("num_clause_covered", 0)
        total_clause_evaluable += r.get("num_clause_evaluable", 0)
        total_clause_survived_only += r.get("num_clause_survived_only", 0)
        total_clause_no_evaluable += r.get("num_clause_no_evaluable", 0)
        for reason, n in (
            r.get("mutation_quality_rejection_counts", {}) or {}
        ).items():
            quality_rejection_counts[reason] = (
                quality_rejection_counts.get(reason, 0) + n
            )
        for reason, n in (
            r.get("mutation_impact_rejection_counts", {}) or {}
        ).items():
            impact_rejection_counts[reason] = (
                impact_rejection_counts.get(reason, 0) + n
            )
        for reason, n in (
            r.get("mutation_codegen_rejection_counts", {}) or {}
        ).items():
            codegen_rejection_counts[reason] = (
                codegen_rejection_counts.get(reason, 0) + n
            )
        for reason, n in (
            r.get("mutation_sample_repair_counts", {}) or {}
        ).items():
            sample_repair_counts[reason] = (
                sample_repair_counts.get(reason, 0) + n
            )
        # Sum per-operator counts across problems.
        per_op_count = r.get("per_operator_count", {})
        per_op_kr = r.get("per_operator_kill_rate", {})
        for op, n in per_op_count.items():
            op_total[op] = op_total.get(op, 0) + n
            killed = round(per_op_kr.get(op, 0.0) * n)
            op_killed[op] = op_killed.get(op, 0) + killed
        # Sum per-slot counts across problems.
        per_sl_count = r.get("per_slot_count", {})
        per_sl_kr = r.get("per_slot_kill_rate", {})
        for slot, n in per_sl_count.items():
            slot_total[slot] = slot_total.get(slot, 0) + n
            killed = round(per_sl_kr.get(slot, 0.0) * n)
            slot_killed[slot] = slot_killed.get(slot, 0) + killed

    per_op_kill_rate_batch = {
        op: op_killed.get(op, 0) / op_total[op]
        for op in op_total
    }
    per_slot_kill_rate_batch = {
        slot: slot_killed.get(slot, 0) / slot_total[slot]
        for slot in slot_total
    }

    summary = {
        "benchmark": args.benchmark,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "num_problems_attempted": len(batch_problems),
        "num_problems_completed": len(all_results),
        "num_problems_skipped": len(skipped),
        "skipped_ids": skipped,
        "mean_sc": sum(sc_values) / len(sc_values) if sc_values else None,
        "mean_sc_evaluable": (
            sum(sc_evaluable_values) / len(sc_evaluable_values)
            if sc_evaluable_values else None
        ),
        "mean_gr": sum(gr_values) / len(gr_values) if gr_values else None,
        "mean_survival_rate": sum(surv_values) / len(surv_values) if surv_values else None,
        "mean_noop_rate": sum(noop_values) / len(noop_values) if noop_values else None,
        "mean_kill_rate": sum(kill_values) / len(kill_values) if kill_values else None,
        "mean_applicable_ops_per_clause": (
            sum(mean_applicable) / len(mean_applicable)
            if mean_applicable else None
        ),
        "num_clause_covered": total_clause_covered,
        "num_clause_evaluable": total_clause_evaluable,
        "num_clause_survived_only": total_clause_survived_only,
        "num_clause_no_evaluable": total_clause_no_evaluable,
        "num_mutations_skipped_structural": total_structural_skips,
        "num_mutations_skipped_duplicate": total_duplicate_skips,
        "num_mutations_rejected_quality": total_quality_rejections,
        "mutation_quality_rejection_counts": quality_rejection_counts,
        "num_mutations_rejected_impact": total_impact_rejections,
        "mutation_impact_rejection_counts": impact_rejection_counts,
        "num_mutations_rejected_codegen": total_codegen_rejections,
        "mutation_codegen_rejection_counts": codegen_rejection_counts,
        "num_mutations_repaired_samples": total_sample_repairs,
        "mutation_sample_repair_counts": sample_repair_counts,
        "num_classic_mutation_candidates": total_classic_candidates,
        "num_classic_mutations_accepted": total_classic_accepted,
        "num_operators_skipped_budget": total_operator_budget_skips,
        "num_mutation_candidates_skipped_budget": (
            total_candidate_budget_skips
        ),
        "per_operator_kill_rate": per_op_kill_rate_batch,
        "per_operator_count": op_total,
        "per_slot_kill_rate": per_slot_kill_rate_batch,
        "per_slot_count": slot_total,
        "operators_used": config.operators,
        "mutants_per_operator": config.mutants_per_operator,
        "classic_mutants": config.classic_mutants,
        "llm_mutants": config.llm_mutants,
        "classic_mutants_per_operator": config.classic_mutants_per_operator,
        "max_mutants_per_clause": config.max_mutants_per_clause,
        "max_operators_per_clause": config.max_operators_per_clause,
        "max_codegen_failures_per_clause": (
            config.max_codegen_failures_per_clause
        ),
        "llm_timeout_seconds": config.llm_timeout_seconds,
        "llm_max_attempts": config.llm_max_attempts,
        "retry_failed_refs": config.retry_failed_refs,
        "results_root": str(config.results_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    completed_this_run = len(all_results) - len(completed_records)
    logger.info(
        f"  Batch summary: {completed_this_run}/{len(batch_problems)} "
        f"completed this run, {len(skipped)} skipped, "
        f"mean SC={summary['mean_sc']}, mean GR={summary['mean_gr']}"
    )
    logger.info(f"  Results written to {run_dir}")


def _read_survived(mut: dict) -> bool:
    """Read the survived flag from a mutation row."""
    return bool(mut.get("survived", False))


def _verdict_for(mut: dict) -> tuple[str, int, int]:
    """Return (verdict_label, fail_count, total_count) for a mutation
    dict as stored in the per-problem result."""
    total = len(mut["test_results"])
    fail_count = sum(1 for t in mut["test_results"] if not t)
    if mut.get("noop"):
        label = "NOOP"
    elif _read_survived(mut):
        label = "SURVIVED"
    elif fail_count > 0:
        label = "KILLED"
    else:
        label = "SURVIVED"
    return label, fail_count, total


def _unified_diff(original: str, mutated: str) -> str:
    """Return a unified diff (git-style) between the two spec texts."""
    orig_lines = original.splitlines()
    mut_lines = mutated.splitlines()
    diff_lines = list(difflib.unified_diff(
        orig_lines, mut_lines,
        fromfile="original_spec",
        tofile="mutated_spec",
        lineterm="",
        n=3,
    ))
    if not diff_lines:
        return "(no textual difference)"
    return "\n".join(diff_lines)


def _write_overview(path: Path, result: dict):
    """Top-level overview of one problem: metrics, untested clauses,
    ungrounded tests, and a verdict table across all mutations."""
    lines = []
    lines.append(f"# {result['task_id']}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- SC (all clauses): `{result['sc']}`")
    if result.get("sc_evaluable") is not None:
        lines.append(f"- SC (evaluable clauses): `{result['sc_evaluable']}`")
    lines.append(f"- GR: `{result['gr']}`")
    kr = result.get("kill_rate")
    kr_str = f"`{kr:.3f}`" if kr is not None else "`—`"
    lines.append(f"- Mutation kill rate (mutant-level): {kr_str}")
    surv = result.get("survival_rate")
    if surv is not None:
        lines.append(f"- Survival rate: `{surv:.2f}`")
    else:
        lines.append("- Survival rate: `—`")
    lines.append(f"- Noop rate: `{result.get('noop_rate', 0.0):.2f}`")
    lines.append(f"- Clauses: {result['num_clauses']}")
    if result.get("num_clause_covered") is not None:
        lines.append(
            "- Clause status: "
            f"covered={result.get('num_clause_covered', 0)}, "
            f"survived_only={result.get('num_clause_survived_only', 0)}, "
            f"no_evaluable={result.get('num_clause_no_evaluable', 0)}"
        )
    lines.append(f"- Tests: {result['num_tests']}")
    n_skipped = result.get("num_mutations_skipped_structural", 0)
    if n_skipped:
        lines.append(
            "- Mutations skipped structurally "
            f"(type, anchor, applicability, or quality gate): {n_skipped}"
        )
    n_quality = result.get("num_mutations_rejected_quality", 0)
    if n_quality:
        counts = result.get("mutation_quality_rejection_counts", {}) or {}
        detail = ", ".join(
            f"{reason}={counts[reason]}" for reason in sorted(counts)
        )
        lines.append(
            f"- Mutations rejected by quality gate: {n_quality}"
            + (f" ({detail})" if detail else "")
        )
    n_impact = result.get("num_mutations_rejected_impact", 0)
    if n_impact:
        counts = result.get("mutation_impact_rejection_counts", {}) or {}
        detail = ", ".join(
            f"{reason}={counts[reason]}" for reason in sorted(counts)
        )
        lines.append(
            f"- Mutations rejected by impact gate: {n_impact}"
            + (f" ({detail})" if detail else "")
        )
    n_codegen = result.get("num_mutations_rejected_codegen", 0)
    if n_codegen:
        counts = result.get("mutation_codegen_rejection_counts", {}) or {}
        detail = ", ".join(
            f"{reason}={counts[reason]}" for reason in sorted(counts)
        )
        lines.append(
            f"- Mutations rejected by codegen responsiveness gate: {n_codegen}"
            + (f" ({detail})" if detail else "")
        )
    n_repaired = result.get("num_mutations_repaired_samples", 0)
    if n_repaired:
        counts = result.get("mutation_sample_repair_counts", {}) or {}
        detail = ", ".join(
            f"{reason}={counts[reason]}" for reason in sorted(counts)
        )
        lines.append(
            f"- Mutations with repaired Sample Output payloads: {n_repaired}"
            + (f" ({detail})" if detail else "")
        )
    lines.append("")

    # Per-operator kill-rate breakdown. Only render the table if we
    # actually have any killed (non-survived) mutants to report on.
    per_op_kr = result.get("per_operator_kill_rate", {}) or {}
    per_op_n = result.get("per_operator_count", {}) or {}
    if per_op_n:
        lines.append("## Per-operator kill rate")
        lines.append("")
        lines.append("| Operator | Killed mutants | Killed | Kill rate |")
        lines.append("|----------|----------------|--------|-----------|")
        for op in sorted(per_op_n):
            n = per_op_n[op]
            rate = per_op_kr.get(op, 0.0)
            killed = round(rate * n)
            lines.append(
                f"| {op} | {n} | {killed} | {rate:.3f} |"
            )
        lines.append("")

    # Example pairs extracted from the spec at load time. Showing
    # them lets a reader audit whether the benchmark loader is
    # parsing examples from the spec correctly.
    example_pairs = result.get("example_pairs") or []
    if example_pairs:
        lines.append("## Extracted example pairs")
        lines.append("")
        lines.append(
            f"*{len(example_pairs)} (input, expected) pair"
            f"{'s' if len(example_pairs) != 1 else ''} extracted from "
            f"the specification by the loader.*"
        )
        lines.append("")
        lines.append("| # | Input args | Expected output |")
        lines.append("|---|------------|-----------------|")
        for k, p in enumerate(example_pairs):
            args_repr = repr(p.get("args", []))
            exp_repr = repr(p.get("expected"))
            # Truncate long values for table readability; the full
            # values are still in the JSONL.
            if len(args_repr) > 200:
                args_repr = args_repr[:197] + "..."
            if len(exp_repr) > 200:
                exp_repr = exp_repr[:197] + "..."
            # Escape pipe characters so the markdown table renders.
            args_repr = args_repr.replace("|", "\\|")
            exp_repr = exp_repr.replace("|", "\\|")
            lines.append(f"| {k} | `{args_repr}` | `{exp_repr}` |")
        lines.append("")

    # Per-slot kill-rate breakdown. Populated by deterministic
    # rule-based candidates.
    per_slot_kr = result.get("per_slot_kill_rate", {}) or {}
    per_slot_n = result.get("per_slot_count", {}) or {}
    if per_slot_n:
        lines.append("## Slot-style mutants: per-slot-type kill rate")
        lines.append("")
        lines.append("| Slot type | Killed mutants | Killed | Kill rate |")
        lines.append("|-----------|----------------|--------|-----------|")
        for slot in sorted(per_slot_n):
            n = per_slot_n[slot]
            rate = per_slot_kr.get(slot, 0.0)
            killed = round(rate * n)
            lines.append(
                f"| {slot} | {n} | {killed} | {rate:.3f} |"
            )
        lines.append("")

    # Verdict table across all executed mutations.
    lines.append("## Mutation verdicts")
    lines.append("")
    lines.append("| Clause | Operator | Variant | Slot | Verdict | Failed / Total |")
    lines.append("|--------|----------|---------|------|---------|----------------|")
    for m in result["mutations"]:
        label, fc, tot = _verdict_for(m)
        variant_cell = "—"
        if m.get("variant_total", 1) > 1:
            variant_cell = f"{m.get('variant_idx', 1)}/{m.get('variant_total', 1)}"
        slot_cell = "—"
        if m.get("slot_type"):
            slot_cell = (
                f"{m['slot_type']}: "
                f"{m.get('slot_original', '?')!r} → "
                f"{m.get('slot_target', '?')!r}"
            )
        lines.append(
            f"| {m['clause_idx']} | {m['operator']} | "
            f"{variant_cell} | {slot_cell} | {label} | {fc}/{tot} |"
        )
    lines.append("")

    # Untested clauses (empty rows in the matrix).
    untested = [
        (i, c) for i, c in enumerate(result["clauses"])
        if i < len(result["matrix"]) and not any(result["matrix"][i])
    ]
    if untested:
        lines.append("## Untested clauses")
        lines.append("")
        for i, c in untested:
            lines.append(f"- Clause {i}: {c}")
        lines.append("")

    # Ungrounded tests (empty columns).
    matrix = result["matrix"]
    if matrix and matrix[0]:
        num_tests = len(matrix[0])
        ungrounded = sum(
            1 for j in range(num_tests)
            if not any(matrix[i][j] for i in range(len(matrix)))
        )
        if ungrounded > 0:
            lines.append(
                f"## Ungrounded tests: {ungrounded}/{num_tests}"
            )
            lines.append("")
            lines.append(
                "These tests are not killed by any mutation, so nothing "
                "in the current matrix vouches for their grounding in "
                "the specification."
            )
            lines.append("")

    path.write_text("\n".join(lines))


def _write_clauses(path: Path, result: dict):
    """Detailed list of decomposed clauses with quotes, types, and
    applicable operators, one per heading."""
    quotes = result.get("quotes", [])
    types = result.get("clause_types", [])
    ops_lists = result.get("clause_applicable_ops", [])

    lines = []
    lines.append(f"# Clauses for {result['task_id']}")
    lines.append("")
    for i, clause in enumerate(result["clauses"]):
        quote = quotes[i] if i < len(quotes) else ""
        type_tag = types[i] if i < len(types) else "unknown"
        ops = ops_lists[i] if i < len(ops_lists) else []

        lines.append(f"## Clause {i}")
        lines.append("")
        lines.append(f"**Text:** {clause}")
        lines.append("")
        lines.append(f"**Type:** `{type_tag}`")
        lines.append("")
        lines.append(
            f"**Applicable operators:** "
            f"{', '.join(f'`{o}`' for o in ops) if ops else '(none)'}"
        )
        lines.append("")
        if quote:
            lines.append("**Quote:**")
            lines.append("")
            lines.append("```")
            lines.append(quote)
            lines.append("```")
        else:
            lines.append("**Quote:** (none, distributed across spec)")
        lines.append("")

    path.write_text("\n".join(lines))


def _write_mutation(mut_dir: Path, result: dict, mut: dict):
    """Write one mutation file containing the clause context, the
    verdict, a unified diff against the original spec, the full
    mutated spec, and the generated mutant code."""
    label, fail_count, total = _verdict_for(mut)
    clause_idx = mut["clause_idx"]
    operator = mut["operator"]
    clause_text = result["clauses"][clause_idx]
    quote = (
        result.get("quotes", [""] * len(result["clauses"]))[clause_idx]
        if clause_idx < len(result.get("quotes", []))
        else ""
    )
    type_tag = (
        result.get("clause_types", ["unknown"] * len(result["clauses"]))[clause_idx]
        if clause_idx < len(result.get("clause_types", []))
        else "unknown"
    )

    lines = []
    lines.append(f"# Clause {clause_idx} [{operator}] -> {label}")
    lines.append("")
    lines.append(f"- Verdict: **{label}**")
    lines.append(f"- Failed tests: **{fail_count} / {total}**")
    lines.append(f"- Clause type: `{type_tag}`")
    if mut.get("variant_total", 1) > 1:
        lines.append(
            f"- Variant: `{mut.get('variant_idx', 1)} / "
            f"{mut.get('variant_total', 1)}`"
        )
    if mut.get("slot_type"):
        lines.append(
            f"- Slot substitution: `{mut['slot_type']}` "
            f"`{mut.get('slot_original', '?')}` → `{mut.get('slot_target', '?')}`"
        )
    lines.append("")
    lines.append("## Clause")
    lines.append("")
    lines.append(f"> {clause_text}")
    lines.append("")
    if quote:
        lines.append("**Anchor quote:**")
        lines.append("")
        lines.append("```")
        lines.append(quote)
        lines.append("```")
        lines.append("")
    else:
        lines.append("**Anchor quote:** (none, distributed across spec)")
        lines.append("")

    # Unified diff against the original spec. Shows exactly what the
    # mutator edited, which is usually the single most informative
    # artifact in this file.
    lines.append("## Diff (original -> mutated)")
    lines.append("")
    lines.append("```diff")
    lines.append(_unified_diff(result["spec"], mut["mutated_spec"]))
    lines.append("```")
    lines.append("")

    # Full mutated spec for completeness. The original is already in
    # ../spec.txt, so we do not duplicate it here.
    lines.append("## Full mutated spec")
    lines.append("")
    lines.append("```")
    lines.append(mut["mutated_spec"])
    lines.append("```")
    lines.append("")

    # Mutant code is essential for understanding why the behaviour
    # changed (or did not), especially for SURVIVED mutants.
    lines.append("## Mutant program")
    lines.append("")
    lines.append("```python")
    lines.append(mut.get("mutant_code", ""))
    lines.append("```")
    lines.append("")

    # Test-failure summary. Listing all booleans for a 1000-test suite
    # would be unreadable, so we only list the first 50 failing
    # indices as a hint.
    if label == "KILLED" and fail_count > 0:
        failing = [
            i for i, passed in enumerate(mut["test_results"]) if not passed
        ]
        lines.append("## Failing test indices")
        lines.append("")
        shown = failing[:50]
        lines.append(
            ", ".join(str(i) for i in shown)
            + (f" ... (+{len(failing) - len(shown)} more)" if len(failing) > 50 else "")
        )
        lines.append("")

    # Legacy swap mutations carry slot info that is included in the
    # filename so multiple slot variants of the same clause do not
    # collide. Newer (substitute) mutations have no slot_type and
    # fall through to the simpler filename.
    variant_suffix = ""
    if mut.get("variant_total", 1) > 1:
        variant_suffix = f"_v{mut.get('variant_idx', 1):02d}"
    if mut.get("slot_type"):
        sanitised_target = re.sub(r"[^a-zA-Z0-9]+", "_", str(mut.get("slot_target", ""))).strip("_") or "x"
        filename = (
            f"clause{clause_idx:02d}_{operator}{variant_suffix}_"
            f"{mut['slot_type']}_{sanitised_target}_{label.lower()}.md"
        )
    else:
        filename = (
            f"clause{clause_idx:02d}_{operator}"
            f"{variant_suffix}_{label.lower()}.md"
        )
    (mut_dir / filename).write_text("\n".join(lines))


def _write_simplified(run_dir: Path, result: dict):
    """Write a richly-structured per-problem view of one pipeline
    result. Layout:

        simplified/<TASK_ID>/
            overview.md           high-level metrics + verdict table
            spec.txt              benchmark specification used by this run
            reference.py          reference program (code used as the
                                  survival baseline)
            clauses.md            per-clause details with quotes, types,
                                  applicable operators
            mutations/
                clause00_remove_killed.md
                clause00_weaken_survived.md
                ...               one file per executed mutation with
                                  clause context, diff, full mutated
                                  spec, and mutant code
    """
    task_safe = result["task_id"].replace("/", "_")
    out_dir = run_dir / "simplified" / task_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-level overview and raw artifacts.
    _write_overview(out_dir / "overview.md", result)
    (out_dir / "spec.txt").write_text(result["spec"])
    (out_dir / "reference.py").write_text(
        result.get("reference_program", "")
    )
    _write_clauses(out_dir / "clauses.md", result)

    # One file per mutation.
    mut_dir = out_dir / "mutations"
    mut_dir.mkdir(exist_ok=True)
    for mut in result["mutations"]:
        _write_mutation(mut_dir, result, mut)


if __name__ == "__main__":
    main()
