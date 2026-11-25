"""Run post-hoc action experiments for classified survived mutants.

This script consumes ``survivor_actionable_pipeline*.jsonl`` and executes the
developer action implied by each category where an automated action exists.
It writes a separate sidecar; canonical mutation-testing results are not
modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from specmut.analysis.analyze_equivalence_emd_prompt import (
    _DEFAULT_CODE_CHARS,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MODEL as _DEFAULT_JUDGE_MODEL,
    _build_joint_messages,
    _load_jsonl,
    _mutation_change_summary,
    _parse_joint_response,
    _truncate_middle,
)
from specmut.analysis.analyze_actionable_pipeline import _is_contract_sensitive_change
from specmut.benchmarks import Problem, load_benchmark, load_livecodebench
from specmut.codegen import generate, replace_docstring
from specmut.config import Config
from specmut.execute import _values_equal, run_outputs, run_tests
from specmut.llm import LLMClient


_METHOD = "survivor_action_experiments_v4"
_DEFAULT_OUTPUT = "survivor_action_experiments.jsonl"
_DEFAULT_TEST_GENERATION_MAX_TOKENS = 4096
_DEFAULT_TEST_GENERATION_ROUNDS = 2
_DEFAULT_TEST_GENERATION_SAMPLES = 5
_ACTION_EMPHASIS = "Pay particular attention to this requirement:"
_IMPLEMENTATION_FOCUS_TITLE = "Implementation focus:"
_IMPLEMENTATION_FOCUS_NOTE = (
    "The following requirement is part of the problem statement and should be "
    "used when implementing the corresponding behavior."
)
_FOCUS_REQUIREMENT_OPEN = "<<<requirement"
_FOCUS_REQUIREMENT_CLOSE = "requirement>>>"
_REQUIREMENT_GUARD_OPEN = "<requirement_to_implement>"
_REQUIREMENT_GUARD_CLOSE = "</requirement_to_implement>"
_ACTION_CATEGORIES = {
    "equivalent_or_redundant",
    "mutation_not_reflected",
    "test_weakness",
    "generator_fidelity",
    "manual_review",
}
_REGENERATION_CATEGORIES = {
    "mutation_not_reflected",
    "generator_fidelity",
}
_CATEGORY_BY_JOINT_LABEL = {
    "EQUIV_REFLECTED": "equivalent_or_redundant",
    "EQUIV_NOT_REFLECTED": "mutation_not_reflected",
    "NON_EQUIV_REFLECTED": "test_weakness",
    "NON_EQUIV_NOT_REFLECTED": "generator_fidelity",
    "JUDGE_UNKNOWN": "manual_review",
}
_CATEGORY_BY_ACTION_KIND = {
    "downrank_equivalent_mutant": "equivalent_or_redundant",
    "generate_mutation_guided_tests": "test_weakness",
    # Backward compatibility for older survivor_actionable_pipeline JSONL.
    "add_or_select_tests": "test_weakness",
    "manual_review": "manual_review",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_batch_config(batch_dir: Path) -> dict:
    path = batch_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_problem_index(batch_config: dict, *, timeout: int) -> dict[str, Problem]:
    benchmark = batch_config.get("benchmark", "humaneval")
    if benchmark == "livecodebench":
        problems = load_livecodebench(
            cutoff_date=batch_config.get("lcb_cutoff", "2024-01-01"),
            split_file=batch_config.get("lcb_split_file", "test.jsonl"),
            timeout=timeout,
        )
    else:
        problems = load_benchmark(benchmark, timeout=timeout)
    return {problem.task_id: problem for problem in problems}


def _load_mutation_index(batch_dir: Path) -> dict[tuple[str, int], tuple[dict, dict]]:
    index: dict[tuple[str, int], tuple[dict, dict]] = {}
    for row in _load_jsonl(batch_dir / "results.jsonl"):
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        for mutation_idx, mutation in enumerate(row.get("mutations") or []):
            index[(task_id, mutation_idx)] = (row, mutation)
    return index


def default_action_analysis_path(batch_dir: Path) -> Path:
    canonical = batch_dir / "survivor_actionable_pipeline.jsonl"
    if canonical.exists():
        return canonical
    candidates = [
        path
        for path in batch_dir.glob("survivor_actionable_pipeline*.jsonl")
        if not path.name.endswith("_summary.jsonl")
    ]
    if not candidates:
        return canonical
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_categories(raw: str | None) -> set[str]:
    if raw is None or raw.strip() in {"", "all"}:
        return set(_ACTION_CATEGORIES)
    categories = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = categories - _ACTION_CATEGORIES
    if unknown:
        raise ValueError(f"unknown action categories: {sorted(unknown)}")
    return categories


def _action_category(record: dict) -> str | None:
    category = record.get("action_category")
    if isinstance(category, str) and category:
        return category
    joint_label = record.get("joint_label")
    if (
        joint_label == "EQUIV_REFLECTED"
        and isinstance(record.get("mutation_change"), dict)
        and _is_contract_sensitive_change(record["mutation_change"], record)
    ):
        return "test_weakness"
    if isinstance(joint_label, str) and joint_label in _CATEGORY_BY_JOINT_LABEL:
        return _CATEGORY_BY_JOINT_LABEL[joint_label]
    action_kind = record.get("action_kind") or record.get("action")
    if isinstance(action_kind, str):
        if action_kind == "strengthen_mutated_clause_and_regenerate":
            if record.get("equivalent") is False:
                return "generator_fidelity"
            return "mutation_not_reflected"
        return _CATEGORY_BY_ACTION_KIND.get(action_kind)
    return None


def _selected_records(
    records: list[dict],
    *,
    categories: set[str],
    start: int,
    limit: int | None,
) -> list[dict]:
    eligible = [
        record
        for record in records
        if _action_category(record) in categories
    ]
    if start:
        eligible = eligible[start:]
    if limit is not None:
        eligible = eligible[:limit]
    return eligible


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _normalise_confidence(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    confidence = value.strip().lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    return None


def _parse_llm_test_generation_response(response: str) -> tuple[list[dict], str | None]:
    text = _strip_json_fence(response)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return [], "no_json_object"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], "invalid_json"

    tests = payload.get("tests") if isinstance(payload, dict) else None
    if tests is None and isinstance(payload, list):
        tests = payload
    if not isinstance(tests, list):
        return [], "missing_tests_array"
    parsed: list[dict] = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        candidate = dict(test)
        if "confidence" in candidate:
            candidate["confidence"] = _normalise_confidence(candidate.get("confidence"))
        parsed.append(candidate)
    return parsed, None


def _problem_uses_stdin_text(problem: Problem) -> bool:
    if getattr(problem, "mode", "function") == "stdin":
        return True
    for test_input in problem.test_inputs[:5]:
        if isinstance(test_input, str):
            return True
        if (
            isinstance(test_input, (list, tuple))
            and len(test_input) == 1
            and isinstance(test_input[0], str)
        ):
            return True
    return False


def _stdin_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return json.dumps(value, ensure_ascii=False, default=repr)


def _format_seed_examples(problem: Problem, *, limit: int = 5) -> list[dict]:
    examples = []
    for idx, test_input in enumerate(problem.test_inputs[: max(0, limit)]):
        examples.append({"index": idx, "input": _stdin_text(test_input)})
    return examples


def _coerce_candidate_probe(
    candidate: dict,
    problem: Problem,
) -> tuple[Any | None, str | None]:
    raw_input = candidate.get("input")
    if _problem_uses_stdin_text(problem):
        if not isinstance(raw_input, str):
            return None, "input_must_be_stdin_string"
        if getattr(problem, "mode", "function") == "stdin":
            return raw_input, None
        return [raw_input], None
    if isinstance(raw_input, list):
        return raw_input, None
    return None, "input_must_be_argument_list"


def _expectation(candidate: dict, key: str) -> dict:
    raw = candidate.get(key)
    if not isinstance(raw, dict):
        return {"status": "ok", "output": None}
    status = str(raw.get("status", "ok")).strip().lower()
    status = status.replace(" ", "_")
    if status in {"pass", "passes", "normal", "normal_output"}:
        status = "ok"
    elif status in {"error", "raise", "raises", "assert", "assertion"}:
        status = "reject"
    elif status in {"different", "different_from_original", "diff"}:
        status = "different_output"
    if status not in {"ok", "reject", "different_output"}:
        status = "ok"
    return {"status": status, "output": raw.get("output")}


def _matches_expected(
    observed: tuple[str, Any],
    expected: dict,
) -> bool:
    status, output = observed
    expected_status = expected.get("status")
    if expected_status == "reject":
        return status != "ok"
    if expected_status != "ok":
        return False
    if status != "ok":
        return False
    expected_output = expected.get("output")
    if expected_output is None:
        return True
    return _values_equal(output, expected_output)


def _mutant_fails_original_oracle(
    reference: tuple[str, Any],
    mutant: tuple[str, Any],
    expected_original: dict,
) -> bool:
    ref_status, ref_output = reference
    mut_status, mut_output = mutant
    if expected_original.get("status") == "reject":
        return mut_status == "ok"
    if ref_status != "ok":
        return False
    if mut_status != "ok":
        return True
    expected_output = expected_original.get("output")
    if expected_output is not None:
        return not _values_equal(mut_output, expected_output)
    return not _values_equal(mut_output, ref_output)


def _expectations_distinguish(expected_original: dict, expected_mutated: dict) -> bool:
    original_status = expected_original.get("status")
    mutated_status = expected_mutated.get("status")
    if mutated_status == "different_output":
        return True
    if original_status != mutated_status:
        return True
    if mutated_status != "ok":
        return False
    original_output = expected_original.get("output")
    mutated_output = expected_mutated.get("output")
    if mutated_output is None:
        return False
    if original_output is None:
        return True
    return not _values_equal(original_output, mutated_output)


def _observed_same(reference: tuple[str, Any], mutant: tuple[str, Any]) -> bool:
    ref_status, ref_output = reference
    mut_status, mut_output = mutant
    if ref_status != mut_status:
        return False
    if ref_status != "ok":
        return True
    return _values_equal(ref_output, mut_output)


def _matches_mutated_expectation(
    reference: tuple[str, Any],
    mutant: tuple[str, Any],
    expected_original: dict,
    expected_mutated: dict,
) -> bool:
    status, output = mutant
    expected_status = expected_mutated.get("status")
    if expected_status == "different_output":
        if status != "ok":
            return True
        expected_output = expected_mutated.get("output")
        if expected_output is not None:
            return _values_equal(output, expected_output)
        ref_status, ref_output = reference
        if ref_status == "ok":
            return not _values_equal(output, ref_output)
        original_output = expected_original.get("output")
        if original_output is not None:
            return not _values_equal(output, original_output)
        return False
    return _matches_expected(mutant, expected_mutated)


def _short_value(value: Any, *, limit: int = 500) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else json.dumps(
        value,
        ensure_ascii=False,
        default=repr,
    )
    return _truncate_middle(text, limit)


def _mutation_change_for_prompt(
    action_record: dict,
    row: dict,
    mutation: dict,
) -> dict:
    change = _mutation_change_summary(row, mutation)
    existing = action_record.get("mutation_change")
    if isinstance(existing, dict):
        for key, value in existing.items():
            if value not in (None, ""):
                change[key] = value
    return change


def _build_llm_test_generation_messages(
    *,
    action_record: dict,
    row: dict,
    mutation: dict,
    problem: Problem,
    samples: int,
    prior_attempts: list[dict],
) -> list[dict]:
    change = _mutation_change_for_prompt(action_record, row, mutation)
    touched = change.get("anchored_text") or change.get("original_clause") or ""
    payload = {
        "task_id": problem.task_id,
        "entry_point": problem.entry_point,
        "requested_tests": max(1, int(samples or 1)),
        "original_spec": _truncate_middle(row.get("spec", problem.spec), 9000),
        "specification_change": _truncate_middle(change.get("spec_diff", ""), 5000),
        "touched_original_requirement": touched,
        "existing_inputs": _format_seed_examples(problem, limit=5),
    }
    system = (
        "You are a software test engineer adding targeted tests for a "
        "surviving specification mutant. Return only the requested JSON."
    )
    user = (
        "You are adding targeted tests for a surviving specification mutant.\n\n"
        "We have an original natural-language specification and a small mutation "
        "to that specification. The mutation represents a possible fault that "
        "survived the existing tests.\n\n"
        "Original specification:\n"
        f"{payload['original_spec']}\n\n"
        "Specification change:\n"
        f"{payload['specification_change']}\n\n"
        "Touched original requirement:\n"
        f"{payload['touched_original_requirement']}\n\n"
        "Existing inputs, shown only to illustrate the input format:\n"
        f"{json.dumps(payload['existing_inputs'], ensure_ascii=False, indent=2)}\n\n"
        "Task:\n"
        "Write candidate tests that distinguish the original specification from "
        "the mutated specification implied by the change above.\n\n"
        "For each candidate, describe the expected behavior on both sides. A "
        "side may produce normal stdout or reject the input when the touched "
        "requirement is an input contract.\n\n"
        "Focus on the touched requirement. Good tests usually exercise a special "
        "value, missing case, or corner case introduced by the change.\n\n"
        "If the changed requirement is an input contract, a useful test may be "
        "accepted by one specification and rejected by the other, for example by "
        "an assertion or error. Do not use malformed inputs unrelated to the "
        "changed requirement.\n\n"
        "Set confidence to high only when the expected distinction follows "
        "directly from the touched requirement; use low when the test relies on "
        "uncertain interpretation.\n\n"
    )
    if prior_attempts:
        user += (
            "The previous tests did not catch the mutant.\n\n"
            "Observed executions:\n"
            f"{json.dumps(prior_attempts[-12:], ensure_ascii=False, indent=2)}\n\n"
            "Try again. Keep the test focused on the touched requirement. "
            "Prefer a sharper special value, missing case, or corner case. "
            "If the changed requirement affects the input contract, a reject-vs-ok "
            "test is allowed.\n\n"
        )
    user += (
        "Return JSON only:\n"
        "{\n"
        '  "tests": [\n'
        "    {\n"
        '      "input": "<complete stdin string>",\n'
        '      "expected_original": {\n'
        '        "status": "ok | reject",\n'
        '        "output": "<expected stdout if status is ok, otherwise null>"\n'
        "      },\n"
        '      "expected_mutated": {\n'
        '        "status": "ok | reject | different_output",\n'
        '        "output": "<expected stdout if known, otherwise null>"\n'
        "      },\n"
        '      "confidence": "high | medium | low",\n'
        '      "why_this_catches_the_mutant": "<short explanation>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _evaluate_generated_tests(
    *,
    candidates: list[dict],
    row: dict,
    mutation: dict,
    problem: Problem,
    execution_timeout: int,
    max_stored_tests: int,
) -> tuple[list[dict], int, int]:
    prepared: list[tuple[dict, Any]] = []
    evaluations: list[dict] = []
    for candidate in candidates:
        probe, error = _coerce_candidate_probe(candidate, problem)
        if error is not None:
            evaluations.append(
                {
                    "candidate": candidate,
                    "parse_status": error,
                    "kills_mutant": False,
                }
            )
            continue
        prepared.append((candidate, probe))

    if not prepared:
        return evaluations, 0, 0

    probes = [probe for _, probe in prepared]
    mode = getattr(problem, "mode", "function")
    ref_outputs = run_outputs(
        row.get("reference_program", ""),
        problem.entry_point,
        probes,
        timeout=execution_timeout,
        mode=mode,
    )
    mut_outputs = run_outputs(
        mutation.get("mutant_code", ""),
        problem.entry_point,
        probes,
        timeout=execution_timeout,
        mode=mode,
    )

    killing_count = 0
    stored_killing_count = 0
    for (candidate, probe), ref, mut in zip(prepared, ref_outputs, mut_outputs):
        expected_original = _expectation(candidate, "expected_original")
        expected_mutated = _expectation(candidate, "expected_mutated")
        ref_matches_original = _matches_expected(ref, expected_original)
        mutant_fails = _mutant_fails_original_oracle(ref, mut, expected_original)
        expected_mutated_distinguishes = _expectations_distinguish(
            expected_original,
            expected_mutated,
        )
        reference_mutant_same = _observed_same(ref, mut)
        mutant_matches_mutated = _matches_mutated_expectation(
            ref,
            mut,
            expected_original,
            expected_mutated,
        )
        kills = ref_matches_original and mutant_fails
        if kills:
            killing_count += 1
        store_full = kills and stored_killing_count < max(0, int(max_stored_tests))
        if store_full:
            stored_killing_count += 1
        evaluations.append(
            {
                "candidate": candidate if store_full else {
                    "input": candidate.get("input"),
                    "confidence": candidate.get("confidence"),
                    "why_this_catches_the_mutant": candidate.get(
                        "why_this_catches_the_mutant"
                    ),
                },
                "probe": probe if store_full else _short_value(probe),
                "expected_original": expected_original,
                "expected_mutated": expected_mutated,
                "reference_status": ref[0],
                "reference_output": _short_value(ref[1]),
                "mutant_status": mut[0],
                "mutant_output": _short_value(mut[1]),
                "reference_matches_expected_original": ref_matches_original,
                "expected_mutated_distinguishes_original": (
                    expected_mutated_distinguishes
                ),
                "mutant_matches_expected_mutated": mutant_matches_mutated,
                "reference_mutant_observably_same": reference_mutant_same,
                "mutant_fails_original_oracle": mutant_fails,
                "kills_mutant": kills,
            }
        )
    return evaluations, killing_count, len(prepared)


def _execution_sanity_check(evaluations: list[dict]) -> dict:
    executable = [
        evaluation
        for evaluation in evaluations
        if evaluation.get("parse_status") is None
    ]
    distinguishing = [
        evaluation
        for evaluation in executable
        if evaluation.get("reference_matches_expected_original")
        and evaluation.get("expected_mutated_distinguishes_original")
    ]
    observed_divergence = [
        evaluation
        for evaluation in distinguishing
        if not evaluation.get("reference_mutant_observably_same")
    ]
    matched_mutated = [
        evaluation
        for evaluation in distinguishing
        if evaluation.get("mutant_matches_expected_mutated")
    ]
    high_confidence = [
        evaluation
        for evaluation in distinguishing
        if (evaluation.get("candidate") or {}).get("confidence") == "high"
    ]
    if distinguishing and not observed_divergence:
        label = "judge_misclassified_or_mutation_not_reflected"
        reason = (
            "Generated tests expected a behavioral distinction, but executing "
            "the reference and mutant programs showed no observable divergence."
        )
    elif distinguishing:
        label = "observed_mutant_divergence"
        reason = (
            "At least one generated test produced different reference and "
            "mutant behavior."
        )
    elif executable:
        label = "no_distinguishing_generated_test"
        reason = (
            "Generated tests executed, but none encoded a checked distinction "
            "between the original and mutated behavior."
        )
    else:
        label = "no_executable_generated_test"
        reason = "No generated test could be executed."
    return {
        "label": label,
        "reason": reason,
        "executable_test_count": len(executable),
        "distinguishing_candidate_count": len(distinguishing),
        "high_confidence_distinguishing_candidate_count": len(high_confidence),
        "observed_divergence_count": len(observed_divergence),
        "mutant_matches_expected_mutated_count": len(matched_mutated),
    }


def _run_llm_test_generation_action(
    action_record: dict,
    row: dict,
    mutation: dict,
    problem: Problem,
    *,
    client: LLMClient,
    test_generation_model: str,
    test_generation_temperature: float | None,
    test_generation_reasoning_effort: str | None,
    test_generation_max_tokens: int,
    test_generation_rounds: int,
    test_generation_samples: int,
    execution_timeout: int,
    max_generated_tests: int,
) -> dict:
    prior_attempts: list[dict] = []
    all_evaluations: list[dict] = []
    parse_errors: Counter[str] = Counter()
    llm_calls = 0
    generated_count = 0
    executable_count = 0
    killing_count = 0

    rounds = max(1, int(test_generation_rounds or 1))
    for round_idx in range(rounds):
        messages = _build_llm_test_generation_messages(
            action_record=action_record,
            row=row,
            mutation=mutation,
            problem=problem,
            samples=test_generation_samples,
            prior_attempts=prior_attempts,
        )
        try:
            response = client.call(
                messages,
                model=test_generation_model,
                max_tokens=test_generation_max_tokens,
                temperature=test_generation_temperature,
                reasoning_effort=test_generation_reasoning_effort,
                cache_salt=(
                    f"{_METHOD}:llm-testgen:{row.get('task_id')}:"
                    f"{action_record.get('mutation_idx')}:{round_idx}:"
                    f"{test_generation_model}"
                ),
            )
            llm_calls += 1
        except Exception as exc:
            return {
                "experiment_kind": "llm_mutation_guided_test_generation",
                "experiment_status": "llm_test_generation_failed",
                "success": False,
                "evidence": {
                    "llm_calls": llm_calls,
                    "test_generation_model": test_generation_model,
                    "test_generation_reasoning_effort": (
                        test_generation_reasoning_effort
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            }

        candidates, parse_error = _parse_llm_test_generation_response(response)
        if parse_error is not None:
            parse_errors[parse_error] += 1
            prior_attempts.append(
                {
                    "round": round_idx,
                    "parse_error": parse_error,
                    "raw_response": _truncate_middle(response, 1000),
                }
            )
            continue

        generated_count += len(candidates)
        evaluations, round_killing_count, round_executable_count = (
            _evaluate_generated_tests(
                candidates=candidates,
                row=row,
                mutation=mutation,
                problem=problem,
                execution_timeout=execution_timeout,
                max_stored_tests=max_generated_tests,
            )
        )
        executable_count += round_executable_count
        killing_count += round_killing_count
        all_evaluations.extend(evaluations)
        prior_attempts.extend(
            {
                "round": round_idx,
                "input": evaluation.get("candidate", {}).get("input"),
                "confidence": evaluation.get("candidate", {}).get("confidence"),
                "reference_status": evaluation.get("reference_status"),
                "reference_output": evaluation.get("reference_output"),
                "mutant_status": evaluation.get("mutant_status"),
                "mutant_output": evaluation.get("mutant_output"),
                "kills_mutant": evaluation.get("kills_mutant"),
            }
            for evaluation in evaluations
            if evaluation.get("parse_status") is None
        )
        if round_killing_count:
            break

    success = killing_count > 0
    sanity_check = _execution_sanity_check(all_evaluations)
    if success:
        experiment_status = "generated_killing_tests"
    elif sanity_check["label"] == "judge_misclassified_or_mutation_not_reflected":
        experiment_status = "judge_misclassified_or_mutation_not_reflected"
    else:
        experiment_status = "no_generated_killing_test"
    return {
        "experiment_kind": "llm_mutation_guided_test_generation",
        "experiment_status": experiment_status,
        "success": success,
        "evidence": {
            "llm_calls": llm_calls,
            "test_generation_model": test_generation_model,
            "test_generation_temperature": test_generation_temperature,
            "test_generation_rounds": rounds,
            "test_generation_samples": test_generation_samples,
            "generated_test_count": generated_count,
            "executable_test_count": executable_count,
            "generated_killing_test_count": killing_count,
            "stored_evaluation_count": len(all_evaluations),
            "parse_error_counts": dict(parse_errors),
            "execution_sanity_check": sanity_check,
            "evaluations": all_evaluations,
            "note": (
                "Tests were generated by an LLM from the original specification, "
                "the focused specification diff, and the touched original "
                "requirement. Benchmark held-out tests were not used."
            ),
        },
    }


def _first_added_diff_line(spec_diff: str | None) -> str | None:
    for line in (spec_diff or "").splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        candidate = line[1:].strip()
        if candidate:
            return candidate
    return None


def _candidate_mutated_anchor(action_record: dict, mutation: dict) -> str | None:
    change = action_record.get("mutation_change") or {}
    anchor = change.get("anchored_text") or change.get("original_clause")
    original = mutation.get("slot_original")
    target = mutation.get("slot_target")
    if (
        isinstance(anchor, str)
        and isinstance(original, str)
        and isinstance(target, str)
        and original in anchor
    ):
        return anchor.replace(original, target, 1).strip()
    return None


def _guard_requirement_line(spec: str, target: str | None) -> str | None:
    if not spec or not target:
        return None
    target = target.strip()
    if not target or _REQUIREMENT_GUARD_OPEN in spec:
        return spec
    lines = spec.splitlines()
    insert_at: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == target or target in stripped:
            insert_at = idx
            break
    if insert_at is None:
        return None

    line = lines[insert_at]
    indent = line[: len(line) - len(line.lstrip())]
    lines[insert_at:insert_at + 1] = [
        indent + _ACTION_EMPHASIS,
        indent + _REQUIREMENT_GUARD_OPEN,
        line,
        indent + _REQUIREMENT_GUARD_CLOSE,
    ]
    out = "\n".join(lines)
    if spec.endswith("\n"):
        out += "\n"
    return out


def _valid_requirement_target(target: str | None) -> str | None:
    if not isinstance(target, str):
        return None
    target = target.strip()
    return target or None


def _prepend_implementation_focus(spec: str, target: str | None) -> str:
    target = _valid_requirement_target(target)
    if not spec or not target or _IMPLEMENTATION_FOCUS_TITLE in spec:
        return spec
    return "\n".join([
        _IMPLEMENTATION_FOCUS_TITLE,
        _IMPLEMENTATION_FOCUS_NOTE,
        _FOCUS_REQUIREMENT_OPEN,
        target,
        _FOCUS_REQUIREMENT_CLOSE,
        "",
        spec,
    ])


def _emphasized_mutated_spec(action_record: dict, mutation: dict) -> str:
    mutated_spec = mutation.get("mutated_spec", "") or ""
    change = action_record.get("mutation_change") or {}
    targets = [
        _first_added_diff_line(change.get("spec_diff")),
        _candidate_mutated_anchor(action_record, mutation),
    ]
    valid_targets = [
        target for target in (_valid_requirement_target(target) for target in targets)
        if target is not None
    ]
    for target in targets:
        emphasized = _guard_requirement_line(mutated_spec, target)
        if emphasized is not None:
            return _prepend_implementation_focus(emphasized, target)
    if _REQUIREMENT_GUARD_OPEN in mutated_spec:
        fallback = mutated_spec
    else:
        fallback = "\n".join([
            _ACTION_EMPHASIS,
            _REQUIREMENT_GUARD_OPEN,
            mutated_spec,
            _REQUIREMENT_GUARD_CLOSE,
        ])
    if valid_targets:
        return _prepend_implementation_focus(fallback, valid_targets[0])
    return fallback


def _regeneration_prompt(problem: Problem, codegen_spec: str) -> str:
    if getattr(problem, "mode", "function") == "stdin":
        return ""
    return replace_docstring(problem.prompt, codegen_spec)


def _run_regeneration_action(
    action_record: dict,
    row: dict,
    mutation: dict,
    problem: Problem,
    *,
    client: LLMClient,
    regeneration_model: str,
    regeneration_temperature: float | None,
    regeneration_reasoning_effort: str | None,
    execution_timeout: int,
    judge_regenerated: bool,
    judge_model: str,
    judge_reasoning_effort: str | None,
    judge_max_tokens: int,
    judge_code_chars: int,
) -> dict:
    if not mutation.get("mutated_spec"):
        return {
            "experiment_kind": "emphasis_regeneration",
            "experiment_status": "missing_mutated_spec",
            "success": False,
            "evidence": {},
        }

    codegen_spec = _emphasized_mutated_spec(action_record, mutation)
    mode = getattr(problem, "mode", "function")
    try:
        regenerated_code = generate(
            client,
            codegen_spec,
            _regeneration_prompt(problem, codegen_spec),
            problem.entry_point,
            model=regeneration_model,
            mode=mode,
            cache_salt=(
                f"{_METHOD}:regen:{row.get('task_id')}:"
                f"{action_record.get('mutation_idx')}:{regeneration_model}"
            ),
            temperature=regeneration_temperature,
        )
    except Exception as exc:
        return {
            "experiment_kind": "emphasis_regeneration",
            "experiment_status": "codegen_failed",
            "success": False,
            "evidence": {
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
        }

    test_results = run_tests(
        regenerated_code,
        problem.entry_point,
        problem.test_inputs,
        problem.expected_outputs,
        timeout=execution_timeout,
        mode=mode,
    )
    passed = sum(1 for result in test_results if result)
    total = len(test_results)
    survived_existing_tests = bool(total) and passed == total
    evidence: dict[str, Any] = {
        "regeneration_model": regeneration_model,
        "regeneration_temperature": regeneration_temperature,
        "regeneration_reasoning_effort": regeneration_reasoning_effort,
        "emphasis_mode": "implementation_focus_and_localized_guarded_requirement",
        "emphasis_note": _ACTION_EMPHASIS,
        "implementation_focus_note": _IMPLEMENTATION_FOCUS_NOTE,
        "regenerated_mutant_code": regenerated_code,
        "benchmark_passed": passed,
        "benchmark_total": total,
        "survived_existing_tests": survived_existing_tests,
        "killed_by_existing_tests": not survived_existing_tests,
    }

    success: bool | None = None
    status = "generated_unjudged"
    if judge_regenerated:
        judged_mutation = dict(mutation)
        judged_mutation["mutant_code"] = regenerated_code
        try:
            response = client.call(
                _build_joint_messages(
                    row=row,
                    mutation=judged_mutation,
                    prompt_context="spec-aware",
                    code_chars=judge_code_chars,
                ),
                model=judge_model,
                max_tokens=judge_max_tokens,
                temperature=0,
                reasoning_effort=judge_reasoning_effort,
                cache_salt=(
                    f"{_METHOD}:judge-regenerated:{row.get('task_id')}:"
                    f"{action_record.get('mutation_idx')}:{judge_model}"
                ),
            )
            pred = _parse_joint_response(response)
            evidence.update(
                {
                    "post_action_judge_model": judge_model,
                    "post_action_judge_reasoning_effort": judge_reasoning_effort,
                    "post_action_prediction": pred.prediction,
                    "post_action_equivalent": pred.equivalent,
                    "post_action_mutation_reflected": pred.mutation_reflected,
                    "post_action_joint_label": pred.joint_label,
                    "post_action_confidence": pred.confidence,
                    "post_action_reason": pred.reason,
                    "post_action_raw_response": pred.raw_response,
                }
            )
            success = pred.mutation_reflected is True
            status = (
                "regenerated_reflects_mutation"
                if success
                else "regenerated_not_reflected"
            )
        except Exception as exc:
            evidence.update(
                {
                    "post_action_judge_model": judge_model,
                    "post_action_error_type": type(exc).__name__,
                    "post_action_error": str(exc)[:500],
                }
            )
            success = False
            status = "post_action_judge_failed"

    return {
        "experiment_kind": "emphasis_regeneration",
        "experiment_status": status,
        "success": success,
        "evidence": evidence,
    }


def _account_equivalent(action_record: dict) -> dict:
    return {
        "experiment_kind": "equivalent_accounting",
        "experiment_status": "downranked_equivalent_or_redundant",
        "success": None,
        "evidence": {
            "note": (
                "No code-generation action is run for this category. The "
                "mutant is counted separately from test-weakness evidence."
            ),
            "judge_reason": action_record.get("judge_reason"),
            "judge_confidence": action_record.get("judge_confidence"),
        },
    }


def _manual_review(action_record: dict) -> dict:
    return {
        "experiment_kind": "manual_review",
        "experiment_status": "manual_review_required",
        "success": None,
        "evidence": {
            "judge_reason": action_record.get("judge_reason"),
            "judge_confidence": action_record.get("judge_confidence"),
        },
    }


def dry_run_summary(
    batch_dir: Path,
    *,
    action_analysis: Path,
    categories: set[str],
    start: int = 0,
    limit: int | None = None,
    judge_regenerated: bool = False,
    attempts: int | None = None,
    test_generation_rounds: int = _DEFAULT_TEST_GENERATION_ROUNDS,
) -> dict:
    action_rows = _load_jsonl(action_analysis)
    selected = _selected_records(
        action_rows,
        categories=categories,
        start=start,
        limit=limit,
    )
    category_counts = Counter(_action_category(record) for record in selected)
    regeneration_count = sum(
        1
        for record in selected
        if _action_category(record) in _REGENERATION_CATEGORIES
    )
    test_generation_count = sum(
        1
        for record in selected
        if _action_category(record) == "test_weakness"
    )
    judge_calls = regeneration_count if judge_regenerated else 0
    max_attempts = max(1, int(attempts if attempts else Config().llm_max_attempts))
    logical_llm_calls = (
        regeneration_count
        + judge_calls
        + test_generation_count * max(1, int(test_generation_rounds or 1))
    )
    return {
        "batch": str(batch_dir),
        "method": _METHOD,
        "dry_run": True,
        "action_analysis": str(action_analysis),
        "selected_actions": len(selected),
        "category_counts": dict(category_counts),
        "regeneration_codegen_calls": regeneration_count,
        "test_generation_calls": test_generation_count,
        "test_generation_rounds": max(1, int(test_generation_rounds or 1)),
        "post_action_judge_calls": judge_calls,
        "logical_llm_calls": logical_llm_calls,
        "max_attempts_per_llm_call": max_attempts,
        "max_external_llm_api_calls": logical_llm_calls * max_attempts,
        "start": start,
        "limit": limit,
        "categories": sorted(categories),
    }


def run_action_experiments(
    batch_dir: Path,
    *,
    action_analysis: Path | None = None,
    output: Path | None = None,
    categories: set[str] | None = None,
    start: int = 0,
    limit: int | None = None,
    dry_run: bool = False,
    regeneration_model: str | None = None,
    regeneration_temperature: float | None = 0,
    regeneration_reasoning_effort: str | None = None,
    test_generation_model: str | None = None,
    test_generation_temperature: float | None = 0,
    test_generation_reasoning_effort: str | None = None,
    test_generation_max_tokens: int = _DEFAULT_TEST_GENERATION_MAX_TOKENS,
    test_generation_rounds: int = _DEFAULT_TEST_GENERATION_ROUNDS,
    test_generation_samples: int = _DEFAULT_TEST_GENERATION_SAMPLES,
    judge_regenerated: bool = False,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    judge_reasoning_effort: str | None = None,
    judge_max_tokens: int = _DEFAULT_MAX_TOKENS,
    judge_code_chars: int = _DEFAULT_CODE_CHARS,
    execution_timeout: int | None = None,
    llm_timeout: float | None = None,
    llm_attempts: int | None = None,
    max_generated_tests: int = 10,
    stream_output: bool = False,
    progress_every: int = 0,
) -> tuple[list[dict], dict]:
    batch_dir = batch_dir.resolve()
    if not (batch_dir / "results.jsonl").exists():
        raise FileNotFoundError(f"no results.jsonl found under {batch_dir}")

    action_analysis = (action_analysis or default_action_analysis_path(batch_dir)).resolve()
    if not action_analysis.exists():
        raise FileNotFoundError(f"no action analysis JSONL found at {action_analysis}")

    categories = categories or set(_ACTION_CATEGORIES)
    if dry_run:
        return [], dry_run_summary(
            batch_dir,
            action_analysis=action_analysis,
            categories=categories,
            start=start,
            limit=limit,
            judge_regenerated=judge_regenerated,
            attempts=llm_attempts,
            test_generation_rounds=test_generation_rounds,
        )

    batch_config = _load_batch_config(batch_dir)
    config = Config()
    if llm_timeout is not None:
        config.llm_timeout_seconds = llm_timeout
    if llm_attempts is not None:
        config.llm_max_attempts = max(1, int(llm_attempts))
    regeneration_model = (
        regeneration_model
        or batch_config.get("model_generator")
        or config.model_generator
    )
    if regeneration_reasoning_effort is None:
        regeneration_reasoning_effort = batch_config.get(
            "generator_reasoning_effort"
        )
    test_generation_model = (
        test_generation_model
        or batch_config.get("model_tooling")
        or config.model_tooling
        or regeneration_model
    )
    if test_generation_reasoning_effort is None:
        test_generation_reasoning_effort = batch_config.get(
            "tooling_reasoning_effort"
        )
    config.model_generator = regeneration_model
    config.generator_reasoning_effort = regeneration_reasoning_effort
    config.model_tooling = test_generation_model
    config.tooling_reasoning_effort = test_generation_reasoning_effort
    execution_timeout = int(
        execution_timeout
        if execution_timeout is not None
        else batch_config.get("execution_timeout", config.execution_timeout)
    )

    mutation_index = _load_mutation_index(batch_dir)
    problem_index = _load_problem_index(batch_config, timeout=execution_timeout)
    action_rows = _load_jsonl(action_analysis)
    selected = _selected_records(
        action_rows,
        categories=categories,
        start=start,
        limit=limit,
    )
    client = LLMClient(config)
    output = output or (batch_dir / _DEFAULT_OUTPUT)

    records: list[dict] = []
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    missing_mutation_records = 0
    missing_problems = 0
    logical_llm_calls = 0

    stream_handle = None
    if stream_output:
        output.parent.mkdir(parents=True, exist_ok=True)
        stream_handle = output.open("w")

    try:
        for index, action_record in enumerate(selected, start=1):
            task_id = action_record.get("task_id")
            mutation_idx = action_record.get("mutation_idx")
            action_category = _action_category(action_record)
            base = {
                "task_id": task_id,
                "mutation_idx": mutation_idx,
                "method": _METHOD,
                "source_action_analysis": str(action_analysis),
                "action_category": action_category,
                "action_kind": action_record.get("action_kind"),
                "joint_label": action_record.get("joint_label"),
                "pre_action_equivalent": action_record.get("equivalent"),
                "pre_action_mutation_reflected": action_record.get(
                    "mutation_reflected"
                ),
                "pre_action_judge_confidence": action_record.get(
                    "judge_confidence"
                ),
            }

            indexed = (
                mutation_index.get((task_id, mutation_idx))
                if isinstance(task_id, str) and isinstance(mutation_idx, int)
                else None
            )
            problem = problem_index.get(task_id) if isinstance(task_id, str) else None
            if indexed is None:
                missing_mutation_records += 1
                result = {
                    "experiment_kind": "lookup",
                    "experiment_status": "missing_mutation_record",
                    "success": False,
                    "evidence": {},
                }
            elif problem is None:
                missing_problems += 1
                result = {
                    "experiment_kind": "lookup",
                    "experiment_status": "missing_benchmark_problem",
                    "success": False,
                    "evidence": {},
                }
            else:
                row, mutation = indexed
                if action_category == "test_weakness":
                    result = _run_llm_test_generation_action(
                        action_record,
                        row,
                        mutation,
                        problem,
                        client=client,
                        test_generation_model=test_generation_model,
                        test_generation_temperature=test_generation_temperature,
                        test_generation_reasoning_effort=(
                            test_generation_reasoning_effort
                        ),
                        test_generation_max_tokens=test_generation_max_tokens,
                        test_generation_rounds=test_generation_rounds,
                        test_generation_samples=test_generation_samples,
                        execution_timeout=execution_timeout,
                        max_generated_tests=max(0, int(max_generated_tests)),
                    )
                    logical_llm_calls += int(
                        result.get("evidence", {}).get("llm_calls", 0)
                    )
                elif action_category in _REGENERATION_CATEGORIES:
                    logical_llm_calls += 1
                    if judge_regenerated:
                        logical_llm_calls += 1
                    result = _run_regeneration_action(
                        action_record,
                        row,
                        mutation,
                        problem,
                        client=client,
                        regeneration_model=regeneration_model,
                        regeneration_temperature=regeneration_temperature,
                        regeneration_reasoning_effort=(
                            regeneration_reasoning_effort
                        ),
                        execution_timeout=execution_timeout,
                        judge_regenerated=judge_regenerated,
                        judge_model=judge_model,
                        judge_reasoning_effort=judge_reasoning_effort,
                        judge_max_tokens=judge_max_tokens,
                        judge_code_chars=judge_code_chars,
                    )
                elif action_category == "equivalent_or_redundant":
                    result = _account_equivalent(action_record)
                else:
                    result = _manual_review(action_record)

            record = {**base, **result}
            records.append(record)
            status_counts[record["experiment_status"]] += 1
            category_counts[str(action_category)] += 1
            kind_counts[record["experiment_kind"]] += 1
            success_counts[str(record["success"])] += 1

            if stream_handle is not None:
                stream_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream_handle.flush()
            if progress_every and index % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "status_counts": dict(status_counts),
                            "success_counts": dict(success_counts),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if stream_handle is not None:
            stream_handle.close()

    max_attempts = max(1, int(config.llm_max_attempts))
    summary = {
        "batch": str(batch_dir),
        "method": _METHOD,
        "action_analysis": str(action_analysis),
        "output": str(output),
        "total_action_records": len(action_rows),
        "selected_actions": len(selected),
        "emitted_records": len(records),
        "categories": sorted(categories),
        "start": start,
        "limit": limit,
        "status_counts": dict(status_counts),
        "action_category_counts": dict(category_counts),
        "experiment_kind_counts": dict(kind_counts),
        "success_counts": dict(success_counts),
        "missing_mutation_records": missing_mutation_records,
        "missing_benchmark_problems": missing_problems,
        "regeneration_model": regeneration_model,
        "regeneration_temperature": regeneration_temperature,
        "regeneration_reasoning_effort": regeneration_reasoning_effort,
        "test_generation_model": test_generation_model,
        "test_generation_temperature": test_generation_temperature,
        "test_generation_reasoning_effort": test_generation_reasoning_effort,
        "test_generation_max_tokens": test_generation_max_tokens,
        "test_generation_rounds": test_generation_rounds,
        "test_generation_samples": test_generation_samples,
        "judge_regenerated": judge_regenerated,
        "judge_model": judge_model if judge_regenerated else None,
        "judge_reasoning_effort": (
            judge_reasoning_effort if judge_regenerated else None
        ),
        "logical_llm_calls": logical_llm_calls,
        "max_attempts_per_llm_call": max_attempts,
        "max_external_llm_api_calls": logical_llm_calls * max_attempts,
        "execution_timeout": execution_timeout,
        "max_generated_tests_stored": max_generated_tests,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute post-hoc developer actions for survived-mutant "
            "action categories."
        )
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--action-analysis", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--categories",
        default="all",
        help=(
            "Comma-separated action categories to run, or 'all'. Categories: "
            + ", ".join(sorted(_ACTION_CATEGORIES))
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regeneration-model", type=str, default=None)
    parser.add_argument("--regeneration-temperature", type=float, default=0)
    parser.add_argument(
        "--regeneration-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Optional reasoning_effort for the regeneration model. "
            "Defaults to the batch generator_reasoning_effort when present."
        ),
    )
    parser.add_argument("--test-generation-model", type=str, default=None)
    parser.add_argument("--test-generation-temperature", type=float, default=0)
    parser.add_argument(
        "--test-generation-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Optional reasoning_effort for the test-generation model. "
            "Defaults to the batch tooling_reasoning_effort when present."
        ),
    )
    parser.add_argument(
        "--test-generation-max-tokens",
        type=int,
        default=_DEFAULT_TEST_GENERATION_MAX_TOKENS,
    )
    parser.add_argument(
        "--test-generation-rounds",
        type=int,
        default=_DEFAULT_TEST_GENERATION_ROUNDS,
    )
    parser.add_argument(
        "--test-generation-samples",
        type=int,
        default=_DEFAULT_TEST_GENERATION_SAMPLES,
    )
    parser.add_argument("--judge-regenerated", action="store_true")
    parser.add_argument("--judge-model", type=str, default=_DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning_effort for the post-regeneration judge.",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--judge-code-chars", type=int, default=_DEFAULT_CODE_CHARS)
    parser.add_argument("--execution-timeout", type=int, default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-attempts", type=int, default=None)
    parser.add_argument(
        "--max-generated-tests",
        type=int,
        default=10,
        help=(
            "Maximum generated killing tests to store with full candidate "
            "details per mutant. Counts still include all generated killers."
        ),
    )
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    try:
        categories = _parse_categories(args.categories)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    records, summary = run_action_experiments(
        args.batch_dir,
        action_analysis=args.action_analysis,
        output=args.output,
        categories=categories,
        start=args.start,
        limit=args.limit,
        dry_run=args.dry_run,
        regeneration_model=args.regeneration_model,
        regeneration_temperature=args.regeneration_temperature,
        regeneration_reasoning_effort=args.regeneration_reasoning_effort,
        test_generation_model=args.test_generation_model,
        test_generation_temperature=args.test_generation_temperature,
        test_generation_reasoning_effort=args.test_generation_reasoning_effort,
        test_generation_max_tokens=args.test_generation_max_tokens,
        test_generation_rounds=args.test_generation_rounds,
        test_generation_samples=args.test_generation_samples,
        judge_regenerated=args.judge_regenerated,
        judge_model=args.judge_model,
        judge_reasoning_effort=args.judge_reasoning_effort,
        judge_max_tokens=args.judge_max_tokens,
        judge_code_chars=args.judge_code_chars,
        execution_timeout=args.execution_timeout,
        llm_timeout=args.llm_timeout,
        llm_attempts=args.llm_attempts,
        max_generated_tests=args.max_generated_tests,
        stream_output=args.stream_output,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    output = args.output or (args.batch_dir.resolve() / _DEFAULT_OUTPUT)
    _write_jsonl(output, records)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
