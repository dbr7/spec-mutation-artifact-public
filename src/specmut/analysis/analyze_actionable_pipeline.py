"""Post-hoc actionable routing for survived NL-spec mutants.

This script consumes the joint equivalence/reflection judgment produced by
``specmut.analysis.analyze_equivalence_emd_prompt`` and turns it into a
developer-facing action report. It does not change canonical experiment results
or generate new tests; it only writes a sidecar JSONL plus summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specmut.analysis.analyze_equivalence_emd_prompt import (
    _action_from_joint_label,
    _joint_label_from_bools,
    _load_jsonl,
    _mutation_change_summary,
)


_METHOD = "survivor_actionable_pipeline_v2"
_DEFAULT_OUTPUT = "survivor_actionable_pipeline.jsonl"
_DEFAULT_SUMMARY = "survivor_actionable_pipeline_summary.json"
_CONTRACT_CLAUSE_TYPES = {"boundary", "precondition"}
_CONTRACT_TEXT_RE = re.compile(
    r"(?i)\b("
    r"input|constraints?|given|satisf(?:y|ies)|valid\s+inputs?|allowed|"
    r"precondition|assume|consists?\s+of|consisting\s+of|"
    r"length|integer|string|letters?|digits?|uppercase|lowercase|"
    r"positive|negative|non-negative|between"
    r")\b|<=|>=|\\leq|\\geq|≤|≥"
)


@dataclass
class ActionRoute:
    joint_label: str
    action_category: str
    action_category_label: str
    action_kind: str
    developer_insight: str
    priority: str


_ROUTES = {
    "EQUIV_REFLECTED": ActionRoute(
        joint_label="EQUIV_REFLECTED",
        action_category="equivalent_or_redundant",
        action_category_label="Equivalent or redundant mutant",
        action_kind="downrank_equivalent_mutant",
        developer_insight=(
            "The intended mutation appears reflected in code, but the generated "
            "reference and mutant programs are judged equivalent. Treat this as "
            "an equivalent or redundant mutant candidate, not as test weakness."
        ),
        priority="low",
    ),
    "EQUIV_NOT_REFLECTED": ActionRoute(
        joint_label="EQUIV_NOT_REFLECTED",
        action_category="mutation_not_reflected",
        action_category_label="Mutation not reflected",
        action_kind="strengthen_mutated_clause_and_regenerate",
        developer_insight=(
            "The mutant program is judged equivalent and the intended mutation "
            "does not appear reflected in code. Regenerate with the mutated "
            "clause emphasized before drawing conclusions."
        ),
        priority="medium",
    ),
    "NON_EQUIV_REFLECTED": ActionRoute(
        joint_label="NON_EQUIV_REFLECTED",
        action_category="test_weakness",
        action_category_label="Test weakness",
        action_kind="generate_mutation_guided_tests",
        developer_insight=(
            "The mutant program is judged non-equivalent and the intended "
            "mutation appears reflected in code. Treat this as test weakness "
            "evidence and try mutation-guided test generation focused on the "
            "changed requirement."
        ),
        priority="high",
    ),
    "NON_EQUIV_NOT_REFLECTED": ActionRoute(
        joint_label="NON_EQUIV_NOT_REFLECTED",
        action_category="generator_fidelity",
        action_category_label="Generator fidelity",
        action_kind="strengthen_mutated_clause_and_regenerate",
        developer_insight=(
            "The programs are judged non-equivalent, but the difference does "
            "not appear caused by the intended mutation. Treat this as a "
            "generator-fidelity issue before using it as test evidence."
        ),
        priority="medium",
    ),
    "JUDGE_UNKNOWN": ActionRoute(
        joint_label="JUDGE_UNKNOWN",
        action_category="manual_review",
        action_category_label="Manual review",
        action_kind="manual_review",
        developer_insight=(
            "The joint judge did not produce a usable equivalence/reflection "
            "decision. Exclude this mutant from automated action counts or "
            "review it manually."
        ),
        priority="review",
    ),
}

_CONTRACT_TEST_ROUTE = ActionRoute(
    joint_label="EQUIV_REFLECTED",
    action_category="test_weakness",
    action_category_label="Contract-test weakness",
    action_kind="generate_mutation_guided_tests",
    developer_insight=(
        "The intended mutation appears reflected in code and touches an input "
        "contract, boundary, or precondition. Even if the programs are judged "
        "equivalent on inputs valid under the original specification, treat "
        "this as a contract-test candidate rather than downranking it as a "
        "redundant mutant."
    ),
    priority="high",
)


def _is_contract_sensitive_change(mutation_change: dict, mutation_meta: dict) -> bool:
    """True when an EQUIV_REFLECTED mutation should still drive test action.

    The joint EMD prompt judges equivalence only on inputs valid under the
    original specification. That is right for semantic equivalence, but it
    would downrank mutants that intentionally change an input bound, domain, or
    precondition. Those mutants are useful contract-test candidates.
    """
    clause_type = mutation_change.get("clause_type")
    if clause_type in _CONTRACT_CLAUSE_TYPES:
        return True

    slot_type = mutation_meta.get("slot_type") or mutation_change.get("slot_type")
    text = " ".join(
        str(mutation_change.get(key) or "")
        for key in ("original_clause", "anchored_text", "spec_diff")
    )
    if not _CONTRACT_TEXT_RE.search(text):
        return False
    # Numeric/comparison perturbations and type/domain substitutions are the
    # contract-sensitive cases. Avoid treating generic output changes such as
    # "Print 0" as contract tests just because a number changed.
    return (
        mutation_change.get("operator") == "perturb"
        or str(slot_type or "").startswith(
            (
                "classic_numeric",
                "classic_comparison",
                "classic_type",
                "classic_class",
                "classic_case",
            )
        )
    )


def _route_for_record(joint_label: str, mutation_change: dict, mutation_meta: dict) -> ActionRoute:
    route = _ROUTES[joint_label]
    if joint_label == "EQUIV_REFLECTED" and _is_contract_sensitive_change(
        mutation_change,
        mutation_meta,
    ):
        return _CONTRACT_TEST_ROUTE
    return route


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_behavior_index(path: Path | None) -> dict[tuple[str, int], dict]:
    if path is None or not path.exists():
        return {}
    index: dict[tuple[str, int], dict] = {}
    for row in _load_jsonl(path):
        task_id = row.get("task_id")
        mutation_idx = row.get("mutation_idx")
        if isinstance(task_id, str) and isinstance(mutation_idx, int):
            index[(task_id, mutation_idx)] = row
    return index


def _load_mutation_index(batch_dir: Path) -> dict[tuple[str, int], tuple[dict, dict]]:
    index: dict[tuple[str, int], tuple[dict, dict]] = {}
    for row in _load_jsonl(batch_dir / "results.jsonl"):
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        for mutation_idx, mutation in enumerate(row.get("mutations") or []):
            index[(task_id, mutation_idx)] = (row, mutation)
    return index


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "y", "1"}:
            return True
        if lower in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return None


def _normalise_joint_label(judge_record: dict) -> str:
    raw = judge_record.get("joint_label")
    if raw in _ROUTES and raw != "JUDGE_UNKNOWN":
        return str(raw)
    equivalent = _coerce_bool(judge_record.get("equivalent"))
    reflected = _coerce_bool(judge_record.get("mutation_reflected"))
    return _joint_label_from_bools(equivalent, reflected) or "JUDGE_UNKNOWN"


def _regeneration_payload(mutation_change: dict) -> dict:
    return {
        "emphasis_target": mutation_change.get("original_clause")
        or mutation_change.get("anchored_text"),
        "intended_change": mutation_change.get("intended_change"),
        "instruction": (
            "Regenerate the mutant program while treating the mutation change "
            "as mandatory. The generated code should implement the changed "
            "requirement, not merely preserve the original behavior."
        ),
    }


def _action_payload(
    *,
    route: ActionRoute,
    mutation_change: dict,
    behavior: dict,
) -> dict:
    if route.action_kind == "generate_mutation_guided_tests":
        return {
            "note": (
                "Generate targeted tests from the original specification, "
                "the focused specification diff, and the touched original "
                "requirement. This action does not use benchmark held-out "
                "tests or probe-search side effects."
            ),
        }
    if route.action_kind == "strengthen_mutated_clause_and_regenerate":
        return _regeneration_payload(mutation_change)
    if route.action_kind == "downrank_equivalent_mutant":
        return {
            "note": (
                "Keep this mutant for equivalent-mutant accounting, but do not "
                "treat it as evidence that tests are weak."
            )
        }
    return {}


def _evidence_policy(route: ActionRoute) -> dict:
    return {
        "counts_as_test_weakness": route.action_category == "test_weakness",
        "requires_regeneration": route.action_kind
        == "strengthen_mutated_clause_and_regenerate",
        "requires_test_action": route.action_kind
        == "generate_mutation_guided_tests",
        "counts_as_equivalent_mutant": route.action_category
        == "equivalent_or_redundant",
    }


def build_action_records(
    batch_dir: Path,
    *,
    joint_analysis: Path,
    behavior_analysis: Path | None = None,
) -> tuple[list[dict], dict]:
    mutation_index = _load_mutation_index(batch_dir)
    behavior_index = _load_behavior_index(behavior_analysis)
    joint_rows = _load_jsonl(joint_analysis)

    records: list[dict] = []
    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    missing_mutation_records = 0

    for judge in joint_rows:
        task_id = judge.get("task_id")
        mutation_idx = judge.get("mutation_idx")
        row = mutation = None
        if isinstance(task_id, str) and isinstance(mutation_idx, int):
            indexed = mutation_index.get((task_id, mutation_idx))
            if indexed:
                row, mutation = indexed
        if row is None or mutation is None:
            missing_mutation_records += 1
            mutation_change: dict = {}
            mutation_meta: dict = {
                "clause_idx": judge.get("clause_idx"),
                "operator": judge.get("operator"),
                "clause_type": judge.get("clause_type"),
                "slot_type": None,
                "slot_original": None,
                "slot_target": None,
            }
        else:
            mutation_change = _mutation_change_summary(row, mutation)
            mutation_meta = {
                "clause_idx": mutation.get("clause_idx"),
                "operator": mutation.get("operator"),
                "clause_type": mutation_change.get("clause_type"),
                "slot_type": mutation.get("slot_type"),
                "slot_original": mutation.get("slot_original"),
                "slot_target": mutation.get("slot_target"),
            }

        joint_label = _normalise_joint_label(judge)
        route = _route_for_record(joint_label, mutation_change, mutation_meta)
        behavior = (
            behavior_index.get((task_id, mutation_idx), {})
            if isinstance(task_id, str) and isinstance(mutation_idx, int)
            else {}
        )
        action_kind = (
            judge.get("action")
            if judge.get("action") and joint_label != "JUDGE_UNKNOWN"
            else _action_from_joint_label(joint_label) or route.action_kind
        )
        if action_kind != route.action_kind and joint_label in _ROUTES:
            action_kind = route.action_kind

        record = {
            "task_id": task_id,
            "mutation_idx": mutation_idx,
            "method": _METHOD,
            **mutation_meta,
            "equivalent": _coerce_bool(judge.get("equivalent")),
            "mutation_reflected": _coerce_bool(judge.get("mutation_reflected")),
            "joint_label": joint_label,
            "action_category": route.action_category,
            "action_category_label": route.action_category_label,
            "action": action_kind,
            "action_kind": action_kind,
            "developer_insight": route.developer_insight,
            "priority": route.priority,
            "evidence_policy": _evidence_policy(route),
            "judge_prediction": judge.get("prediction"),
            "judge_reason": judge.get("reason"),
            "judge_confidence": judge.get("confidence"),
            "judge_model": judge.get("model"),
            "judge_method": judge.get("method"),
            "behavior_label": behavior.get("label"),
            "behavior_probe_engine": behavior.get("probe_engine"),
            "mutation_change": mutation_change,
            "action_payload": _action_payload(
                route=route,
                mutation_change=mutation_change,
                behavior=behavior,
            ),
        }
        records.append(record)
        label_counts[joint_label] += 1
        category_counts[route.action_category] += 1
        action_counts[action_kind] += 1
        priority_counts[route.priority] += 1

    summary = {
        "batch": str(batch_dir),
        "method": _METHOD,
        "joint_analysis": str(joint_analysis),
        "behavior_analysis": str(behavior_analysis) if behavior_analysis else None,
        "total_joint_records": len(joint_rows),
        "total_action_records": len(records),
        "missing_mutation_records": missing_mutation_records,
        "joint_label_counts": dict(label_counts),
        "action_category_counts": dict(category_counts),
        "action_counts": dict(action_counts),
        "priority_counts": dict(priority_counts),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Route joint equivalence/reflection judgments into developer-facing "
            "actions for survived NL-spec mutants."
        ),
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--joint-analysis",
        type=Path,
        default=None,
        help=(
            "JSONL produced by specmut.analysis.analyze_equivalence_emd_prompt. "
            "Defaults to survivor_equivalence_reflection_joint.jsonl under the "
            "batch."
        ),
    )
    parser.add_argument(
        "--behavior-analysis",
        type=Path,
        default=None,
        help=(
            "Optional survivor_behavior_analysis.jsonl sidecar. If omitted, "
            "the script uses the batch-local file when it exists."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not (batch_dir / "results.jsonl").exists():
        raise SystemExit(f"no results.jsonl found under {batch_dir}")

    joint_analysis = args.joint_analysis or (
        batch_dir / "survivor_equivalence_reflection_joint.jsonl"
    )
    joint_analysis = joint_analysis.resolve()
    if not joint_analysis.exists():
        raise SystemExit(f"no joint analysis JSONL found at {joint_analysis}")

    behavior_analysis = args.behavior_analysis
    if behavior_analysis is None:
        candidate = batch_dir / "survivor_behavior_analysis.jsonl"
        behavior_analysis = candidate if candidate.exists() else None
    if behavior_analysis is not None:
        behavior_analysis = behavior_analysis.resolve()

    output = args.output or (batch_dir / _DEFAULT_OUTPUT)
    records, summary = build_action_records(
        batch_dir,
        joint_analysis=joint_analysis,
        behavior_analysis=behavior_analysis,
    )
    _write_jsonl(output, records)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
