"""LLM-based joint survivor judgment for generated-code mutant pairs.

This is a post-hoc analysis script for survived NL-spec mutants. It treats
each survived mutant as a pair of generated Python programs and asks an LLM
two questions plus a confidence estimate in one JSON response:

1. Are the generated programs equivalent on inputs valid under the original
   specification?
2. Does the mutant program appear to reflect the intended spec mutation?

The output is intentionally separate from survivor behavior analysis and from
the canonical mutation-testing verdicts.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from specmut.config import Config
from specmut.llm import LLMClient


_METHOD = "emd_joint_equivalence_reflection_prompt_v2"
_DEFAULT_MODEL = "openai/gpt-5.4"
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_CODE_CHARS = 12000
_DEFAULT_SPEC_CHARS = 5000
_DEFAULT_DIFF_CHARS = 3000


@dataclass
class EMDPrediction:
    prediction: str
    equivalent: bool | None
    raw_response: str
    reason: str
    confidence: str | None = None
    mutation_reflected: bool | None = None
    joint_label: str | None = None
    action: str | None = None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_survived(mutation: dict) -> bool:
    if mutation.get("survived") is True:
        return True
    if mutation.get("status") == "survived":
        return True
    if mutation.get("equivalent") is True:
        return True
    return False


def _is_evaluable_survivor(mutation: dict) -> bool:
    if not _read_survived(mutation):
        return False
    if mutation.get("noop"):
        return False
    return True


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, (limit - 24) // 2)
    return text[:half] + "\n...<truncated>...\n" + text[-half:]


def _mutation_context(row: dict, mutation: dict) -> dict:
    clauses = row.get("clauses") or []
    quotes = row.get("quotes") or []
    clause_idx = mutation.get("clause_idx")
    clause = (
        clauses[clause_idx]
        if isinstance(clause_idx, int) and 0 <= clause_idx < len(clauses)
        else ""
    )
    quote = (
        quotes[clause_idx]
        if isinstance(clause_idx, int) and 0 <= clause_idx < len(quotes)
        else ""
    )
    clause_types = row.get("clause_types") or []
    clause_type = (
        clause_types[clause_idx]
        if isinstance(clause_idx, int) and 0 <= clause_idx < len(clause_types)
        else None
    )
    return {
        "task_id": row.get("task_id"),
        "clause_idx": clause_idx,
        "operator": mutation.get("operator"),
        "clause_type": clause_type,
        "slot_type": mutation.get("slot_type"),
        "slot_original": mutation.get("slot_original"),
        "slot_target": mutation.get("slot_target"),
        "original_clause": clause or quote,
        "anchored_text": quote,
    }


def _compact_spec_diff(
    original_spec: str,
    mutated_spec: str,
    *,
    max_chars: int = _DEFAULT_DIFF_CHARS,
) -> str:
    if not mutated_spec:
        return ""
    if original_spec == mutated_spec:
        return "No textual diff found between original_spec and mutated_spec."

    diff_lines = list(
        difflib.unified_diff(
            original_spec.splitlines(),
            mutated_spec.splitlines(),
            fromfile="original_spec",
            tofile="mutated_spec",
            lineterm="",
            n=1,
        )
    )
    diff_text = "\n".join(diff_lines)
    return _truncate_middle(diff_text, max_chars)


def _mutation_change_summary(row: dict, mutation: dict) -> dict:
    context = _mutation_context(row, mutation)
    slot_original = context.get("slot_original")
    slot_target = context.get("slot_target")
    if slot_original is not None or slot_target is not None:
        change = f"{slot_original!r} -> {slot_target!r}"
    else:
        change = "see spec_diff"

    return {
        "task_id": context.get("task_id"),
        "clause_idx": context.get("clause_idx"),
        "operator": context.get("operator"),
        "clause_type": context.get("clause_type"),
        "slot_type": context.get("slot_type"),
        "intended_change": change,
        "original_clause": context.get("original_clause"),
        "anchored_text": context.get("anchored_text"),
        "spec_diff": _compact_spec_diff(
            row.get("spec", ""),
            mutation.get("mutated_spec", ""),
        ),
    }


def _build_joint_messages(
    *,
    row: dict,
    mutation: dict,
    prompt_context: str,
    code_chars: int,
) -> list[dict]:
    reference_code = _truncate_middle(row.get("reference_program", ""), code_chars)
    mutant_code = _truncate_middle(mutation.get("mutant_code", ""), code_chars)

    instruction = (
        "You are judging survived generated-code mutants for a natural "
        "language specification mutation experiment. Return only a compact "
        "JSON object. Do not include markdown."
    )
    if prompt_context == "spec-aware":
        payload = {
            "task_id": row.get("task_id"),
            "original_specification": _truncate_middle(
                row.get("spec", ""),
                _DEFAULT_SPEC_CHARS,
            ),
            "mutation_change": _mutation_change_summary(row, mutation),
        }
        context = (
            "Original task and intended spec change:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        )
    else:
        payload = {
            "task_id": row.get("task_id"),
            "mutation_change": _mutation_change_summary(row, mutation),
        }
        context = (
            "Intended spec change:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        )

    content = (
        "Decide the equivalence/reflection judgment about Code 1 and Code 2.\n\n"
        "1. equivalent: whether Code 1 and Code 2 produce the same observable "
        "output for every input that is valid under the original specification. "
        "Do not use inputs that are valid only under the mutated specification "
        "as evidence of non-equivalence. Harmless trailing newline or trailing "
        "whitespace differences should not count as non-equivalence when the "
        "printed value is the same.\n"
        "2. mutation_reflected: whether Code 2's behavior appears to implement "
        "the intended spec change shown in the diff, relative to Code 1 and "
        "the original specification. Answer false when Code 2 ignores the "
        "change, only refactors formatting, or differs for an unrelated reason. "
        "mutation_reflected can be true even when equivalent is true, for example "
        "when the mutation changes an input bound or domain outside the original "
        "valid inputs.\n\n"
        "Also set confidence to high, medium, or low. Use high only when the "
        "code and diff make the judgment clear; use low when the judgment "
        "depends on uncertain interpretation or incomplete evidence.\n\n"
        "Return exactly this JSON shape with booleans, confidence, and a short "
        "reason:\n"
        '{"equivalent": true, "mutation_reflected": false, '
        '"confidence": "high", "reason": "..."}'
        "\n\n"
        f"{context}"
        "Input:\n"
        "'''Code 1\n"
        f"{reference_code}\n"
        "'''\n"
        "'''Code 2\n"
        f"{mutant_code}\n"
        "'''"
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]


def _json_loads_loose(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty JSON response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object found")


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


def _coerce_confidence(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    confidence = value.strip().lower()
    if confidence in {"high", "medium", "low"}:
        return confidence
    return None


def _joint_label_from_bools(
    equivalent: bool | None,
    mutation_reflected: bool | None,
) -> str | None:
    if equivalent is None or mutation_reflected is None:
        return None
    if equivalent and mutation_reflected:
        return "EQUIV_REFLECTED"
    if equivalent and not mutation_reflected:
        return "EQUIV_NOT_REFLECTED"
    if not equivalent and mutation_reflected:
        return "NON_EQUIV_REFLECTED"
    return "NON_EQUIV_NOT_REFLECTED"


def _action_from_joint_label(joint_label: str | None) -> str | None:
    return {
        "EQUIV_REFLECTED": "downrank_equivalent_mutant",
        "EQUIV_NOT_REFLECTED": "strengthen_mutated_clause_and_regenerate",
        "NON_EQUIV_REFLECTED": "generate_mutation_guided_tests",
        "NON_EQUIV_NOT_REFLECTED": "strengthen_mutated_clause_and_regenerate",
    }.get(joint_label)


def _parse_joint_response(response: str) -> EMDPrediction:
    try:
        payload = _json_loads_loose(response)
    except (ValueError, json.JSONDecodeError) as exc:
        return EMDPrediction(
            prediction="llm_joint_unknown",
            equivalent=None,
            raw_response=response,
            reason=f"LLM response was not parseable JSON: {type(exc).__name__}",
        )
    if not isinstance(payload, dict):
        return EMDPrediction(
            prediction="llm_joint_unknown",
            equivalent=None,
            raw_response=response,
            reason="LLM response JSON was not an object",
        )

    equivalent = _coerce_bool(payload.get("equivalent"))
    mutation_reflected = _coerce_bool(
        payload.get("mutation_reflected")
        if "mutation_reflected" in payload
        else payload.get("faithful")
    )
    confidence = _coerce_confidence(payload.get("confidence"))
    joint_label = _joint_label_from_bools(equivalent, mutation_reflected)
    if joint_label is None:
        return EMDPrediction(
            prediction="llm_joint_unknown",
            equivalent=equivalent,
            mutation_reflected=mutation_reflected,
            raw_response=response,
            confidence=confidence,
            reason=(
                "LLM response did not contain unambiguous equivalent and "
                "mutation_reflected booleans"
            ),
        )

    return EMDPrediction(
        prediction=f"llm_{joint_label.lower()}",
        equivalent=equivalent,
        mutation_reflected=mutation_reflected,
        joint_label=joint_label,
        action=_action_from_joint_label(joint_label),
        raw_response=response,
        confidence=confidence,
        reason=str(payload.get("reason") or ""),
    )


def _load_behavior_index(path: Path | None) -> dict[tuple[str, int], dict]:
    if path is None:
        return {}
    index: dict[tuple[str, int], dict] = {}
    for row in _load_jsonl(path):
        task_id = row.get("task_id")
        mutation_idx = row.get("mutation_idx")
        if isinstance(task_id, str) and isinstance(mutation_idx, int):
            index[(task_id, mutation_idx)] = row
    return index


def _selection_stats(
    batch_dir: Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> dict:
    rows = _load_jsonl(batch_dir / "results.jsonl")
    total_evaluable = 0
    selected = 0
    skipped = 0
    for row in rows:
        for mutation in row.get("mutations") or []:
            if not _is_evaluable_survivor(mutation):
                skipped += 1
                continue
            if total_evaluable >= start and (limit is None or selected < limit):
                selected += 1
            total_evaluable += 1
    return {
        "batch": str(batch_dir),
        "total_results": len(rows),
        "total_evaluable_survivors": total_evaluable,
        "selected_survivors": selected,
        "start": start,
        "limit": limit,
        "total_skipped_mutations": skipped,
    }


def analyze_batch(
    batch_dir: Path,
    *,
    model: str,
    prompt_context: str,
    max_tokens: int,
    code_chars: int,
    limit: int | None = None,
    start: int = 0,
    timeout: float | None = None,
    attempts: int | None = None,
    behavior_analysis: Path | None = None,
    stream_output_path: Path | None = None,
    progress_every: int = 0,
) -> tuple[list[dict], dict]:
    config = Config()
    if timeout is not None:
        config.llm_timeout_seconds = timeout
    if attempts is not None:
        config.llm_max_attempts = max(1, int(attempts))
    max_attempts_per_mutant = max(1, int(config.llm_max_attempts))
    client = LLMClient(config)

    rows = _load_jsonl(batch_dir / "results.jsonl")
    behavior_index = _load_behavior_index(behavior_analysis)
    records: list[dict] = []
    counts: Counter[str] = Counter()
    inspected = 0
    skipped = 0
    emitted = 0

    stream_handle = None
    if stream_output_path is not None:
        stream_output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_handle = stream_output_path.open("w")

    try:
        for row in rows:
            task_id = row.get("task_id")
            for mutation_idx, mutation in enumerate(row.get("mutations") or []):
                if not _is_evaluable_survivor(mutation):
                    skipped += 1
                    continue
                if inspected < start:
                    inspected += 1
                    continue
                if limit is not None and emitted >= limit:
                    break
                inspected += 1

                messages = _build_joint_messages(
                    row=row,
                    mutation=mutation,
                    prompt_context=prompt_context,
                    code_chars=code_chars,
                )
                try:
                    response = client.call(
                        messages,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=0,
                        cache_salt=(
                            f"emd:{_METHOD}:"
                            f"{prompt_context}:{task_id}:"
                            f"{mutation_idx}:{model}"
                        ),
                    )
                    pred = _parse_joint_response(response)
                except Exception as exc:
                    pred = EMDPrediction(
                        prediction="llm_joint_failed",
                        equivalent=None,
                        raw_response="",
                        reason=f"LLM joint judge call failed: {type(exc).__name__}",
                    )

                behavior = behavior_index.get((str(task_id), mutation_idx), {})
                record = {
                    "task_id": task_id,
                    "mutation_idx": mutation_idx,
                    "clause_idx": mutation.get("clause_idx"),
                    "operator": mutation.get("operator"),
                    "method": _METHOD,
                    "model": model,
                    "prompt_strategy": "joint-equivalence-reflection-prompt",
                    "prompt_context": prompt_context,
                    "behavior_label": behavior.get("label"),
                    "behavior_probe_engine": behavior.get("probe_engine"),
                    **asdict(pred),
                }
                records.append(record)
                counts[pred.prediction] += 1
                emitted += 1

                if stream_handle is not None:
                    stream_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream_handle.flush()
                if progress_every and emitted % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "emitted": emitted,
                                "inspected": inspected,
                                "skipped": skipped,
                                "predictions": dict(counts),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            if limit is not None and emitted >= limit:
                break
    finally:
        if stream_handle is not None:
            stream_handle.close()

    summary = {
        "batch": str(batch_dir),
        "method": _METHOD,
        "model": model,
        "prompt_strategy": "joint-equivalence-reflection-prompt",
        "prompt_context": prompt_context,
        "total_results": len(rows),
        "total_emitted": emitted,
        "total_inspected_survivors": inspected,
        "total_skipped_mutations": skipped,
        "prediction_counts": dict(counts),
        "limit": limit,
        "start": start,
        "max_tokens": max_tokens,
        "code_chars": code_chars,
        "llm_timeout": timeout,
        "llm_attempts": attempts,
        "logical_llm_calls": emitted,
        "llm_calls_per_mutant": 1,
        "max_llm_attempts_per_mutant": max_attempts_per_mutant,
        "max_external_llm_api_calls": emitted * max_attempts_per_mutant,
        "behavior_analysis": str(behavior_analysis) if behavior_analysis else None,
    }
    return records, summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run joint LLM equivalence/reflection judging on survived "
            "generated-code mutant pairs."
        ),
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--emd-model",
        "--model",
        dest="model",
        default=_DEFAULT_MODEL,
        help=(
            "LLM used only for joint equivalence/reflection judging. This is "
            "separate from generator/tooling models used by experiment runs."
        ),
    )
    parser.add_argument(
        "--prompt-context",
        choices=["code-only", "spec-aware"],
        default="spec-aware",
        help=(
            "code-only adds only mutation context; spec-aware also adds the "
            "original NL spec."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--code-chars", type=int, default=_DEFAULT_CODE_CHARS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-attempts", type=int, default=None)
    parser.add_argument("--behavior-analysis", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report selected survivor and expected LLM call counts.",
    )
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not (batch_dir / "results.jsonl").exists():
        raise SystemExit(f"no results.jsonl found under {batch_dir}")

    if args.dry_run:
        stats = _selection_stats(
            batch_dir,
            start=args.start,
            limit=args.limit,
        )
        max_attempts = max(
            1,
            int(
                args.llm_attempts
                if args.llm_attempts
                else Config().llm_max_attempts
            ),
        )
        stats.update(
            {
                "method": _METHOD,
                "model": args.model,
                "prompt_strategy": "joint-equivalence-reflection-prompt",
                "prompt_context": args.prompt_context,
                "logical_llm_calls": stats["selected_survivors"],
                "llm_calls_per_mutant": 1,
                "max_llm_attempts_per_mutant": max_attempts,
                "max_external_llm_api_calls": (
                    stats["selected_survivors"] * max_attempts
                ),
                "note": (
                    "Logical calls count one LLM classification per selected "
                    "survivor. External API calls can be lower with cache hits "
                    "or higher up to max attempts when retryable errors occur."
                ),
            }
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    output = args.output or (batch_dir / "survivor_equivalence_reflection_joint.jsonl")
    records, summary = analyze_batch(
        batch_dir,
        model=args.model,
        prompt_context=args.prompt_context,
        max_tokens=args.max_tokens,
        code_chars=args.code_chars,
        limit=args.limit,
        start=args.start,
        timeout=args.llm_timeout,
        attempts=args.llm_attempts,
        behavior_analysis=args.behavior_analysis,
        stream_output_path=output if args.stream_output else None,
        progress_every=args.progress_every,
    )
    _write_jsonl(output, records)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
