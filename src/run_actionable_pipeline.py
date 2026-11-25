"""Run the full post-hoc actionable survivor pipeline.

This is the single entry point for the post-hoc pipeline:

1. judge survived generated-code mutants with the joint LLM prompt;
2. route those judgments into developer-facing actions.

The script writes only sidecar files. It does not modify canonical
``results.jsonl`` or mutation-testing scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specmut.analysis.analyze_actionable_pipeline import build_action_records
from specmut.analysis.analyze_actionable_pipeline import _write_jsonl as _write_action_jsonl
from specmut.analysis.analyze_equivalence_emd_prompt import (
    _DEFAULT_CODE_CHARS,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MODEL,
    _METHOD as _JOINT_METHOD,
    _load_jsonl,
    _selection_stats,
    _write_jsonl as _write_joint_jsonl,
    analyze_batch,
)
from specmut.config import Config


_ACTION_METHOD = "survivor_actionable_pipeline_v2"
_DEFAULT_JOINT_NAME = "survivor_equivalence_reflection_joint.jsonl"
_DEFAULT_ACTION_NAME = "survivor_actionable_pipeline.jsonl"


def _range_suffix(start: int, limit: int | None) -> str:
    if start <= 0 and limit is None:
        return ""
    if limit is None:
        return f"_start{start}"
    return f"_start{start}_limit{limit}"


def default_output_paths(
    batch_dir: Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> tuple[Path, Path]:
    suffix = _range_suffix(start, limit)
    if suffix:
        return (
            batch_dir / f"survivor_equivalence_reflection_joint{suffix}.jsonl",
            batch_dir / f"survivor_actionable_pipeline{suffix}.jsonl",
        )
    return (
        batch_dir / _DEFAULT_JOINT_NAME,
        batch_dir / _DEFAULT_ACTION_NAME,
    )


def _summary_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_summary.json")


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def run_pipeline(
    batch_dir: Path,
    *,
    model: str,
    prompt_context: str,
    max_tokens: int,
    code_chars: int,
    start: int = 0,
    limit: int | None = None,
    timeout: float | None = None,
    attempts: int | None = None,
    behavior_analysis: Path | None = None,
    joint_output: Path | None = None,
    action_output: Path | None = None,
    reuse_joint_analysis: Path | None = None,
    stream_output: bool = False,
    progress_every: int = 0,
    dry_run: bool = False,
) -> dict:
    batch_dir = batch_dir.resolve()
    default_joint, default_action = default_output_paths(
        batch_dir,
        start=start,
        limit=limit,
    )
    joint_output = (joint_output or default_joint).resolve()
    action_output = (action_output or default_action).resolve()
    behavior_analysis = (
        behavior_analysis.resolve()
        if behavior_analysis is not None
        else (
            (batch_dir / "survivor_behavior_analysis.jsonl").resolve()
            if (batch_dir / "survivor_behavior_analysis.jsonl").exists()
            else None
        )
    )
    reuse_joint_analysis = (
        reuse_joint_analysis.resolve()
        if reuse_joint_analysis is not None
        else None
    )

    selection = _selection_stats(batch_dir, start=start, limit=limit)
    max_attempts = max(
        1,
        int(attempts if attempts else Config().llm_max_attempts),
    )
    dry_summary = {
        "batch": str(batch_dir),
        "joint_output": str(joint_output),
        "action_output": str(action_output),
        "reuse_joint_analysis": (
            str(reuse_joint_analysis) if reuse_joint_analysis else None
        ),
        "behavior_analysis": str(behavior_analysis) if behavior_analysis else None,
        "selected_survivors": selection["selected_survivors"],
        "start": start,
        "limit": limit,
        "joint_method": _JOINT_METHOD,
        "action_method": _ACTION_METHOD,
        "model": model,
        "prompt_context": prompt_context,
        "logical_llm_calls": (
            0 if reuse_joint_analysis else selection["selected_survivors"]
        ),
        "llm_calls_per_mutant": 1,
        "max_llm_attempts_per_mutant": max_attempts,
        "max_external_llm_api_calls": (
            0
            if reuse_joint_analysis
            else selection["selected_survivors"] * max_attempts
        ),
    }
    if dry_run:
        dry_summary["dry_run"] = True
        return dry_summary

    if reuse_joint_analysis is None:
        joint_records, joint_summary = analyze_batch(
            batch_dir,
            model=model,
            prompt_context=prompt_context,
            max_tokens=max_tokens,
            code_chars=code_chars,
            limit=limit,
            start=start,
            timeout=timeout,
            attempts=attempts,
            behavior_analysis=behavior_analysis,
            stream_output_path=joint_output if stream_output else None,
            progress_every=progress_every,
        )
        _write_joint_jsonl(joint_output, joint_records)
        _write_summary(_summary_path(joint_output), joint_summary)
        joint_analysis = joint_output
    else:
        if not reuse_joint_analysis.exists():
            raise FileNotFoundError(f"joint analysis not found: {reuse_joint_analysis}")
        reused_joint_records = len(_load_jsonl(reuse_joint_analysis))
        dry_summary["selected_survivors"] = reused_joint_records
        dry_summary["reused_joint_records"] = reused_joint_records
        joint_summary = {"total_emitted": reused_joint_records}
        joint_analysis = reuse_joint_analysis

    action_records, action_summary = build_action_records(
        batch_dir,
        joint_analysis=joint_analysis,
        behavior_analysis=behavior_analysis,
    )
    _write_action_jsonl(action_output, action_records)
    _write_summary(_summary_path(action_output), action_summary)

    return {
        **dry_summary,
        "dry_run": False,
        "joint_summary": joint_summary,
        "action_summary": action_summary,
        "joint_summary_path": str(_summary_path(joint_output)),
        "action_summary_path": str(_summary_path(action_output)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run joint LLM judging and actionable survivor routing as one "
            "post-hoc pipeline."
        ),
    )
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--emd-model",
        "--model",
        dest="model",
        default=_DEFAULT_MODEL,
        help="LLM used only for the joint equivalence/reflection judge.",
    )
    parser.add_argument(
        "--prompt-context",
        choices=["code-only", "spec-aware"],
        default="spec-aware",
    )
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--code-chars", type=int, default=_DEFAULT_CODE_CHARS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-attempts", type=int, default=None)
    parser.add_argument("--behavior-analysis", type=Path, default=None)
    parser.add_argument("--joint-output", type=Path, default=None)
    parser.add_argument("--action-output", type=Path, default=None)
    parser.add_argument(
        "--reuse-joint-analysis",
        type=Path,
        default=None,
        help=(
            "Skip LLM judging and build actionable output from this existing "
            "joint JSONL."
        ),
    )
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show output paths and expected LLM calls without writing files.",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not (batch_dir / "results.jsonl").exists():
        raise SystemExit(f"no results.jsonl found under {batch_dir}")

    summary = run_pipeline(
        batch_dir,
        model=args.model,
        prompt_context=args.prompt_context,
        max_tokens=args.max_tokens,
        code_chars=args.code_chars,
        start=args.start,
        limit=args.limit,
        timeout=args.llm_timeout,
        attempts=args.llm_attempts,
        behavior_analysis=args.behavior_analysis,
        joint_output=args.joint_output,
        action_output=args.action_output,
        reuse_joint_analysis=args.reuse_joint_analysis,
        stream_output=args.stream_output,
        progress_every=args.progress_every,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
