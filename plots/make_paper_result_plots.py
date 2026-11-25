#!/usr/bin/env python3
"""Generate the two task-level result figures used in the SpecMut paper.

The script uses the latest all-operator o4-mini runs under
../results/openai_o4-mini. It excludes the deprecated remove operator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "openai_o4-mini"

OPERATOR_RENAME = {
    "example": "incorrect example",
    "partial_edit": "requirement inconsistency",
    "inconsistent_example": "example inconsistency",
    "negate": "negate",
    "perturb": "perturb",
    "substitute": "substitute",
    "weaken": "weaken",
}

COLORS = {
    "LiveCodeBench": "#3B6EA8",
    "HumanEval+": "#C06C44",
}


def all_batch_dirs() -> list[Path]:
    dirs = []
    for path in sorted(RESULTS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if "allops_final" not in path.name:
            continue
        if (path / "results.jsonl").exists():
            dirs.append(path)
    return dirs


def benchmark_for_dir(path: Path) -> str:
    if path.name.startswith("livecodebench"):
        return "LiveCodeBench"
    if path.name.startswith("humaneval"):
        return "HumanEval+"
    raise ValueError(f"Unknown benchmark for {path}")


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def include_mutation(mutation: dict) -> bool:
    return mutation.get("operator") in OPERATOR_RENAME


def mutation_killed(mutation: dict) -> bool:
    return not bool(mutation.get("survived", False))


def collect_tasks() -> list[dict]:
    tasks = []

    for batch_dir in all_batch_dirs():
        benchmark = benchmark_for_dir(batch_dir)
        for record in read_jsonl(batch_dir / "results.jsonl"):
            task_id = record["task_id"]
            matrix = record.get("matrix") or []
            num_clauses = int(record.get("num_clauses") or len(record.get("clauses", [])) or len(matrix))
            num_tests = int(record.get("num_tests") or (len(matrix[0]) if matrix else 0))

            per_clause_eval = [0] * num_clauses
            per_clause_kill = [0] * num_clauses
            killed_test_cols = set()
            task_eval = 0
            task_killed = 0
            task_survived = 0

            for mutation in record.get("mutations", []):
                if not include_mutation(mutation):
                    continue

                clause_idx = int(mutation.get("clause_idx", -1))
                killed = mutation_killed(mutation)
                task_eval += 1
                if killed:
                    task_killed += 1
                else:
                    task_survived += 1

                if 0 <= clause_idx < num_clauses:
                    per_clause_eval[clause_idx] += 1
                    if killed:
                        per_clause_kill[clause_idx] += 1
                        test_results = mutation.get("test_results") or []
                        fail_cols = [idx for idx, passed in enumerate(test_results) if passed is False]
                        if not fail_cols and clause_idx < len(matrix):
                            fail_cols = [idx for idx, value in enumerate(matrix[clause_idx]) if value]
                        killed_test_cols.update(fail_cols)

            covered = sum(1 for count in per_clause_kill if count > 0)
            evaluable = sum(1 for count in per_clause_eval if count > 0)
            survived_only = sum(
                1 for eval_count, kill_count in zip(per_clause_eval, per_clause_kill) if eval_count > 0 and kill_count == 0
            )
            no_evaluable = max(0, num_clauses - evaluable)
            tasks.append(
                {
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "num_clauses": num_clauses,
                    "num_tests": num_tests,
                    "evaluable_mutants": task_eval,
                    "killed_mutants": task_killed,
                    "survived_mutants": task_survived,
                    "covered_clauses": covered,
                    "evaluable_clauses": evaluable,
                    "survived_only_clauses": survived_only,
                    "no_evaluable_clauses": no_evaluable,
                    "kill_ratio": task_killed / task_eval if task_eval else 0.0,
                    "sc": covered / num_clauses if num_clauses else 0.0,
                    "sc_evaluable": covered / evaluable if evaluable else 0.0,
                    "gr": len(killed_test_cols) / num_tests if num_tests else 0.0,
                }
            )

    return tasks


def save_csv(name: str, rows: list[dict]):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
        }
    )


def savefig(fig, stem: str):
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", transparent=True)
    plt.close(fig)


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if not var_x or not var_y:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / (var_x * var_y) ** 0.5


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if not var_x:
        return 0.0, mean_y
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def plot_sc_gr(tasks: list[dict]):
    rows = [
        {
            "benchmark": t["benchmark"],
            "task_id": t["task_id"],
            "sc": t["sc"],
            "gr": t["gr"],
            "evaluable_mutants": t["evaluable_mutants"],
        }
        for t in tasks
    ]
    save_csv("fig_sc_gr_by_task.csv", rows)

    fig, ax = plt.subplots(figsize=(4.2, 3.45))
    for benchmark in ["LiveCodeBench", "HumanEval+"]:
        bench_rows = [t for t in tasks if t["benchmark"] == benchmark]
        ax.scatter(
            [r["sc"] for r in bench_rows],
            [r["gr"] for r in bench_rows],
            s=28,
            color=COLORS[benchmark],
            alpha=0.62,
            edgecolor="white",
            linewidth=0.4,
            label=benchmark,
        )

    ax.set_xlabel("SC")
    ax.set_ylabel("GR")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")
    savefig(fig, "fig_sc_gr_by_task")


def plot_kill_ratio_sc(tasks: list[dict]):
    rows = [
        {
            "benchmark": t["benchmark"],
            "task_id": t["task_id"],
            "kill_ratio": t["kill_ratio"],
            "sc": t["sc"],
            "num_clauses": t["num_clauses"],
        }
        for t in tasks
    ]
    save_csv("fig_kill_ratio_sc_by_task.csv", rows)

    fig, ax = plt.subplots(figsize=(4.55, 3.35))
    for benchmark in ["LiveCodeBench", "HumanEval+"]:
        bench_rows = [t for t in tasks if t["benchmark"] == benchmark]
        ax.scatter(
            [r["kill_ratio"] for r in bench_rows],
            [r["sc"] for r in bench_rows],
            s=28,
            color=COLORS[benchmark],
            alpha=0.62,
            edgecolor="white",
            linewidth=0.45,
            label=benchmark,
        )

    for benchmark in ["LiveCodeBench", "HumanEval+"]:
        bench_rows = [t for t in tasks if t["benchmark"] == benchmark]
        xs = [t["kill_ratio"] for t in bench_rows]
        ys = [t["sc"] for t in bench_rows]
        r_value = pearson(xs, ys)
        slope, intercept = linear_fit(xs, ys)
        x_min, x_max = min(xs), max(xs)
        ax.plot(
            [x_min, x_max],
            [intercept + slope * x_min, intercept + slope * x_max],
            color=COLORS[benchmark],
            linestyle=":",
            linewidth=1.25,
            alpha=0.95,
            zorder=1,
        )
        label_x = 0.72 if benchmark == "LiveCodeBench" else 0.82
        label_y = intercept + slope * label_x
        ax.text(
            label_x,
            label_y,
            f"r = {r_value:.3f}",
            fontsize=8,
            color=COLORS[benchmark],
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.3},
        )

    ax.set_xlabel("Percentage killed")
    ax.set_ylabel("SC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#DDDDDD")
    savefig(fig, "fig_kill_ratio_sc_by_task")


def main():
    set_style()
    tasks = collect_tasks()
    plot_sc_gr(tasks)
    plot_kill_ratio_sc(tasks)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "outputs": [
                    "fig_sc_gr_by_task.pdf",
                    "fig_sc_gr_by_task.png",
                    "fig_kill_ratio_sc_by_task.pdf",
                    "fig_kill_ratio_sc_by_task.png",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
