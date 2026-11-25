"""Load the HumanEval+ and LiveCodeBench benchmarks used in the paper."""

import ast as _ast
import logging
import re
from dataclasses import dataclass, field

from specmut.execute import _run_batch

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    task_id: str
    spec: str
    entry_point: str
    prompt: str
    test_inputs: list
    expected_outputs: list
    canonical_solution: str
    # Structured input-output examples extracted from the specification
    # at load time. Each pair is a (args, expected) tuple where ``args``
    # is a list of positional arguments.
    example_pairs: list = field(default_factory=list)
    # ``function`` is used for HumanEval+ and for the wrapped
    # LiveCodeBench solve(stdin) interface.
    mode: str = "function"


_TYPE_WRAPPERS = {
    "set": set,
    "frozenset": frozenset,
    "list": list,
    "tuple": tuple,
    "dict": dict,
}


def _safe_eval(node):
    """Like ``ast.literal_eval`` but accepts common literal wrappers."""
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
        fname = node.func.id
        if fname in _TYPE_WRAPPERS:
            wrapper = _TYPE_WRAPPERS[fname]
            if not node.args:
                return wrapper()
            if len(node.args) == 1:
                inner = _safe_eval(node.args[0])
                return wrapper(inner)
    return _ast.literal_eval(node)


def _contains_ellipsis(value) -> bool:
    """Recursively check whether a value contains an ellipsis literal."""
    if value is Ellipsis:
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_ellipsis(v) for v in value)
    if isinstance(value, dict):
        return any(
            _contains_ellipsis(k) or _contains_ellipsis(v)
            for k, v in value.items()
        )
    return False


def _resolve_call_args(args_str: str, var_map: dict) -> list | None:
    """Parse function-call arguments, resolving literal variables."""
    try:
        tree = _ast.parse(f"f({args_str})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(tree, _ast.Call) or tree.keywords:
        return None

    args = []
    for arg in tree.args:
        try:
            args.append(_safe_eval(arg))
            continue
        except (ValueError, TypeError, SyntaxError):
            pass
        if isinstance(arg, _ast.Name) and arg.id in var_map:
            args.append(var_map[arg.id])
        else:
            return None
    return args


def _parse_doctest_pairs(spec: str, entry_point: str) -> list:
    """Parse doctest-style examples into ``(args, expected)`` pairs."""
    pairs = []
    lines = spec.splitlines()
    var_map: dict = {}

    re_var_assign = re.compile(r"^>>>\s*(\w+)\s*=\s*(.+)$")
    re_call_assign = re.compile(
        rf"^>>>\s*(\w+)\s*=\s*{re.escape(entry_point)}\s*\((.*)\)\s*$"
    )
    re_direct_call = re.compile(
        rf"^>>>\s*{re.escape(entry_point)}\s*\((.*)\)\s*$"
    )

    def _next_non_empty(idx: int) -> int:
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    def _try_literal(text: str):
        try:
            value = _safe_eval(_ast.parse(text, mode="eval").body)
            if _contains_ellipsis(value):
                return None
            return value
        except (SyntaxError, ValueError, TypeError):
            return None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m_call_assign = re_call_assign.match(line)
        if m_call_assign:
            result_var = m_call_assign.group(1)
            args = _resolve_call_args(m_call_assign.group(2), var_map)
            if args is None:
                i += 1
                continue

            j = i + 1
            re_print = re.compile(
                rf"^>>>\s*print\s*\(\s*{re.escape(result_var)}\s*\)\s*$"
            )
            re_bare = re.compile(rf"^>>>\s*{re.escape(result_var)}\s*$")
            found_print = False
            while j < len(lines):
                ln = lines[j].strip()
                if not ln:
                    j += 1
                    continue
                if re_print.match(ln) or re_bare.match(ln):
                    j += 1
                    found_print = True
                    break
                if ln.startswith(">>>"):
                    break
                j += 1
            if found_print:
                k = _next_non_empty(j)
                if k < len(lines) and not lines[k].strip().startswith(">>>"):
                    expected = _try_literal(lines[k].strip())
                    if expected is not None and not _contains_ellipsis(args):
                        pairs.append((args, expected))
            i += 1
            continue

        m_var = re_var_assign.match(line)
        if m_var and entry_point not in m_var.group(2):
            try:
                value = _safe_eval(_ast.parse(m_var.group(2).strip(), mode="eval").body)
                var_map[m_var.group(1)] = value
            except (SyntaxError, ValueError, TypeError):
                pass
            i += 1
            continue

        m_direct = re_direct_call.match(line)
        if m_direct:
            args = _resolve_call_args(m_direct.group(1), var_map)
            if args is None:
                i += 1
                continue
            j = _next_non_empty(i + 1)
            if j < len(lines) and not lines[j].strip().startswith(">>>"):
                expected = _try_literal(lines[j].strip())
                if expected is not None and not _contains_ellipsis(args):
                    pairs.append((args, expected))
            i += 1
            continue

        i += 1

    return pairs


def _extract_docstring(prompt: str, entry_point: str | None = None) -> str:
    """Extract the entry-point docstring from a benchmark prompt."""
    if entry_point:
        pattern = rf'def\s+{re.escape(entry_point)}\s*\(.*?\).*?"""(.*?)"""'
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        pattern = rf"def\s+{re.escape(entry_point)}\s*\(.*?\).*?'''(.*?)'''"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
    matches = list(re.finditer(r'"""(.*?)"""', prompt, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    matches = list(re.finditer(r"'''(.*?)'''", prompt, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return prompt.strip()


def _compute_expected_outputs(
    canonical_code: str, entry_point: str, test_inputs: list, timeout: int = 30
) -> list | None:
    """Run the canonical solution on all test inputs."""
    results = _run_batch(canonical_code, entry_point, test_inputs, timeout)
    outputs = []
    for status, output in results:
        if status != "ok":
            return None
        outputs.append(output)
    return outputs


def load_humaneval(timeout: int = 30) -> list[Problem]:
    """Load HumanEval+ through EvalPlus."""
    from evalplus.data import get_human_eval_plus

    data = get_human_eval_plus()
    problems = []
    for task_id, entry in data.items():
        spec = _extract_docstring(entry["prompt"], entry["entry_point"])
        test_inputs = list(entry.get("base_input", []))
        test_inputs += list(entry.get("plus_input", []))

        canonical_code = entry["prompt"] + entry["canonical_solution"]
        expected = _compute_expected_outputs(
            canonical_code, entry["entry_point"], test_inputs, timeout
        )
        if expected is None:
            logger.warning(f"Skipping {task_id}: canonical solution failed")
            continue

        problems.append(Problem(
            task_id=task_id,
            spec=spec,
            entry_point=entry["entry_point"],
            prompt=entry["prompt"],
            test_inputs=test_inputs,
            expected_outputs=expected,
            canonical_solution=canonical_code,
            example_pairs=_parse_doctest_pairs(spec, entry["entry_point"]),
        ))
    return problems


def load_livecodebench(
    cutoff_date: str = "2024-01-01",
    split_file: str = "test.jsonl",
    timeout: int = 30,
) -> list[Problem]:
    """Load the post-cutoff LiveCodeBench code-generation tasks."""
    import base64 as _b64
    import datetime as _dt
    import json as _json
    import pickle as _pickle
    import zlib as _zlib

    from huggingface_hub import hf_hub_download

    cutoff = _dt.datetime.fromisoformat(cutoff_date)
    path = hf_hub_download(
        "livecodebench/code_generation_lite",
        split_file,
        repo_type="dataset",
    )

    problems: list[Problem] = []
    skipped_functional = 0
    skipped_pre_cutoff = 0
    skipped_decode = 0

    with open(path) as f:
        for line in f:
            try:
                entry = _json.loads(line)
            except _json.JSONDecodeError:
                continue

            try:
                contest_date = _dt.datetime.fromisoformat(entry["contest_date"])
            except (KeyError, ValueError):
                continue
            if contest_date < cutoff:
                skipped_pre_cutoff += 1
                continue

            if entry.get("starter_code", "").strip():
                skipped_functional += 1
                continue

            question = entry.get("question_content", "")
            if not question.strip():
                continue

            try:
                public = _json.loads(entry.get("public_test_cases", "[]"))
            except _json.JSONDecodeError:
                public = []

            try:
                priv_blob = entry.get("private_test_cases", "")
                if priv_blob:
                    inner = _pickle.loads(_zlib.decompress(_b64.b64decode(priv_blob)))
                    private = _json.loads(inner) if isinstance(inner, str) else inner
                else:
                    private = []
            except Exception:
                private = []
                skipped_decode += 1

            all_cases = list(public) + list(private)
            stdin_cases = [c for c in all_cases if c.get("testtype") == "stdin"]
            if not stdin_cases:
                continue

            test_inputs = [[c["input"]] for c in stdin_cases]
            expected_outputs = [c["output"] for c in stdin_cases]
            example_pairs = [
                ([c["input"]], c["output"])
                for c in public
                if c.get("testtype") == "stdin"
            ][:5]

            entry_point = "solve"
            prompt = (
                f'def solve(stdin: str) -> str:\n    """\n'
                f'{question.strip()}\n    """\n'
            )

            problems.append(Problem(
                task_id=f"LCB/{entry.get('question_id', 'unknown')}",
                spec=question.strip(),
                entry_point=entry_point,
                prompt=prompt,
                test_inputs=test_inputs,
                expected_outputs=expected_outputs,
                canonical_solution="",
                example_pairs=example_pairs,
            ))

    logger.info(
        f"LiveCodeBench: loaded {len(problems)} stdin problems "
        f"(cutoff>={cutoff_date}, skipped {skipped_pre_cutoff} pre-cutoff, "
        f"{skipped_functional} functional, {skipped_decode} decode failures)"
    )
    return problems


def load_benchmark(name: str, timeout: int = 30) -> list[Problem]:
    """Load one of the two benchmarks used in the paper."""
    loaders = {
        "humaneval": load_humaneval,
        "livecodebench": load_livecodebench,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown benchmark: {name}. Available: {list(loaders.keys())}"
        )
    return loaders[name](timeout=timeout)
