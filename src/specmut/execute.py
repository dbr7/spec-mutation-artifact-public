"""Sandboxed code execution for testing generated programs."""

import subprocess
import sys
import tempfile
import textwrap
from typing import Any

_RUNNER_TEMPLATE = textwrap.dedent("""\
    import sys, os, pickle, base64

    # Suppress GUI popups (matplotlib, tkinter, etc.)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["DISPLAY"] = ""

    code = {code!r}
    entry_point = {entry_point!r}
    test_inputs = pickle.loads(base64.b64decode({test_inputs_b64!r}))

    namespace = {{}}
    exec(code, namespace)
    func = namespace[entry_point]

    def _make_picklable(obj):
        try:
            pickle.dumps(obj)
            return obj
        except Exception:
            # For unpicklable objects (e.g. re.Match), preserve truthiness and repr
            return bool(obj) if obj is not None else None

    results = []
    for test_input in test_inputs:
        try:
            if isinstance(test_input, (list, tuple)):
                result = func(*test_input)
            else:
                result = func(test_input)
            results.append(("ok", _make_picklable(result)))
        except Exception as e:
            results.append(("error", str(e)[:200]))

    sys.stdout.buffer.write(base64.b64encode(pickle.dumps(results)))
""")


def _run_batch(code: str, entry_point: str, test_inputs: list, timeout: int) -> list[tuple[str, Any]]:
    """Run code on all test inputs in a single subprocess."""
    import pickle
    import base64

    try:
        test_inputs_b64 = base64.b64encode(pickle.dumps(test_inputs)).decode("ascii")
    except Exception:
        return [("error", "cannot serialise inputs")] * len(test_inputs)

    runner = _RUNNER_TEMPLATE.format(
        code=code,
        entry_point=entry_point,
        test_inputs_b64=test_inputs_b64,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True, timeout=timeout,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                return [("error", result.stderr.decode("utf-8", errors="replace")[:200])] * len(test_inputs)
            outputs = pickle.loads(base64.b64decode(result.stdout.strip()))
            return outputs
    except subprocess.TimeoutExpired:
        return [("timeout", None)] * len(test_inputs)
    except Exception as e:
        return [("error", str(e)[:200])] * len(test_inputs)


def _run_stdin_batch(code: str, test_inputs: list, timeout: int) -> list[tuple[str, Any]]:
    """Run ``code`` as a stand-alone Python script for each test
    input, piping the input to stdin and capturing stdout.

    This matches the native LiveCodeBench evaluation
    protocol: the generated code is not wrapped in a function, it
    uses ``input()`` / ``sys.stdin`` and ``print()`` / ``sys.stdout``
    directly, and each test consists of an entire stdin payload.
    Each test input is run in its own subprocess so a slow or
    hanging test cannot stall the others.
    """
    results: list[tuple[str, Any]] = []
    for stdin_text in test_inputs:
        if isinstance(stdin_text, (list, tuple)):
            # Defensive: old loader variants wrapped each stdin in a
            # single-element list; the new loaders pass flat strings.
            stdin_text = stdin_text[0] if stdin_text else ""
        if not isinstance(stdin_text, str):
            stdin_text = str(stdin_text)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    input=stdin_text.encode("utf-8"),
                    capture_output=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )
                if proc.returncode != 0:
                    err = proc.stderr.decode("utf-8", errors="replace")[:200]
                    results.append(("error", err))
                    continue
                stdout_text = proc.stdout.decode("utf-8", errors="replace")
                results.append(("ok", stdout_text))
        except subprocess.TimeoutExpired:
            results.append(("timeout", None))
        except Exception as e:
            results.append(("error", str(e)[:200]))
    return results


def _values_equal(output, expected) -> bool:
    """Compare two values with tolerance for strings, floats, and non-comparable objects."""
    # String comparison: strip whitespace, then try numeric approximate match
    if isinstance(output, str) and isinstance(expected, str):
        a, b = output.strip(), expected.strip()
        if a == b:
            return True
        # Try line-by-line numeric comparison for multi-line outputs
        a_lines = a.splitlines()
        b_lines = b.splitlines()
        if len(a_lines) != len(b_lines):
            return False
        for al, bl in zip(a_lines, b_lines):
            al, bl = al.strip(), bl.strip()
            if al == bl:
                continue
            # Try numeric approximate comparison
            try:
                if abs(float(al) - float(bl)) < 1e-4:
                    continue
            except ValueError:
                pass
            return False
        return True

    # Direct equality
    try:
        if output == expected:
            return True
    except (ValueError, TypeError):
        pass

    # Float approximate comparison
    if isinstance(output, (int, float)) and isinstance(expected, (int, float)):
        try:
            if abs(float(output) - float(expected)) < 1e-4:
                return True
        except (ValueError, TypeError, OverflowError):
            pass

    # Fallback: compare repr for non-comparable objects (e.g. matplotlib Axes)
    try:
        if repr(output) == repr(expected):
            return True
    except Exception:
        pass

    return False


def _code_defines(code: str, name: str) -> bool:
    """Return True if ``code`` defines a top-level function ``name``."""
    import ast as _ast
    if not name:
        return False
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) \
                and node.name == name:
            return True
    return False


def _dispatch_batch(
    code: str, entry_point: str, test_inputs: list,
    timeout: int, mode: str,
) -> list[tuple[str, Any]]:
    """Pick the right runner for the benchmark's execution mode.

    For ``mode="function"`` benchmarks, we also auto-detect cases where
    the supplied ``code`` is actually a
    raw competitive-programming script (no top-level ``solve`` defn,
    reads from ``sys.stdin``, writes to ``stdout``). LCB's imported
    submissions mix both styles: some samples are
    ``def solve(stdin: str) -> str:`` functions, others are
    ``sys.stdin.read() ... print(...)`` scripts. When the declared
    entry point is not defined in the code, we fall back to the
    stdin-script runner so both formats work transparently. Test
    inputs wrapped as single-element lists (``[[stdin_str]]``) are
    unwrapped for the stdin runner.
    """
    if mode == "stdin":
        return _run_stdin_batch(code, test_inputs, timeout)
    # Declared function mode: check whether the code actually defines
    # the entry point. If not, treat it as a stdin script.
    if entry_point and not _code_defines(code, entry_point):
        stdin_inputs = [
            (ti[0] if isinstance(ti, (list, tuple)) and ti else ti)
            for ti in test_inputs
        ]
        return _run_stdin_batch(code, stdin_inputs, timeout)
    return _run_batch(code, entry_point, test_inputs, timeout)


def run_tests(
    code: str,
    entry_point: str,
    test_inputs: list,
    expected_outputs: list,
    timeout: int = 30,
    mode: str = "function",
) -> list[bool]:
    """Run code on test inputs and compare against expected outputs."""
    results = _dispatch_batch(code, entry_point, test_inputs, timeout, mode)
    passes = []
    for (status, output), expected in zip(results, expected_outputs):
        if status != "ok":
            passes.append(False)
            continue
        passes.append(_values_equal(output, expected))
    return passes


def run_outputs(
    code: str,
    entry_point: str,
    test_inputs: list,
    timeout: int = 30,
    mode: str = "function",
) -> list[tuple[str, Any]]:
    """Run code on test inputs and return raw execution outputs.

    This is used by the mutation pipeline for audit/repair paths that
    need the generated stdout/value rather than a pass/fail comparison.
    """
    return _dispatch_batch(code, entry_point, test_inputs, timeout, mode)
