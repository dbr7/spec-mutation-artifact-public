"""Generate programs from specifications."""

import re

from specmut.config import Config
from specmut.llm import LLMClient


def _extract_code(response: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def replace_docstring(prompt: str, new_spec: str) -> str:
    """Replace the docstring content in a function prompt with a new specification."""
    match = re.search(r'(\"\"\")(.*?)(\"\"\")', prompt, re.DOTALL)
    if match:
        return prompt[:match.start(2)] + "\n    " + new_spec.strip() + "\n    " + prompt[match.end(2):]
    match = re.search(r"(''')(.*?)(''')", prompt, re.DOTALL)
    if match:
        return prompt[:match.start(2)] + "\n    " + new_spec.strip() + "\n    " + prompt[match.end(2):]
    return prompt


def generate(
    client: LLMClient,
    spec: str,
    prompt: str,
    entry_point: str,
    model: str | None = None,
    mode: str = "function",
    cache_salt: str = "",
    temperature: float | None = None,
) -> str:
    """Generate a Python program from a specification.

    ``mode`` picks the prompt template:

    * ``"function"`` (default): a function signature is supplied in
      ``prompt`` and the model is asked to fill in the body. Used for
      HumanEval+ and the wrapped LiveCodeBench interface in this artifact.
    * ``"stdin"``: the model is asked to produce a standalone Python
      script that reads from stdin and writes to stdout, matching the
      native LiveCodeBench execution protocol.

    ``cache_salt`` (default empty) is mixed into the LLM cache key so
    repeated runs with different salts force fresh API calls and
    expose generator stochasticity. ``temperature`` overrides the
    sampling temperature for this call (e.g. 0.6 for variability
    runs, 0 for deterministic runs).
    """
    config = Config()
    if mode == "stdin":
        template = (config.prompts_dir / "codegen_stdin.txt").read_text()
        user_message = template.replace("{spec}", spec)
        system_message = "You are an expert Python programmer."
    else:
        template = (config.prompts_dir / "codegen.txt").read_text()
        user_message = template.replace("{signature}", prompt).replace("{spec}", spec)
        system_message = "You are a Python programming assistant."

    response = client.call(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model=model,
        cache_salt=cache_salt,
        temperature=temperature,
    )
    return _extract_code(response)
