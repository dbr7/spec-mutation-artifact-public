"""Configuration for SpecMut."""

import re
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent


def load_api_keys() -> dict[str, str]:
    """Load the OpenAI API key from the environment or api_key.txt.

    Expected file format (one per line, shell-style export is tolerated):
        OPENAI_API_KEY='<OPENAI_API_KEY>'
    """
    import os as _os

    key_file = REPO_ROOT / "api_key.txt"
    text = key_file.read_text() if key_file.exists() else ""

    keys: dict[str, str] = {}
    env_val = _os.environ.get("OPENAI_API_KEY")
    if env_val:
        keys["openai"] = env_val
        return keys
    m = re.search(r"OPENAI_API_KEY=['\"]([^'\"]+)['\"]", text)
    if m:
        keys["openai"] = m.group(1)
    return keys


@dataclass
class Config:
    api_keys: dict[str, str] = field(default_factory=load_api_keys)
    # Group G (generator / model under study). Drives reference
    # program generation and mutant code generation. This is the model
    # whose SC and GR are characterised by an experimental run.
    model_generator: str = "openai/o4-mini"
    # Group T (mutation tooling). Drives clause decomposition and the
    # active LLM mutation operators. Kept separate so a stronger model
    # can produce high-quality clauses and mutants without confounding
    # the generator evaluation. When left as None, defaults at load time
    # to the generator so single-model runs keep working.
    model_tooling: str | None = "openai/gpt-5.4"
    # Optional OpenAI reasoning controls. These are intentionally split
    # by role because a final run may want a high-effort generator while
    # keeping tooling cheaper or faster.
    generator_reasoning_effort: str | None = None
    tooling_reasoning_effort: str | None = None
    benchmark: str = "humaneval"
    operators: list[str] = field(
        default_factory=lambda: [
            "weaken",
            "negate",
            "perturb",
            "example",
            "substitute",
            "partial_edit",
            "inconsistent_example",
        ]
    )
    mutants_per_operator: int = 1
    classic_mutants: bool = True
    llm_mutants: bool = True
    classic_mutants_per_operator: int = 64
    max_mutants_per_clause: int = 4
    max_operators_per_clause: int = 8
    max_codegen_failures_per_clause: int = 8
    llm_timeout_seconds: float | None = None
    llm_max_attempts: int = 5
    retry_failed_refs: bool = False
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    execution_timeout: int = 30
    max_workers: int = 4

    # Override only when pointing the OpenAI SDK at a compatible endpoint.
    openai_base_url: str | None = None

    def __post_init__(self):
        # A run with only --model-generator supplied should still work.
        if self.model_tooling is None:
            self.model_tooling = self.model_generator
