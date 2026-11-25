# Rethinking Mutation Testing When the Natural Language Specification Is the Program


* **The prompt templates are available in `src/prompts/*.txt`; survivor judging and developer-action prompts are built in `src/specmut/analysis/analyze_equivalence_emd_prompt.py`, `src/specmut/analysis/analyze_actionable_pipeline.py`, and `src/run_action_experiments.py`.**
* **The deterministic mutation rules are available in `src/specmut/mutation/classic_mutation_rules.py`.**
* **The SpecMut implementation is available in `src/`; the main entry points are `src/run.py`, `src/run_actionable_pipeline.py`, and `src/run_action_experiments.py`.**

This artifact accompanies the paper "Rethinking Mutation Testing When the Natural Language Specification Is the Program". It contains the implementation, prompt templates, mutation rules, and curated result files used for the paper tables and figures.


## Contents

- `src/`: implementation of SpecMut.
  - `src/run.py`: main mutation-analysis runner.
  - `src/run_actionable_pipeline.py`: survivor routing.
  - `src/run_action_experiments.py`: mutation-guided test generation and requirement-emphasis regeneration.
  - `src/specmut/`: importable package for benchmark loading, clause decomposition, mutation generation, execution, metrics, survivor judging, and action routing.
  - `src/specmut/mutation/classic_mutation_rules.py`: deterministic mutation templates. This is the full fixed rule set referred to in the paper.
  - `src/prompts/`: prompt templates for code generation, clause decomposition, and LLM-guided mutation.
- `results/openai_o4-mini/`: curated final result batches for the paper evaluation.
- `plots/`: scripts and data for the task-level plots used in the paper.


## Installation

To run the implementation:

```bash
cd src
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Re-running Experiments

Full reruns call external LLM APIs and may incur cost. Set `OPENAI_API_KEY` through an environment variable or an untracked key file before running.

For a small smoke run:

```bash
cd src
python run.py --help
```

The paper configuration uses:

- generator model: `openai/o4-mini` with high reasoning effort and temperature 0;
- auxiliary model: `openai/gpt-5.4` for clause decomposition, LLM-guided mutation, survivor judging, requirement-emphasis regeneration, and mutation-guided test generation;
- benchmarks: post-cutoff LiveCodeBench tasks and all HumanEval+ tasks;
- seven mutation operators: Weaken, Substitute, Negate, Perturb, Incorrect Example, Requirement Inconsistency, and Example Inconsistency.

## Citation


```bibtex
@inproceedings{kim2027rethinking,
  author    = {Kim, Jinhan and Tonella, Paolo},
  title     = {Rethinking Mutation Testing When the Natural Language Specification Is the Program},
  booktitle = {Proceedings of the 49th IEEE/ACM International Conference on Software Engineering},
  year      = {2027}
}
```
