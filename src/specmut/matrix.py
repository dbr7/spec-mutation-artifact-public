"""Build the clause-test matrix and compute SC/GR.

This module also produces a parallel mutant-test rollup that
preserves per-mutant rows. The two views are computed in a single
pass over the mutation list and stored on the same Matrix object,
so callers can choose whichever aggregation level matches the
question they are asking:

* Clause-test matrix (``data``) is the OR-aggregated view: one
  row per decomposed spec clause, used for SC and GR. SC uses
  all decomposed clauses as its denominator; clauses with no
  evaluable mutant remain visible as uncovered clauses instead of
  disappearing from the metric.
* Mutant-test matrix (``mutant_data``) is the raw per-mutant
  view: one row per killed (clause, operator) pair. This is the
  standard mutation-testing aggregation level and is used for the
  classical mutation kill rate and per-operator analysis.

Terminology. We use the classical mutation-testing axis:

* A mutant is **killed** iff at least one test in ``T`` distinguishes
  its output from the reference. This is observable.
* A mutant **survived** iff every test in ``T`` produced the same
  output as the reference. Survived is a strictly weaker property
  than the classical "equivalent mutant" notion, which requires
  proof of identical behaviour on all possible inputs (an
  undecidable property in general). We deliberately avoid the
  word "equivalent" here because what we measure is test-suite
  indistinguishability, not semantic equivalence.

Note on grounding ratios: a column is "grounded" iff at least one
row has a 1 in it. Because the clause matrix is the row-wise OR of
the mutant matrix, the set of grounded columns is identical for
both views. We therefore expose a single ``gr`` and do not store a
separate mutant_gr.
"""

from dataclasses import dataclass, field


@dataclass
class MutationResult:
    clause_idx: int
    clause_text: str
    operator: str
    survived: bool  # mutant outputs match the reference on every test in T
    mutated_spec: str
    mutant_code: str  # generated P_m
    test_results: list[bool]  # True = pass, False = fail
    noop: bool = False  # mutator returned the spec unchanged (no textual edit)
    # Slot-level metadata, populated only for ``swap``-operator
    # mutations. ``slot_type`` names the slot category (e.g.
    # "container_type", "quantifier"); ``slot_original`` and
    # ``slot_target`` record the substituted token pair so the
    # downstream report can attribute kill rates to specific
    # fault categories rather than to a single opaque "swap" bucket.
    slot_type: str | None = None
    slot_original: str | None = None
    slot_target: str | None = None
    # Candidate multiplicity metadata. ``variant_idx`` is 1-based and
    # records which repeated attempt of the same (clause, operator)
    # produced this mutant when --mutants-per-operator > 1.
    variant_idx: int = 1
    variant_total: int = 1
    # Set when this mutation produced a mutated_spec byte-identical
    # to a previously-emitted mutation on the same clause from a
    # DIFFERENT operator. Records the prior operator's name so the
    # collision is auditable. ``noop`` is set at the same time so
    # the matrix and survival statistics already exclude these
    # cross-operator duplicates.
    duplicate_of: str | None = None


@dataclass
class Matrix:
    # ----- Clause-level view (the original "spec coverage" matrix) -----
    data: list[list[int]]  # clauses x tests, 1 = mutation kills test
    clause_names: list[str]
    num_tests: int
    sc: float | None
    gr: float | None
    survival_rate: float
    noop_rate: float
    sc_evaluable: float | None
    num_total_clauses: int
    num_killed_clauses: int
    num_evaluable_clauses: int
    num_survived_only_clauses: int
    num_no_evaluable_clauses: int
    mutations: list[MutationResult]

    # ----- Mutant-level view (classical mutation-testing matrix) -----
    # One row per killed (clause, operator) pair, in the same order
    # they appear in ``mutations``. ``mutant_meta`` holds the
    # metadata for each row so callers can link a row back to its
    # source clause and operator without re-walking ``mutations``.
    mutant_data: list[list[int]] = field(default_factory=list)
    mutant_meta: list[dict] = field(default_factory=list)
    # Classical mutation score: killed mutants / evaluable non-noop
    # mutants. A mutant is "killed" iff at least one cell in its row
    # is 1. None when there are no evaluable mutants.
    kill_rate: float | None = None
    # Per-operator kill rate: same definition restricted to mutants
    # whose ``operator`` matches the key. Useful for the operator
    # effectiveness analysis (RQ3).
    per_operator_kill_rate: dict[str, float] = field(default_factory=dict)
    # Per-operator evaluable mutant counts, useful for
    # interpreting the per-operator kill rate.
    per_operator_count: dict[str, int] = field(default_factory=dict)
    # Per-slot kill rate and counts: only populated for swap-operator
    # mutations, grouped by slot_type. This is the analytical view
    # that lets RQ3 attribute swap kills to specific fault categories
    # (container_type vs quantifier vs order, etc.) rather than to
    # the single opaque "swap" bucket in per_operator_kill_rate.
    per_slot_kill_rate: dict[str, float] = field(default_factory=dict)
    per_slot_count: dict[str, int] = field(default_factory=dict)


def build_matrix(clauses: list[str], mutations: list[MutationResult], num_tests: int) -> Matrix:
    """Build the clause-test and mutant-test matrices from mutation results.

    Both matrices are computed in a single pass: each killed mutation
    contributes one row to ``mutant_data`` and is OR-folded into the
    corresponding clause row in ``data``. Survived and noop mutants
    are excluded from both matrices but counted in the evaluable rates.
    """
    num_clauses = len(clauses)
    data = [[0] * num_tests for _ in range(num_clauses)]

    mutant_data: list[list[int]] = []
    mutant_meta: list[dict] = []
    # Per-operator kill bookkeeping. Counts all evaluable mutants per
    # operator and how many of them were actually distinguished by
    # at least one test.
    op_total: dict[str, int] = {}
    op_killed: dict[str, int] = {}
    # Per-slot kill bookkeeping for swap mutations only.
    slot_total: dict[str, int] = {}
    slot_killed: dict[str, int] = {}

    evaluable_clauses = set()  # clauses with at least one non-noop mutation
    killed_clauses = set()     # clauses with at least one killed mutant
    total_mutations = 0
    evaluable_mutations = 0
    survived_mutations = 0
    noop_mutations = 0
    killed_mutants = 0

    for m_idx, m in enumerate(mutations):
        total_mutations += 1
        if m.noop:
            noop_mutations += 1
            # Noops are also trivially survivors (no textual edit
            # ever propagated to the code), so we count them in both
            # rates: noop_rate isolates the "mutator did nothing"
            # subset, survival_rate is the union.
            survived_mutations += 1
            continue
        evaluable_mutations += 1
        evaluable_clauses.add(m.clause_idx)
        op_total[m.operator] = op_total.get(m.operator, 0) + 1
        if m.slot_type:
            slot_total[m.slot_type] = slot_total.get(m.slot_type, 0) + 1
        if m.survived:
            survived_mutations += 1
            continue

        # Build the per-mutant row.
        row = [0 if passed else 1 for passed in m.test_results]
        mutant_data.append(row)
        mutant_meta.append({
            "mutation_idx": m_idx,
            "clause_idx": m.clause_idx,
            "operator": m.operator,
            "slot_type": m.slot_type,
            "slot_original": m.slot_original,
            "slot_target": m.slot_target,
            "variant_idx": m.variant_idx,
            "variant_total": m.variant_total,
        })

        is_killed = any(row)
        if is_killed:
            killed_mutants += 1
            op_killed[m.operator] = op_killed.get(m.operator, 0) + 1
            killed_clauses.add(m.clause_idx)

        # Per-slot bookkeeping (swap mutations only).
        if m.slot_type and is_killed:
            slot_killed[m.slot_type] = slot_killed.get(m.slot_type, 0) + 1

        # OR the row into the clause-level matrix.
        if is_killed:
            for j, passed in enumerate(m.test_results):
                if not passed:
                    data[m.clause_idx][j] = 1

    # SC: all decomposed clauses are targets. Clauses with no
    # evaluable mutant stay in the denominator because, from the
    # developer's perspective, the tool has still failed to produce
    # evidence that tests exercise that clause.
    sc = len(killed_clauses) / num_clauses if num_clauses > 0 else None

    # Backward-compatible diagnostic: the old denominator restricted
    # SC to clauses that produced at least one evaluable mutant.
    # This remains useful for separating test weakness from mutation
    # generation weakness, but it is no longer the primary score.
    sc_evaluable = (
        len(killed_clauses) / len(evaluable_clauses)
        if evaluable_clauses else None
    )
    num_evaluable_clauses = len(evaluable_clauses)
    num_killed_clauses = len(killed_clauses)
    num_survived_only_clauses = len(evaluable_clauses - killed_clauses)
    num_no_evaluable_clauses = max(0, num_clauses - num_evaluable_clauses)

    # GR: fraction of tests with at least one supporting clause.
    # Identical whether computed from the clause matrix or the mutant
    # matrix, because the clause matrix is the row-OR of the mutant
    # matrix and a column is "supported" iff some row has a 1.
    grounded_tests = sum(1 for j in range(num_tests) if any(data[i][j] for i in range(num_clauses)))
    gr = grounded_tests / num_tests if num_tests > 0 else None

    # Mutation kill rate: classical mutation testing's "mutation
    # score", killed mutants divided by all evaluable non-noop
    # mutants. None when there are no evaluable mutants.
    kill_rate = (
        killed_mutants / evaluable_mutations
        if evaluable_mutations > 0 else None
    )

    per_operator_kill_rate = {
        op: op_killed.get(op, 0) / op_total[op]
        for op in op_total
    }
    per_slot_kill_rate = {
        slot: slot_killed.get(slot, 0) / slot_total[slot]
        for slot in slot_total
    }

    survival_rate = survived_mutations / total_mutations if total_mutations > 0 else 0.0
    noop_rate = noop_mutations / total_mutations if total_mutations > 0 else 0.0

    return Matrix(
        data=data,
        clause_names=clauses,
        num_tests=num_tests,
        sc=sc,
        gr=gr,
        survival_rate=survival_rate,
        noop_rate=noop_rate,
        sc_evaluable=sc_evaluable,
        num_total_clauses=num_clauses,
        num_killed_clauses=num_killed_clauses,
        num_evaluable_clauses=num_evaluable_clauses,
        num_survived_only_clauses=num_survived_only_clauses,
        num_no_evaluable_clauses=num_no_evaluable_clauses,
        mutations=mutations,
        mutant_data=mutant_data,
        mutant_meta=mutant_meta,
        kill_rate=kill_rate,
        per_operator_kill_rate=per_operator_kill_rate,
        per_operator_count=dict(op_total),
        per_slot_kill_rate=per_slot_kill_rate,
        per_slot_count=dict(slot_total),
    )
