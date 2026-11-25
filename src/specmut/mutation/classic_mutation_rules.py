"""Deterministic slot-style mutation rules for natural-language specs.

The classic mutation layer uses these rules to produce local, auditable
NL-spec mutants before the LLM mutator is attempted. This file is only
rule data and compiled matchers; mutation application stays in
``mutate.py`` so the quality/impact gates remain in one place.
"""

from __future__ import annotations

import re


ClassicRule = tuple[str, str, str]


CLASSIC_WORD_RULES: dict[str, tuple[ClassicRule, ...]] = {
    "substitute": (
        ("objective", r"\bmaximum\b", "minimum"),
        ("objective", r"\bminimum\b", "maximum"),
        ("objective", r"\bmaximal\b", "minimal"),
        ("objective", r"\bminimal\b", "maximal"),
        ("objective", r"\blargest\b", "smallest"),
        ("objective", r"\bsmallest\b", "largest"),
        ("objective", r"\bhighest\b", "lowest"),
        ("objective", r"\blowest\b", "highest"),
        ("objective", r"\blongest\b", "shortest"),
        ("objective", r"\bshortest\b", "longest"),
        ("aggregation", r"\bsum\b", "product"),
        ("aggregation", r"\bproduct\b", "sum"),
        ("aggregation", r"\btotal\b", "product"),
        ("position", r"\bfirst\b", "last"),
        ("position", r"\blast\b", "first"),
        ("position", r"\bbefore\b", "after"),
        ("position", r"\bafter\b", "before"),
        ("position", r"\btop\b", "bottom"),
        ("position", r"\bbottom\b", "top"),
        ("position", r"\bfront\b", "back"),
        ("position", r"\bback\b", "front"),
        ("direction", r"\bleft\b", "right"),
        ("direction", r"\bright\b", "left"),
        ("direction", r"\bup\b", "down"),
        ("direction", r"\bdown\b", "up"),
        ("direction", r"\bclockwise\b", "counterclockwise"),
        ("direction", r"\bcounterclockwise\b", "clockwise"),
        ("direction", r"\bhorizontal\b", "vertical"),
        ("direction", r"\bvertical\b", "horizontal"),
        ("axis", r"\brow\b", "column"),
        ("axis", r"\bcolumn\b", "row"),
        ("axis", r"\brows\b", "columns"),
        ("axis", r"\bcolumns\b", "rows"),
        ("ordering", r"\bascending\b", "descending"),
        ("ordering", r"\bdescending\b", "ascending"),
        ("ordering", r"\bincreasing\b", "decreasing"),
        ("ordering", r"\bdecreasing\b", "increasing"),
        ("ordering", r"\bnondecreasing\b", "nonincreasing"),
        ("ordering", r"\bnonincreasing\b", "nondecreasing"),
        ("ordering", r"\bnon-decreasing\b", "non-increasing"),
        ("ordering", r"\bnon-increasing\b", "non-decreasing"),
        ("parity", r"\beven\b", "odd"),
        ("parity", r"\bodd\b", "even"),
        ("equality", r"\bsame\b", "different"),
        ("equality", r"\bdifferent\b", "same"),
        ("connectivity", r"\breachable\b", "unreachable"),
        ("connectivity", r"\bunreachable\b", "reachable"),
        ("connectivity", r"\bconnected\b", "disconnected"),
        ("connectivity", r"\bdisconnected\b", "connected"),
        ("type", r"\bintegers?\b", "strings"),
        ("type", r"\bstrings?\b", "integers"),
        ("type", r"\bnumbers?\b", "strings"),
        ("case", r"\buppercase\b", "lowercase"),
        ("case", r"\blowercase\b", "uppercase"),
        ("class", r"\bletters?\b", "digits"),
        ("class", r"\bdigits?\b", "letters"),
        ("state", r"\bblack\b", "white"),
        ("state", r"\bwhite\b", "black"),
        ("state", r"\bempty\b", "non-empty"),
        ("state", r"\bnon-empty\b", "empty"),
        ("state", r"\bachievable\b", "unachievable"),
        ("state", r"\bunachievable\b", "achievable"),
    ),
    "negate": (
        ("literal", r"\bYes\b", "No"),
        ("literal", r"\bNo\b", "Yes"),
        ("predicate", r"\bpossible\b", "impossible"),
        ("predicate", r"\bimpossible\b", "possible"),
        ("predicate", r"\bvalid\b", "invalid"),
        ("predicate", r"\binvalid\b", "valid"),
        ("predicate", r"\bachievable\b", "unachievable"),
        ("predicate", r"\bunachievable\b", "achievable"),
        ("quantifier", r"\bat least\b", "less than"),
        ("quantifier", r"\bat most\b", "greater than"),
        ("quantifier", r"\bexactly\b", "not exactly"),
        ("case", r"\buppercase\b", "not uppercase"),
        ("case", r"\blowercase\b", "not lowercase"),
    ),
    "weaken": (
        ("quantifier", r"\bexactly\b", "at least"),
        ("quantifier", r"\ball\b", "some"),
        ("quantifier", r"\bevery\b", "some"),
        ("type", r"\bintegers?\b", "numbers"),
        ("comparison", r"\bstrictly less than\b", "less than or equal to"),
        ("comparison", r"\bstrictly greater than\b", "greater than or equal to"),
        ("comparison", r"\bless than\b", "at most"),
        ("comparison", r"\bgreater than\b", "at least"),
        ("adjacency", r"\badjacent\b", "within distance at most 2"),
    ),
}


CLASSIC_SYMBOL_RULES: dict[str, tuple[ClassicRule, ...]] = {
    "substitute": (
        ("aggregation", r"\\sum(?![A-Za-z])", r"\prod"),
        ("aggregation", r"\\prod(?![A-Za-z])", r"\sum"),
        ("objective", r"\\max(?![A-Za-z])", r"\min"),
        ("objective", r"\\min(?![A-Za-z])", r"\max"),
    ),
    "negate": (
        ("comparison", r"\\leq\b|\\le\b|<=", ">"),
        ("comparison", r"\\leq\b|\\le\b|<=", r"\neq"),
        ("comparison", r"\\geq\b|\\ge\b|>=", "<"),
        ("comparison", r"\\geq\b|\\ge\b|>=", r"\neq"),
        ("comparison", r"\\lt\b", r"\geq"),
        ("comparison", r"\\gt\b", r"\leq"),
        ("comparison", r"\u2264", ">"),
        ("comparison", r"\u2265", "<"),
        ("comparison", r"(?<![<>=!])<(?![<>=])", ">="),
        ("comparison", r"(?<![<>=!])<(?![<>=])", r"\neq"),
        ("comparison", r"(?<![<>=!])>(?![<>=])", "<="),
        ("comparison", r"(?<![<>=!])>(?![<>=])", r"\neq"),
        ("comparison", r"(?<![<>=!])=(?!=)", r"\neq"),
        ("comparison", r"(?<![<>=!])=(?!=)", "<"),
        ("comparison", r"(?<![<>=!])=(?!=)", ">"),
        ("comparison", r"\\neq\b|!=", "="),
    ),
    "weaken": (
        ("comparison", r"\\leq\b|\\le\b|<=", r"<"),
        ("comparison", r"\\geq\b|\\ge\b|>=", r">"),
        ("comparison", r"\\lt\b", r"\leq"),
        ("comparison", r"\\gt\b", r"\geq"),
        ("comparison", r"\u2264", "<"),
        ("comparison", r"\u2265", ">"),
        ("comparison", r"(?<![<>=!])<(?![<>=])", r"\leq"),
        ("comparison", r"(?<![<>=!])>(?![<>=])", r"\geq"),
        ("comparison", r"(?<![<>=!])=(?!=)", r"\leq"),
        ("comparison", r"(?<![<>=!])=(?!=)", r"\geq"),
    ),
    "perturb": (
        ("comparison", r"\\leq\b|\\le\b|<=", r"<"),
        ("comparison", r"\\leq\b|\\le\b|<=", "="),
        ("comparison", r"\\leq\b|\\le\b|<=", ">"),
        ("comparison", r"\\geq\b|\\ge\b|>=", r">"),
        ("comparison", r"\\geq\b|\\ge\b|>=", "="),
        ("comparison", r"\\geq\b|\\ge\b|>=", "<"),
        ("comparison", r"\\lt\b", r"\leq"),
        ("comparison", r"\\gt\b", r"\geq"),
        ("comparison", r"\u2264", "<"),
        ("comparison", r"\u2265", ">"),
        ("comparison", r"(?<![<>=!])<(?![<>=])", r"\leq"),
        ("comparison", r"(?<![<>=!])<(?![<>=])", "="),
        ("comparison", r"(?<![<>=!])<(?![<>=])", ">"),
        ("comparison", r"(?<![<>=!])>(?![<>=])", r"\geq"),
        ("comparison", r"(?<![<>=!])>(?![<>=])", "="),
        ("comparison", r"(?<![<>=!])>(?![<>=])", "<"),
        ("comparison", r"(?<![<>=!])=(?!=)", "<"),
        ("comparison", r"(?<![<>=!])=(?!=)", ">"),
        ("comparison", r"(?<![<>=!])=(?!=)", r"\leq"),
        ("comparison", r"(?<![<>=!])=(?!=)", r"\geq"),
        ("comparison", r"\\neq\b|!=", "="),
        ("aggregation", r"\\sum(?![A-Za-z])", r"\prod"),
        ("aggregation", r"\\prod(?![A-Za-z])", r"\sum"),
        ("arithmetic", r"(?<=[A-Za-z_}\)\d])\+(?=\d|[A-Z_{}\(])", "-"),
        ("arithmetic", r"(?<=[A-Za-z_}\)\d])\+(?=\d|[A-Z_{}\(])", r"\times "),
        ("arithmetic", r"(?<=[A-Za-z_}\)\d])-(?=\d|[A-Z_{}\(])", "+"),
        ("arithmetic", r"(?<=[A-Za-z_}\)\d])-(?=\d|[A-Z_{}\(])", r"\times "),
        ("arithmetic", r"(?<=[}\)\]\w|])\s+\+\s+(?=[\w\\{\(|])", " - "),
        ("arithmetic", r"(?<=[}\)\]\w|])\s+\+\s+(?=[\w\\{\(|])", r" \times "),
        ("arithmetic", r"(?<=[}\)\]\w|])\s+-\s+(?=[\w\\{\(|])", " + "),
        ("arithmetic", r"(?<=[}\)\]\w|])\s+-\s+(?=[\w\\{\(|])", r" \times "),
    ),
}


CLASSIC_NUMERIC_LITERAL_RE = re.compile(r"(?<![A-Za-z_])\d+(?![A-Za-z_])")

CLASSIC_SEMANTIC_HEAD_RE = re.compile(
    r"(?i)\b("
    r"if|when|print|output|return|find|determine|count|compute|"
    r"minimi[sz]e|maximi[sz]e|is|are|has|have|contains?|must|"
    r"can|may"
    r")\b"
)

CLASSIC_PATH_ADJACENCY_FORMULA_RE = re.compile(
    r"\|i_\{k\+1\}\s*-\s*i_k\|\s*\+\s*"
    r"\|j_\{k\+1\}\s*-\s*j_k\|\s*=\s*1"
)

CLASSIC_PATH_ADJACENCY_PHRASE = (
    "make K moves to adjacent cells (up, down, left, or right)"
)

CLASSIC_PATH_ADJACENCY_RULES_BY_OPERATOR: dict[str, tuple[ClassicRule, ...]] = {
    "weaken": (
        (
            r"\leq 2",
            "make K moves to cells at Manhattan distance at most 2",
            "adjacent",
            "Manhattan distance at most 2",
        ),
    ),
    "negate": (
        (
            r"\neq 1",
            "make K moves to cells that are not at Manhattan distance exactly 1",
            "adjacent",
            "not Manhattan distance 1",
        ),
    ),
    "perturb": (
        (
            "= 2",
            "make K moves to cells at Manhattan distance exactly 2",
            "1",
            "2",
        ),
    ),
}

CLASSIC_ANSWER_TRANSFORM_PHRASES: tuple[tuple[str, str], ...] = (
    ("one more than ", "+1 answer"),
    ("one less than ", "-1 answer"),
    ("twice ", "2x answer"),
)

CLASSIC_ANSWER_OBJECT_RE = re.compile(
    r"(?is)\b(the\s+(?:number|count|sum|total|answer|result|value|score)\s+of\b|"
    r"the\s+(?:minimum|maximum|largest|smallest)\b|"
    r"the\s+answer\b|the\s+result\b|the\s+value\b|the\s+score\b)"
)

CLASSIC_TWO_VALUE_DOMAIN_RE = re.compile(
    r"(?i)\b(is|are|must\s+be|can\s+be)\s+"
    r"([A-Z][A-Za-z0-9_#.]*)\s+or\s+([A-Z][A-Za-z0-9_#.]*)\b"
)
