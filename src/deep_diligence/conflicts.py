"""Conflict detection: where supported findings disagree.

The verifier (verifier.py) checks each claim against its own
evidence. It cannot see two claims side by side, so before this pass
a contradiction surfaced only as two findings sitting in the report
body, and nothing in the pipeline ever said they disagreed. This
pass runs after verification, over the claims that survived it, and
makes disagreement a first-class record.

Detection is deterministic. Each supported claim is reduced to
assertions — (metric, period, value) — extracted in code: a governed
metric lexicon finds the metric mentions in the claim, figure spans
are typed (money / percent / count), each figure is assigned to the
nearest mentioned metric of its type, and each figure takes the
period stated nearest to it (periods are normalised to the year, so
FY2025 and 2025 are the same period and FY2024 is a different one).
Claims are then keyed by (entity, metric, period) — the entity is
the case subject — and where two supported claims assert different
values for the same key, a conflict record is created holding both
sides: the claims, their evidence, both values verbatim.

Nothing here resolves anything. Each side is labelled in code by
source class — primary sources (filings, official reports) rank
above secondary ones (news-style coverage, vendor marketing) — and,
for same-period facts, the side resting on the later-dated document
is marked more recent. Those labels are context for the analyst,
presented in the report; the system takes no position on which
value is true, neither side is ever suppressed or merged, and the
analyst's only move is to mark the conflict NOTED with a note. The
record stays in the report permanently.
"""

from __future__ import annotations

import re

from .store import Store, utcnow

# ---------------------------------------------------------------------------
# Periods. A period token is an FY year, a month-year, or a bare
# year; all normalise to the 4-digit year. Period spans are masked
# out of figure extraction so a year is never read as a figure.
# ---------------------------------------------------------------------------
PERIOD = re.compile(
    r"\bFY\s?(\d{4})\b"
    r"|\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})\b"
    r"|\b((?:19|20)\d{2})\b"
)

# ---------------------------------------------------------------------------
# Figures, typed. Money and percent spans win over bare numbers they
# contain ($412M is one figure, not $412M plus 412).
# ---------------------------------------------------------------------------
MONEY = re.compile(r"\$[\d,]+(?:\.\d+)?[MBK]?")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# ---------------------------------------------------------------------------
# The metric lexicon: (name, value type, patterns). A metric is
# "mentioned" where one of its patterns matches. Overlaps resolve
# per value type, longest match first — "net debt" beats "debt",
# "operating cash flow" beats "cash", "gross margin" beats "margin".
# ---------------------------------------------------------------------------
METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # money
    ("operating cash flow", "money", (r"operating cash flow", r"cash flow")),
    ("net debt", "money", (r"net debt",)),
    ("debt", "money", (r"\bdebt\b", r"\bborrowing\b")),
    ("cash", "money", (r"\bcash\b",)),
    ("revenue", "money", (r"\brevenue\b",)),
    ("price", "money", (r"\bpricing\b", r"\bprice\b", r"starts at")),
    ("funding raised", "money", (r"funding round", r"\braised\b")),
    # percent
    ("gross margin", "percent", (r"gross margin",)),
    ("EBITDA margin", "percent", (r"ebitda margin",)),
    ("margin", "percent", (r"\bmargin\b",)),
    ("revenue growth", "percent", (r"revenue\s+(?:grew|growth)",)),
    ("market growth", "percent", (r"market\s+(?:grew|growth)",)),
    ("growth", "percent", (r"\b(?:grew|growth)\b",)),
    ("market share", "percent", (r"\d+(?:\.\d+)?%\s+of\s+the\s+[^.;,]*\bmarket\b",)),
    ("product-range share", "percent", (r"\d+(?:\.\d+)?%\s+of\s+the\s+product range",)),
    # count
    ("headcount", "count", (r"\bemployees\b", r"\bemployed\b", r"\bpeople\b", r"\bworkforce\b", r"\bheadcount\b", r"\bstaff\b")),
    ("result time", "count", (r"result time", r"returns results", r"return results", r"results in")),
    ("runway", "count", (r"\brunway\b",)),
    ("installed base", "count", (r"installed in", r"installed base", r"\binstallations\b")),
    ("deployments", "count", (r"\bdeployed\b", r"\bdeployments\b")),
    ("warehouses", "count", (r"\bwarehouses\b",)),
    ("facilities", "count", (r"\bfacilities\b",)),
    ("fleet size", "count", (r"\btrucks\b", r"\bfleet\b")),
    ("customers", "count", (r"\bcustomers\b",)),
    ("countries", "count", (r"\bcountries\b",)),
)

# Source classes, applied in code from the evidence kind. Primary
# sources rank above secondary ones; the rank orders the labels in
# the report and never picks a winner.
SOURCE_CLASSES: dict[str, tuple[str, int]] = {
    "filing": ("primary", 2),
    "regulatory": ("primary", 2),
    "operations": ("primary", 2),
    "investor": ("primary", 2),
    "news": ("secondary", 1),
    "product": ("secondary", 1),
    "pricing": ("secondary", 1),
}


def source_class(kind: str) -> tuple[str, int]:
    """(class, rank) for an evidence kind; unlisted kinds are
    treated as secondary."""
    return SOURCE_CLASSES.get(kind, ("secondary", 1))


def _periods(text: str) -> list[dict]:
    found = []
    for match in PERIOD.finditer(text):
        year = match.group(1) or match.group(2) or match.group(3)
        found.append({"span": match.span(), "period": year})
    return found


def _figures(text: str, period_spans: list[tuple[int, int]]) -> list[dict]:
    def masked(span: tuple[int, int]) -> bool:
        return any(s <= span[0] and span[1] <= e for s, e in period_spans)

    figures: list[dict] = []
    covered: list[tuple[int, int]] = []
    for pattern, kind in ((MONEY, "money"), (PERCENT, "percent")):
        for match in pattern.finditer(text):
            if masked(match.span()):
                continue
            covered.append(match.span())
            figures.append(
                {"span": match.span(), "type": kind, "text": match.group()}
            )
    for match in NUMBER.finditer(text):
        span = match.span()
        if masked(span):
            continue
        if any(s <= span[0] and span[1] <= e for s, e in covered):
            continue
        # A number glued to a product code (AF-40, HGX-9) is part of
        # the code, not a figure: skip digits that follow a letter,
        # or a hyphen that follows a letter.
        start = span[0]
        if start >= 1 and (
            text[start - 1].isalpha()
            or (text[start - 1] == "-" and start >= 2 and text[start - 2].isalpha())
        ):
            continue
        figures.append({"span": span, "type": "count", "text": match.group()})
    return sorted(figures, key=lambda f: f["span"])


def _normalised(figure: dict) -> tuple:
    """The comparison key for a figure: type, numeric amount, and
    (for money) the magnitude suffix. $412M and $412M agree; $412M
    and $421M do not; 1,150 and 1150 agree."""
    text = figure["text"]
    if figure["type"] == "money":
        body = text[1:]
        suffix = ""
        if body and body[-1] in "MBK":
            suffix = body[-1]
            body = body[:-1]
        return ("money", float(body.replace(",", "")), suffix)
    if figure["type"] == "percent":
        return ("percent", float(text[:-1].replace(",", "")), "")
    return ("count", float(text.replace(",", "")), "")


def _metric_mentions(text: str) -> list[dict]:
    lowered = text.lower()
    raw: list[dict] = []
    for name, kind, patterns in METRICS:
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                raw.append(
                    {"metric": name, "type": kind, "span": match.span()}
                )
    # Resolve overlaps within each value type, longest match first,
    # so the specific metric keeps its span ("net debt" over "debt").
    kept: list[dict] = []
    for kind in ("money", "percent", "count"):
        candidates = sorted(
            (m for m in raw if m["type"] == kind),
            key=lambda m: (-(m["span"][1] - m["span"][0]), m["span"][0]),
        )
        taken: list[tuple[int, int]] = []
        for mention in candidates:
            s, e = mention["span"]
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            kept.append(mention)
    return kept


def _gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def _mention_distance(figure_span: tuple[int, int], mention_span: tuple[int, int]) -> float:
    """Distance from a figure to a metric mention. A mention that
    follows the figure counts half: counts in prose usually precede
    the noun they count ("4,800 foodservice customers"), so the
    following noun is the figure's metric even when another metric's
    noun sits a few characters closer behind it."""
    gap = _gap(figure_span, mention_span)
    if mention_span[0] >= figure_span[1]:
        return gap * 0.5
    return float(gap)


def extract_assertions(claim: str) -> list[dict]:
    """Reduce one claim to its (metric, period, value) assertions,
    deterministically. A claim yields an assertion for each figure
    that a mentioned metric of the figure's type claims as its own
    (nearest mention wins); the assertion's period is the period
    stated nearest the figure, or None when the claim states none.
    Figures no metric claims (a year, a certification number, a
    duration) yield nothing — they cannot conflict."""
    periods = _periods(claim)
    period_spans = [p["span"] for p in periods]
    mentions = _metric_mentions(claim)
    assertions: list[dict] = []
    for figure in _figures(claim, period_spans):
        candidates = [m for m in mentions if m["type"] == figure["type"]]
        if not candidates:
            continue
        metric = min(
            candidates,
            key=lambda m: _mention_distance(figure["span"], m["span"]),
        )["metric"]
        period = None
        if periods:
            nearest = min(
                periods,
                key=lambda p: (
                    _gap(figure["span"], p["span"]),
                    0 if p["span"][0] >= figure["span"][0] else 1,
                ),
            )
            period = nearest["period"]
        assertions.append(
            {
                "metric": metric,
                "period": period,
                "value": _normalised(figure),
                "value_text": figure["text"],
            }
        )
    # One assertion per (metric, period, value) per claim.
    seen: set = set()
    unique: list[dict] = []
    for assertion in assertions:
        key = (assertion["metric"], assertion["period"], assertion["value"])
        if key not in seen:
            seen.add(key)
            unique.append(assertion)
    return unique


def detect_conflicts(store: Store, case: dict, claims: list[dict]) -> list[dict]:
    """Key the supported claims by (entity, metric, period) and
    build a conflict record for every key whose claims assert more
    than one distinct value. Same value from two sources is
    corroboration: the claims sit together on one side, and no
    record is created unless another side disagrees."""
    case_id = case["case_id"]
    entity = case["subject"]
    entity_ref = case["subject_ref"]
    evidence = {ev["evidence_id"]: ev for ev in store.list_evidence(case_id)}
    groups: dict[tuple, dict[tuple, list[dict]]] = {}
    for claim in claims:
        for assertion in extract_assertions(claim["claim"]):
            key = (assertion["metric"], assertion["period"])
            side = groups.setdefault(key, {}).setdefault(assertion["value"], [])
            if any(s["claim_id"] == claim["claim_id"] for s in side):
                continue
            sources = []
            for evidence_id in claim["evidence_ids"]:
                ev = evidence.get(evidence_id)
                if ev is None:
                    continue
                cls, rank = source_class(ev["kind"])
                sources.append(
                    {
                        "evidence_id": evidence_id,
                        "source": ev["source"],
                        "kind": ev["kind"],
                        "doc_date": ev["doc_date"],
                        "source_class": cls,
                        "source_rank": rank,
                    }
                )
            best = max(sources, key=lambda s: s["source_rank"], default=None)
            side.append(
                {
                    "value": assertion["value"],
                    "value_text": assertion["value_text"],
                    "claim_id": claim["claim_id"],
                    "finding_id": claim["finding_id"],
                    "question_id": claim["question_id"],
                    "claim": claim["claim"],
                    "evidence_ids": list(claim["evidence_ids"]),
                    "sources": sources,
                    "source_class": best["source_class"] if best else "secondary",
                    "source_rank": best["source_rank"] if best else 1,
                    "latest_date": max(
                        (s["doc_date"] for s in sources), default=""
                    ),
                }
            )
    records: list[dict] = []
    for (metric, period), by_value in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        if len(by_value) < 2:
            continue
        sides = []
        for value, members in sorted(
            by_value.items(), key=lambda item: (item[0][1], item[0][2])
        ):
            sides.append(
                {
                    "value": list(value),
                    "value_text": members[0]["value_text"],
                    "claims": members,
                }
            )
        latest = max(
            (m["latest_date"] for members in by_value.values() for m in members),
            default="",
        )
        for side in sides:
            for member in side["claims"]:
                member["more_recent"] = bool(
                    latest and member["latest_date"] == latest
                    and any(
                        other["latest_date"] < latest
                        for members in by_value.values()
                        for other in members
                        if other is not member
                    )
                )
        records.append(
            {
                "conflict_id": f"CNF-{len(records) + 1:04d}",
                "case_id": case_id,
                "key": {
                    "entity": entity,
                    "entity_ref": entity_ref,
                    "metric": metric,
                    "period": period,
                },
                "sides": sides,
                "status": "OPEN",
                "created_at": utcnow(),
                "noted_by": None,
                "note": None,
                "noted_at": None,
            }
        )
    return records
