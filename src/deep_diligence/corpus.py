"""The offline evidence corpus and the gold case packs.

Every company, person, and figure here is invented for this blueprint.
Documents are generated in-repo: narrative prose plus a "Key facts"
ledger section, so the deterministic offline reader (and the eval
harness) can work against facts whose placement is known by
construction. Each gold case plants facts across documents, plants at
least one contradiction between documents, and keeps a similarly named
distractor company in the universe but outside every case's scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Keyword stems the offline reader uses to judge which workstream a
# fact belongs to. A fact is written to match its own workstream best.
WORKSTREAM_KEYWORDS: dict[str, list[str]] = {
    "financial-scan": [
        "revenue", "margin", "cash", "debt", "profit", "fund",
        "runway", "burn", "earning", "loss",
    ],
    "product-scan": [
        "product", "system", "platform", "mwh", "deploy", "certif",
        "pric", "service agreement", "instal", "regulator", "facilit",
        "warehouse", "fleet", "distribution", "customer", "employ",
    ],
    "market-scan": [
        "market", "share", "competitor", "demand", "contract",
        "expan", "segment", "adopt", "region",
    ],
}


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    question: str  # gold question id this fact answers


@dataclass(frozen=True)
class CorpusDoc:
    slug: str
    company: str        # subject_ref
    company_name: str
    kind: str           # filing | product | pricing | news | operations | regulatory | investor
    title: str
    source: str         # human-readable source label
    date: str           # ISO date
    body: str           # narrative prose (without the Key facts ledger)
    facts: tuple[Fact, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        ledger = "\n".join(f"- [{f.id}] {f.text}" for f in self.facts)
        return (
            f"# {self.title}\n\nSource: {self.source} · {self.date}\n\n"
            f"{self.body}\n\n## Key facts\n\n{ledger}\n"
        )


def _doc(**kw) -> CorpusDoc:
    kw["facts"] = tuple(Fact(*f) for f in kw.get("facts", ()))
    return CorpusDoc(**kw)


CORPUS: tuple[CorpusDoc, ...] = (
    # ------------------------------------------------------------------
    # Helvion Grid Systems — grid-scale battery storage (vendor case)
    # ------------------------------------------------------------------
    _doc(
        slug="helvion-annual-report-fy2025", company="helvion-grid",
        company_name="Helvion Grid Systems", kind="filing",
        title="Helvion Grid Systems — Annual Report FY2025",
        source="Company filing (annual report)", date="2026-02-14",
        body=(
            "Helvion Grid Systems closed FY2025 with its strongest year since "
            "founding. Order intake from European utilities drove the top "
            "line, and a shift toward larger containers improved the gross "
            "margin profile. Management described the balance sheet as "
            "conservative, with borrowing held against contracted backlog.\n\n"
            "The board noted that working capital absorbed cash early in the "
            "year as two large projects were delivered in parallel, before "
            "milestone payments landed in the fourth quarter."
        ),
        facts=(
            ("F-HEL-101", "Revenue grew 18% year over year to $412M in FY2025.", "q1"),
            ("F-HEL-102", "Gross margin was 31% in FY2025, up from 27% in FY2024.", "q1"),
            ("F-HEL-103", "Operating cash flow was positive at $36M in FY2025.", "q1"),
            ("F-HEL-104", "Net debt stood at $58M at the end of FY2025.", "q1"),
        ),
    ),
    _doc(
        slug="helvion-product-overview", company="helvion-grid",
        company_name="Helvion Grid Systems", kind="product",
        title="Helvion HGX-9 Product Overview",
        source="Vendor product documentation", date="2025-11-03",
        body=(
            "The HGX-9 is Helvion's flagship containerised storage system for "
            "utility-scale projects. The overview positions it against "
            "smaller modular units on footprint and round-trip efficiency, "
            "and lists the certifications the platform carries for grid "
            "connection in its main markets."
        ),
        facts=(
            ("F-HEL-201", "The flagship HGX-9 storage system delivers 9.2 MWh per container.", "q2"),
            ("F-HEL-202", "Helvion has deployed 140 storage systems across 9 countries.", "q2"),
            ("F-HEL-203", "The HGX-9 carries UL 9540 and IEC 62933 certification.", "q2"),
        ),
    ),
    _doc(
        slug="helvion-pricing-sheet", company="helvion-grid",
        company_name="Helvion Grid Systems", kind="pricing",
        title="HGX-9 Pricing and Commercial Terms",
        source="Vendor pricing sheet", date="2026-01-20",
        body=(
            "The pricing sheet covers the HGX-9 hardware, installation "
            "options, and the service bundle attached to each sale. Volume "
            "discounts are quoted on application and are not listed."
        ),
        facts=(
            ("F-HEL-204", "List pricing for the HGX-9 starts at $1.9M per system before installation.", "q2"),
            ("F-HEL-205", "A 10-year service agreement is bundled with every HGX-9 sale.", "q2"),
        ),
    ),
    _doc(
        slug="helvion-news-iberian-contract", company="helvion-grid",
        company_name="Helvion Grid Systems", kind="news",
        title="Helvion Wins Iberian Storage Contract",
        source="Trade press article", date="2026-03-08",
        body=(
            "Trade coverage of Helvion's results season led with the Iberian "
            "contract win, the company's largest single award to date. The "
            "same coverage summarised the annual results for a general "
            "readership; its revenue figure differs from the annual report's."
        ),
        facts=(
            ("F-HEL-301", "Helvion won a major storage contract with an Iberian utility in March 2026.", "q3"),
            ("F-HEL-106", "Coverage of the results put Helvion's FY2025 revenue at $421M.", "q1"),
        ),
    ),
    _doc(
        slug="helvion-news-expansion", company="helvion-grid",
        company_name="Helvion Grid Systems", kind="news",
        title="Helvion Expands European Footprint",
        source="Trade press article", date="2026-04-19",
        body=(
            "Helvion used an industry conference to set out its European "
            "expansion, including a second factory and a larger service "
            "organisation. Analysts quoted in the piece sized the company's "
            "current share of the regional market."
        ),
        facts=(
            ("F-HEL-302", "Analysts estimate Helvion holds about 11% of the European grid-storage market.", "q3"),
            ("F-HEL-303", "Helvion's Poland factory expands its European market coverage from 2027.", "q3"),
        ),
    ),
    # ------------------------------------------------------------------
    # Copperline Provisions — food distribution (acquisition screen)
    # ------------------------------------------------------------------
    _doc(
        slug="copperline-annual-report-fy2025", company="copperline-provisions",
        company_name="Copperline Provisions", kind="filing",
        title="Copperline Provisions — Annual Report FY2025",
        source="Company filing (annual report)", date="2026-02-27",
        body=(
            "Copperline Provisions, a Midwest foodservice distributor, "
            "reported a year of modest growth and thinner profitability. "
            "Fuel and labour costs rose faster than the prices the company "
            "could pass through to independent restaurants, its core "
            "customer group."
        ),
        facts=(
            ("F-COP-101", "Revenue was $268M in FY2025, up 6% on FY2024.", "q1"),
            ("F-COP-102", "EBITDA margin narrowed to 4.1% in FY2025 from 5.0% in FY2024.", "q1"),
            ("F-COP-103", "The company carried $41M of debt against $9M of cash at year end.", "q1"),
        ),
    ),
    _doc(
        slug="copperline-operations-overview", company="copperline-provisions",
        company_name="Copperline Provisions", kind="operations",
        title="Copperline Operations Overview",
        source="Company operations brief", date="2025-10-12",
        body=(
            "The operations brief describes Copperline's warehouse network, "
            "refrigerated fleet, and the private-label range it has built "
            "over the last decade, together with the headcount the network "
            "carried at the end of the financial year."
        ),
        facts=(
            ("F-COP-201", "Copperline operates 6 regional distribution warehouses across the Midwest.", "q2"),
            ("F-COP-202", "Its fleet of 210 refrigerated trucks serves 4,800 foodservice customers.", "q2"),
            ("F-COP-203", "Private-label lines account for 22% of the product range.", "q2"),
            ("F-COP-204", "Copperline employed 1,150 people at the end of FY2025.", "q2"),
        ),
    ),
    _doc(
        slug="copperline-news-regional", company="copperline-provisions",
        company_name="Copperline Provisions", kind="news",
        title="Regional Distributors Feel the Squeeze",
        source="Trade press article", date="2026-01-25",
        body=(
            "A trade survey of regional foodservice distribution placed "
            "Copperline among the larger independents and described a "
            "market growing slowly, with institutional demand — schools "
            "and hospitals — the brightest spot."
        ),
        facts=(
            ("F-COP-301", "Copperline is the second-largest independent foodservice distributor in its region.", "q3"),
            ("F-COP-302", "The regional foodservice distribution market grew 3% in 2025 as demand from schools and hospitals rose.", "q3"),
        ),
    ),
    _doc(
        slug="copperline-news-hiring", company="copperline-provisions",
        company_name="Copperline Provisions", kind="news",
        title="Copperline Adds Staff Ahead of Spring Season",
        source="Regional business journal", date="2026-02-02",
        body=(
            "A regional business journal profiled Copperline's spring "
            "hiring drive and the arrival of a national competitor in its "
            "home territory. Quoting the chief executive, the profile gave "
            "a year-end workforce figure that does not match the operations "
            "brief."
        ),
        facts=(
            ("F-COP-303", "Its largest competitor, a national distributor, entered the region in 2025.", "q3"),
            ("F-COP-304", "Copperline ended FY2025 with roughly 1,300 employees, the chief executive said.", "q2"),
        ),
    ),
    # ------------------------------------------------------------------
    # Selene Diagnostics — rapid diagnostics (partnership case)
    # ------------------------------------------------------------------
    _doc(
        slug="selene-product-brief", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="product",
        title="Selene Rapid Respiratory Panel — Product Brief",
        source="Vendor product documentation", date="2025-09-15",
        body=(
            "Selene's product brief describes a benchtop rapid respiratory "
            "panel aimed at hospital laboratories, its claimed result "
            "time, and the installed base the company reports."
        ),
        facts=(
            ("F-SEL-101", "Selene's lead product is a rapid respiratory panel that returns results in 25 minutes.", "q1"),
            ("F-SEL-103", "The system is installed in 85 hospital laboratories.", "q1"),
        ),
    ),
    _doc(
        slug="selene-regulatory-note", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="regulatory",
        title="Regulatory Status Note — Respiratory Panel",
        source="Company regulatory summary", date="2025-07-02",
        body=(
            "The regulatory summary records the panel's clearance status "
            "in the United States and the submissions in progress "
            "elsewhere."
        ),
        facts=(
            ("F-SEL-102", "The panel holds regulatory clearance from the FDA, granted in June 2025.", "q1"),
        ),
    ),
    _doc(
        slug="selene-investor-update", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="investor",
        title="Selene Diagnostics — Investor Update Q3 2025",
        source="Investor update letter", date="2025-10-06",
        body=(
            "The quarterly investor letter covers the Series B close, the "
            "runway management guides to, and early revenue, which the "
            "company attributes almost entirely to consumable cartridges."
        ),
        facts=(
            ("F-SEL-201", "Selene raised a $34M Series B funding round in September 2025.", "q2"),
            ("F-SEL-202", "Management guided to 24 months of runway at the current burn rate.", "q2"),
            ("F-SEL-203", "Revenue reached $6.2M in 2025, almost entirely from consumable cartridges.", "q2"),
        ),
    ),
    _doc(
        slug="selene-news-distribution", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="news",
        title="Selene Signs European Distribution Deals",
        source="Trade press article", date="2025-11-18",
        body=(
            "Trade coverage of Selene's European push described growing "
            "adoption through new distribution agreements — and noted a "
            "rival panel entering the same segment."
        ),
        facts=(
            ("F-SEL-301", "Adoption grew as distribution agreements extended Selene's market reach to 14 European countries.", "q3"),
            ("F-SEL-302", "A competitor launched a rival respiratory panel in late 2025.", "q3"),
        ),
    ),
    _doc(
        slug="selene-news-adoption", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="news",
        title="Hospital Groups Prepare for Winter Surge",
        source="Health-sector newsletter", date="2026-01-12",
        body=(
            "A health-sector newsletter on winter surge planning "
            "described hospital groups adopting rapid panels, Selene's "
            "among them, for seasonal capacity."
        ),
        facts=(
            ("F-SEL-303", "Hospital groups in two regions adopted the panel for winter surge capacity.", "q3"),
        ),
    ),
    _doc(
        slug="selene-field-evaluation", company="selene-diagnostics",
        company_name="Selene Diagnostics", kind="news",
        title="Independent Field Evaluation — Selene Panel",
        source="Independent evaluation summary", date="2026-03-03",
        body=(
            "An independent laboratory network evaluated the Selene panel "
            "under field conditions. Its measured result time was slower "
            "than the figure in the company's product brief."
        ),
        facts=(
            ("F-SEL-104", "An independent evaluation of the product measured a median result time of 41 minutes.", "q1"),
        ),
    ),
    # ------------------------------------------------------------------
    # Aster Foundry — industrial castings (citation-integrity case).
    # The corpus holds a near-duplicate of the annual report: an
    # earlier preliminary release whose revenue figure is stale. The
    # current filing says $186M; the preliminary release says $171M.
    # ------------------------------------------------------------------
    _doc(
        slug="aster-annual-report-fy2025", company="aster-foundry",
        company_name="Aster Foundry", kind="filing",
        title="Aster Foundry — Annual Report FY2025",
        source="Company filing (annual report)", date="2026-02-10",
        body=(
            "Aster Foundry, an Ohio-based maker of industrial castings "
            "and induction furnaces, closed FY2025 with double-digit "
            "growth as automotive programmes ramped. Management pointed "
            "to the audited annual figures as the record of the year; "
            "the preliminary release issued in January was superseded "
            "by this report after audit adjustments."
        ),
        facts=(
            ("F-AST-101", "Revenue grew 11% year over year to $186M in FY2025.", "q1"),
            ("F-AST-102", "Gross margin was 24% in FY2025, up from 21% in FY2024.", "q1"),
            ("F-AST-103", "Net debt stood at $22M at the end of FY2025.", "q1"),
        ),
    ),
    _doc(
        slug="aster-preliminary-results-fy2025", company="aster-foundry",
        company_name="Aster Foundry", kind="news",
        title="Aster Foundry — Preliminary FY2025 Results",
        source="Company preliminary release (unaudited)", date="2026-01-15",
        body=(
            "Aster Foundry issued unaudited preliminary results for "
            "FY2025 ahead of its annual report. The release carried the "
            "same narrative as the later filing — double-digit growth "
            "on automotive programme ramps — but its revenue figure was "
            "preliminary and was revised when the audited report "
            "appeared the following month."
        ),
        facts=(
            ("F-AST-901", "Preliminary results put Aster's FY2025 revenue at $171M, before audit adjustments.", "q1"),
        ),
    ),
    _doc(
        slug="aster-product-overview", company="aster-foundry",
        company_name="Aster Foundry", kind="product",
        title="Aster Foundry — Product and Operations Overview",
        source="Vendor product documentation", date="2025-11-20",
        body=(
            "The overview covers Aster's induction furnace line, the "
            "casting plants it runs in the Midwest, and the quality "
            "certifications its automotive customers require."
        ),
        facts=(
            ("F-AST-201", "The AF-40 is Aster's flagship induction furnace product platform for foundry customers.", "q2"),
            ("F-AST-202", "Aster operates 3 casting facilities in Ohio and Pennsylvania.", "q2"),
            ("F-AST-203", "The AF-40 carries ISO 9001 certification for automotive supply chains.", "q2"),
        ),
    ),
    # ------------------------------------------------------------------
    # Distractor: a similarly named but different company. Never in
    # scope for a gold case; its facts are nobody's gold.
    # ------------------------------------------------------------------
    _doc(
        slug="helvion-services-annual-note", company="helvion-services",
        company_name="Helvion Grid Services", kind="filing",
        title="Helvion Grid Services — Annual Note FY2025",
        source="Company filing (annual report)", date="2026-02-20",
        body=(
            "Helvion Grid Services is a separate business — a solar "
            "operations and maintenance contractor — that shares only a "
            "name stem with Helvion Grid Systems."
        ),
        facts=(
            ("F-DIS-101", "Revenue was $96M in FY2025 for the services business.", "q1"),
        ),
    ),
)


def docs_for_subject(subject_ref: str) -> list[CorpusDoc]:
    return [d for d in CORPUS if d.company == subject_ref]


def companies() -> list[dict]:
    seen: dict[str, str] = {}
    for d in CORPUS:
        seen.setdefault(d.company, d.company_name)
    return [
        {"subject_ref": ref, "subject": name}
        for ref, name in sorted(seen.items())
    ]


def _gold_facts(subject_ref: str) -> dict[str, str]:
    gold: dict[str, str] = {}
    for doc in docs_for_subject(subject_ref):
        for fact in doc.facts:
            gold[fact.id] = fact.question
    return gold


# The eval split, fixed here in the eval data — never chosen per
# run. The TUNING cases are the ones a profile author iterates
# against; the HELD-OUT cases are the ones the profile regression
# gate (promotion.py) judges on, so a profile tuned to the tuning
# cases still has to earn its promotion on cases it was not tuned
# against. The integrity pack is held out: its stale-figure trap and
# poisoned findings are exactly the behaviour a promotion must not
# be able to rehearse. Every pack sits in exactly one subset.
GOLD_CASES: tuple[dict, ...] = (
    {
        "key": "gold-helvion-vendor",
        "split": "tuning",
        "brief": {
            "subject": "Helvion Grid Systems",
            "subject_ref": "helvion-grid",
            "analyst": "Eval Harness",
            "questions": [
                {"id": "q1", "text": "What is Helvion's financial health: revenue, margins, cash flow, and debt?", "workstream": "financial-scan"},
                {"id": "q2", "text": "What does Helvion sell: product capability, deployments, certification, and commercial terms?", "workstream": "product-scan"},
                {"id": "q3", "text": "What is Helvion's market position: share, contract wins, and expansion?", "workstream": "market-scan"},
            ],
            "allowed_sources": [],
            "budget": {"max_steps": 120, "max_subagents": 6, "max_tokens": 400000, "max_wall_seconds": 600},
        },
        "gold_facts": _gold_facts("helvion-grid"),
        "contradictions": [["F-HEL-101", "F-HEL-106"]],
    },
    {
        "key": "gold-copperline-acquisition",
        "split": "tuning",
        "brief": {
            "subject": "Copperline Provisions",
            "subject_ref": "copperline-provisions",
            "analyst": "Eval Harness",
            "questions": [
                {"id": "q1", "text": "What is Copperline's financial health: revenue, profitability, debt, and cash?", "workstream": "financial-scan"},
                {"id": "q2", "text": "What is the scale of Copperline's operation: facilities, fleet, product range, and headcount?", "workstream": "product-scan"},
                {"id": "q3", "text": "What is Copperline's market position and competitive pressure?", "workstream": "market-scan"},
            ],
            "allowed_sources": [],
            "budget": {"max_steps": 120, "max_subagents": 6, "max_tokens": 400000, "max_wall_seconds": 600},
        },
        "gold_facts": _gold_facts("copperline-provisions"),
        "contradictions": [["F-COP-204", "F-COP-304"]],
    },
    {
        "key": "gold-selene-partnership",
        "split": "held_out",
        "brief": {
            "subject": "Selene Diagnostics",
            "subject_ref": "selene-diagnostics",
            "analyst": "Eval Harness",
            "questions": [
                {"id": "q1", "text": "What is Selene's product and its regulatory status?", "workstream": "product-scan"},
                {"id": "q2", "text": "What is Selene's financial position: funding, runway, and revenue?", "workstream": "financial-scan"},
                {"id": "q3", "text": "How is Selene's product being adopted in the market?", "workstream": "market-scan"},
            ],
            "allowed_sources": [],
            "budget": {"max_steps": 120, "max_subagents": 6, "max_tokens": 400000, "max_wall_seconds": 600},
        },
        "gold_facts": _gold_facts("selene-diagnostics"),
        "contradictions": [["F-SEL-101", "F-SEL-104"]],
    },
)


# Citation-integrity scenarios: gold packs whose grading goes beyond
# recall and citation presence to citation *correctness*. Aster's
# corpus contains a near-duplicate preliminary release with a stale
# revenue figure ($171M against the current filing's $186M). The stale
# fact (F-AST-901) is deliberately NOT gold: the pack passes only if
# the current figure's finding cites the current document, and — as in
# every pack — every figure a claim states appears in the evidence it
# cites (checked in evals/run_evals.py).
#
# The pack also carries poisoned findings: findings injected into the
# case record before execution, standing in for a sub-agent (or a live
# model) overstating its evidence. One overstates the filing's revenue
# figure; the other names an entity its cited document never mentions.
# Gold: the verifier quarantines both, with the expected reasons, and
# neither appears in the final report.
INTEGRITY_CASES: tuple[dict, ...] = (
    {
        "key": "integrity-aster-citation",
        "split": "held_out",
        "brief": {
            "subject": "Aster Foundry",
            "subject_ref": "aster-foundry",
            "analyst": "Eval Harness",
            "questions": [
                {"id": "q1", "text": "What is Aster's financial health: revenue, margins, and debt?", "workstream": "financial-scan"},
                {"id": "q2", "text": "What does Aster make and where does it operate: products, facilities, and certification?", "workstream": "product-scan"},
            ],
            "allowed_sources": [],
            "budget": {"max_steps": 120, "max_subagents": 6, "max_tokens": 400000, "max_wall_seconds": 600},
        },
        "gold_facts": {
            "F-AST-101": "q1",
            "F-AST-102": "q1",
            "F-AST-103": "q1",
            "F-AST-201": "q2",
            "F-AST-202": "q2",
            "F-AST-203": "q2",
        },
        "contradictions": [["F-AST-101", "F-AST-901"]],
        "citation_checks": [
            {"fact_ref": "F-AST-101", "doc_slug": "aster-annual-report-fy2025"},
        ],
        "poisoned_findings": [
            {
                "question_id": "q1",
                "claim": (
                    "Revenue grew 11% year over year to $198M in FY2025."
                ),
                "doc_slug": "aster-annual-report-fy2025",
                "expect_reason": "figure_not_in_evidence",
            },
            {
                "question_id": "q2",
                "claim": (
                    "Aster Foundry supplies furnace castings to Helvion "
                    "Grid Services under a long-term supply agreement."
                ),
                "doc_slug": "aster-product-overview",
                "expect_reason": "entity_mismatch",
            },
        ],
    },
)


ALL_CASES: tuple[dict, ...] = GOLD_CASES + INTEGRITY_CASES

# Case key -> eval subset ("tuning" | "held_out"), read off the case
# records themselves. The promotion gate consumes this map; it never
# re-decides which cases are held out.
PACK_SPLITS: dict[str, str] = {case["key"]: case["split"] for case in ALL_CASES}
