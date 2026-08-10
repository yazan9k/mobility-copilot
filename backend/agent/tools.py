"""Tool definitions and implementations.

Structure:
  TOOL_DESCRIPTIONS  — versioned. Phase 3 rewrites these; keeping them in one
                       block makes the change legible as a diff and lets the
                       eval attribute score movement to it.
  _SCHEMAS           — parameter schemas, stable across versions.
  IMPLEMENTATIONS    — the Python functions.
  build_tool_schemas / dispatch — what agent/core.py uses.

Every implementation returns a ToolResult so the agent gets a string and the
eval harness gets structured metadata (notably which policy docs were
retrieved, which is scored deterministically).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from config import AGENT_VERSION, DB_PATH, VISA_DATA_PATH
from rag import retrieve


@dataclass
class ToolResult:
    display: str  # returned to the model
    meta: dict[str, Any] = field(default_factory=dict)  # kept for evaluation


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

COUNTRY_TIERS = {
    "netherlands": 1, "germany": 1, "ireland": 1, "canada": 1, "australia": 1,
    "singapore": 2, "united arab emirates": 2, "uae": 2, "japan": 2,
    "switzerland": 2, "united kingdom": 2, "uk": 2, "united states": 2, "usa": 2,
    "india": 3, "brazil": 3, "poland": 3, "mexico": 3, "south africa": 3,
}

TIER_TIMELINE_WEEKS = {
    1: {"total": "8-12", "documents": "2-3", "filing": "4-6", "predeparture": "2-3",
        "registration": "1-2", "temp_accommodation_days": 30},
    2: {"total": "12-16", "documents": "3-4", "filing": "6-10", "predeparture": "2-3",
        "registration": "2-3", "temp_accommodation_days": 45},
    3: {"total": "18-26", "documents": "6-10", "filing": "10-16", "predeparture": "2-3",
        "registration": "3-5", "temp_accommodation_days": 60},
}

EMPLOYEE_TYPES = {
    "short_term_assignment", "long_term_assignment", "permanent_transfer",
}


def _tier_for(country: str) -> int | None:
    return COUNTRY_TIERS.get(country.strip().lower())


@lru_cache(maxsize=1)
def _visa_data() -> dict:
    return json.loads(VISA_DATA_PATH.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    aliases = {
        "uae": "united arab emirates", "dubai": "united arab emirates",
        "abu dhabi": "united arab emirates", "uk": "united kingdom",
        "britain": "united kingdom", "england": "united kingdom",
        "london": "united kingdom", "usa": "united states", "us": "united states",
        "america": "united states", "san francisco": "united states",
        "new york": "united states", "bangalore": "india", "mumbai": "india",
        "delhi": "india", "amsterdam": "netherlands", "holland": "netherlands",
        "zurich": "switzerland", "geneva": "switzerland", "tokyo": "japan",
        "toronto": "canada", "sydney": "australia", "dublin": "ireland",
        "berlin": "germany", "munich": "germany", "warsaw": "poland",
        "mexico city": "mexico", "sao paulo": "brazil", "cape town": "south africa",
        "johannesburg": "south africa",
    }
    key = s.strip().lower()
    return aliases.get(key, key)


# ---------------------------------------------------------------------------
# Tool descriptions — VERSIONED (Phase 3 lever)
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # v1: a plausible first pass. Each says what the tool does. None of them
    # says when to use it in preference to another, which is the gap Phase 3
    # is expected to expose.
    "v1": {
        "search_policy_kb":
            "Search the company relocation policy knowledge base.",
        "lookup_visa_requirements":
            "Look up visa requirements for a country pair.",
        "generate_document_checklist":
            "Generate a checklist of documents needed for a relocation.",
        "get_relocation_timeline":
            "Get the relocation timeline for a destination.",
        "create_hr_ticket":
            "Create a ticket for the HR team.",
        "escalate_to_human":
            "Escalate the question to a human HR contact.",
    },

    # v2: each description now says WHEN to call and when not to. Finding F7
    # showed v1 descriptions stated capability without scope, and the model
    # reached for whichever tool produced an artefact — escalate_to_human was
    # called 4 times against 20 expected, while generate_document_checklist
    # fired 3.7x more often than warranted. Change C2.
    "v2": {
        "search_policy_kb":
            "Search Meridian's relocation policy documents. Use this for any question "
            "about what a policy says or what an employee is entitled to: allowances, "
            "housing, shipping, tax equalization, family support, escalation rules, or "
            "what happens at the end of an assignment. Prefer this over answering from "
            "memory — entitlements vary by assignment type, band, and destination tier, "
            "and guessing produces confidently wrong figures. Use it alongside other "
            "tools when a question has both a policy and a data component.",

        "lookup_visa_requirements":
            "Look up permit details for one origin/destination country pair from "
            "Meridian's visa dataset: permit name, processing time, maximum stay, "
            "whether local payroll is permitted, and required documents. Use this only "
            "when the employee names both countries and you need permit specifics. It "
            "does NOT cover what the company pays for, employee entitlements, or "
            "whether a dependent may work — those are policy questions. If the country "
            "pair is not in the dataset, say so rather than inferring from a similar one.",

        "generate_document_checklist":
            "Produce a checklist of documents required for a relocation, given a "
            "destination and assignment type. Call this only when the employee is "
            "actually asking what documents they need to gather. Do NOT call it for "
            "questions about timelines, allowances, permits, eligibility, or renewals — "
            "a checklist is not a general-purpose answer. For a renewal rather than a "
            "first application, search the policy documents instead, because renewals "
            "require reissued documents that this checklist does not reflect.",

        "get_relocation_timeline":
            "Return the milestone plan and expected durations for relocating to a given "
            "destination, by tier. Use this when the employee asks how long something "
            "will take, when to start, or what happens in what order. Not needed for "
            "questions about entitlements or documents.",

        "create_hr_ticket":
            "Open a ticket asking the HR team to action something specific for the "
            "employee. Use this only when there is a concrete request to be carried out. "
            "This is NOT the way to route a question you are not permitted to answer — "
            "for anything in the five mandatory escalation categories, call "
            "escalate_to_human instead.",

        "escalate_to_human":
            "Hand the question to a Global Mobility adviser. CALL THIS TOOL — do not "
            "simply advise the employee to contact an adviser, and do not ask their "
            "permission first. You must call it whenever the question involves: (1) an "
            "individual's tax position or a tax decision; (2) a visa refusal, appeal, or "
            "a permit that has expired or is inside the 90-day renewal window; (3) "
            "whether a spouse or dependent may WORK in the destination (their right to "
            "RESIDE is answerable normally); (4) a request for an exception, variation, "
            "or discretionary approval; (5) another employee's package or personal "
            "circumstances. Where a message mixes a restricted question with an "
            "answerable one, answer the answerable part and escalate the rest.",
    },
}

# v3 reuses v2's tool descriptions verbatim. They are not implicated in the v2
# regression — escalation recall tripled under them — so holding them fixed keeps
# the v2 -> v3 comparison attributable to the system prompt alone.
TOOL_DESCRIPTIONS["v3"] = TOOL_DESCRIPTIONS["v2"]

# The three v4 arms differ only in their system prompt's escalation block. Holding
# tool descriptions identical across all of them is what makes that comparison
# clean — otherwise a score difference could be coming from either lever.
for _arm in ("v4-principled", "v4-enumerated", "v4-verbatim"):
    TOOL_DESCRIPTIONS[_arm] = TOOL_DESCRIPTIONS["v2"]


def _descriptions(version: str | None = None) -> dict[str, str]:
    """Descriptions for a given agent version.

    Takes an explicit version rather than reading config at import time: the eval
    runner drives versions per-run, and binding this to the module-level constant
    meant `--version v2` swapped the prompt while silently keeping v1's tool
    descriptions — which would have voided change C2 without any visible error.
    """
    v = version or AGENT_VERSION
    try:
        return TOOL_DESCRIPTIONS[v]
    except KeyError:
        raise ValueError(
            f"No tool descriptions for version {v!r}. Known: {sorted(TOOL_DESCRIPTIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# Parameter schemas — stable across versions
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_policy_kb": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The policy topic to search for.",
            }
        },
        "required": ["query"],
    },
    "lookup_visa_requirements": {
        "type": "object",
        "properties": {
            "from_country": {"type": "string", "description": "Origin country."},
            "to_country": {"type": "string", "description": "Destination country."},
            "visa_type": {
                "type": "string",
                "enum": ["short_term_business", "work_permit",
                         "dependent_residence", "renewal"],
                "description": "Which permit route to look up.",
            },
        },
        "required": ["from_country", "to_country", "visa_type"],
    },
    "generate_document_checklist": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination country."},
            "employee_type": {
                "type": "string",
                "enum": sorted(EMPLOYEE_TYPES),
                "description": "The assignment type.",
            },
        },
        "required": ["destination", "employee_type"],
    },
    "get_relocation_timeline": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination country."}
        },
        "required": ["destination"],
    },
    "create_hr_ticket": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["visa", "housing", "shipping", "tax", "family", "other"],
                "description": "Ticket category.",
            },
            "summary": {"type": "string", "description": "What the employee needs."},
            "urgency": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "How urgent this is.",
            },
        },
        "required": ["category", "summary", "urgency"],
    },
    "escalate_to_human": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why this needs a human.",
            }
        },
        "required": ["reason"],
    },
}


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def search_policy_kb(query: str) -> ToolResult:
    chunks = retrieve.search(query)
    return ToolResult(
        display=retrieve.format_context(chunks),
        meta={
            "retrieved_docs": [c.source_doc for c in chunks],
            "retrieved_sections": [c.section for c in chunks],
            "distances": [round(c.distance, 4) for c in chunks],
        },
    )


def lookup_visa_requirements(
    from_country: str, to_country: str, visa_type: str
) -> ToolResult:
    origin, dest = _norm(from_country), _norm(to_country)
    data = _visa_data()

    match = next(
        (
            p for p in data["pairs"]
            if _norm(p["from_country"]) == origin and _norm(p["to_country"]) == dest
        ),
        None,
    )
    if match is None:
        known = sorted({f"{p['from_country']} -> {p['to_country']}" for p in data["pairs"]})
        return ToolResult(
            display=(
                f"No visa record for {from_country} -> {to_country}. This country pair "
                f"is not in the mobility dataset, so the requirements cannot be "
                f"confirmed here and a Global Mobility adviser should be asked.\n"
                f"Pairs on file: {'; '.join(known)}"
            ),
            meta={"found": False, "from": from_country, "to": to_country},
        )

    route = match["routes"].get(visa_type)
    if route is None:
        return ToolResult(
            display=(
                f"No {visa_type!r} route for {from_country} -> {to_country}. "
                f"Available: {', '.join(match['routes'])}."
            ),
            meta={"found": False, "visa_type": visa_type},
        )

    lo, hi = route["processing_weeks"]
    lines = [
        f"SYNTHETIC visa data — {from_country} -> {to_country} "
        f"(destination tier {match['destination_tier']}), route {visa_type!r}:",
        f"  Permit:            {route['permit_name']}",
        f"  Max stay:          {route['max_stay_months']} months",
        f"  Processing:        {lo}-{hi} weeks",
        f"  Company sponsors:  {'yes' if route['sponsored_by_company'] else 'no'}",
    ]
    if "payroll_transfer_permitted" in route:
        lines.append(
            f"  Local payroll:     "
            f"{'permitted' if route['payroll_transfer_permitted'] else 'NOT permitted'}"
        )
    if "convertible_in_country" in route:
        lines.append(
            f"  Convertible:       "
            f"{'yes' if route['convertible_in_country'] else 'no — requires a fresh application'}"
        )
    if route.get("filed_after_employee_permit"):
        lines.append("  Sequencing:        cannot be filed until the employee permit issues")
    if "carries_work_rights" in route:
        lines.append(
            f"  Work rights:       "
            f"{'yes' if route['carries_work_rights'] else 'NO — work authorisation is a separate matter'}"
        )
    if route.get("initiate_days_before_expiry"):
        lines.append(
            f"  Start renewal:     at least "
            f"{route['initiate_days_before_expiry']} days before expiry"
        )
    lines.append("  Requirements:      " + "; ".join(route["key_requirements"]))
    lines.append(f"  Notes:             {route['notes']}")

    return ToolResult(
        display="\n".join(lines),
        meta={
            "found": True,
            "tier": match["destination_tier"],
            "permit_name": route["permit_name"],
            "visa_type": visa_type,
        },
    )


def generate_document_checklist(destination: str, employee_type: str) -> ToolResult:
    tier = _tier_for(_norm(destination))
    if tier is None:
        return ToolResult(
            display=(
                f"{destination!r} is not a Meridian destination on file, so a tier "
                f"cannot be determined. A Global Mobility adviser should confirm the "
                f"requirements."
            ),
            meta={"found": False},
        )

    etype = employee_type.strip().lower()
    if etype not in EMPLOYEE_TYPES:
        return ToolResult(
            display=(
                f"Unknown assignment type {employee_type!r}. Expected one of: "
                f"{', '.join(sorted(EMPLOYEE_TYPES))}."
            ),
            meta={"found": False},
        )

    items = [
        "Passport valid at least 6 months beyond the intended permit end date, 2+ blank pages",
        "Passport-standard photographs to the destination specification",
        "Signed employment contract or assignment letter from the destination entity",
        "Evidence of qualifications where the role requires them",
        "Completed application form (supplied by Global Mobility or counsel)",
    ]
    if tier >= 2:
        items += [
            "Police certificate from every country resided in 6+ months in the last 5 years "
            "(validity window: typically 3 months at filing)",
            "Educational certificates verified or attested by the issuing institution",
        ]
    if tier == 3:
        items += [
            "Medical examination at a destination-designated panel clinic (valid ~3 months)",
            "Birth certificate (recent extract)",
            "Marriage certificate where applicable (recent extract)",
            "Legalisation rather than apostille — allow 3-8 weeks",
        ]

    if etype == "short_term_assignment":
        items.append(
            "NOTE: short-term assignments under 6 months typically use a business or "
            "short-term route with a lighter document set; confirm the route before collecting"
        )
    else:
        items.append(
            "Proof of address in the home country"
        )

    header = (
        f"Document checklist — destination {destination} (tier {tier}), "
        f"assignment type {etype}:"
    )
    body = "\n".join(f"  [ ] {item}" for item in items)
    footer = (
        "\nDocuments with validity windows cannot be obtained far in advance and held. "
        "For a renewal rather than an initial application, items subject to a validity "
        "window must be reissued."
    )
    return ToolResult(
        display=f"{header}\n{body}\n{footer}",
        meta={"found": True, "tier": tier, "employee_type": etype, "item_count": len(items)},
    )


def get_relocation_timeline(destination: str) -> ToolResult:
    tier = _tier_for(_norm(destination))
    if tier is None:
        return ToolResult(
            display=(
                f"{destination!r} is not a Meridian destination on file, so a timeline "
                f"cannot be given. A Global Mobility adviser should confirm."
            ),
            meta={"found": False},
        )

    t = TIER_TIMELINE_WEEKS[tier]
    display = "\n".join([
        f"Relocation timeline — {destination} (tier {tier}). "
        f"Total from assignment approval to first working day: {t['total']} weeks.",
        f"  1. Assignment approval    entitlements are fixed at this point",
        f"  2. Document collection    {t['documents']} weeks",
        f"  3. Immigration filing     {t['filing']} weeks (outside Meridian's control)",
        f"  4. Pre-departure          {t['predeparture']} weeks (do not book travel before the permit issues)",
        f"  5. Arrival & registration {t['registration']} weeks",
        f"  6. Settling in            temporary accommodation covers {t['temp_accommodation_days']} days",
        "",
        "Dependent applications often cannot be filed until the employee permit issues, "
        "adding 3-6 weeks. Household shipment customs clearance generally requires the "
        "residence permit, so a permit delay also delays the shipment.",
    ])
    return ToolResult(display=display, meta={"found": True, "tier": tier})


def create_hr_ticket(category: str, summary: str, urgency: str) -> ToolResult:
    ticket_id = f"GM-{uuid.uuid4().hex[:8].upper()}"
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO hr_tickets (ticket_id, category, summary, urgency) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, category, summary, urgency),
        )
        conn.commit()
    finally:
        conn.close()

    return ToolResult(
        display=(
            f"Ticket {ticket_id} created ({category}, urgency {urgency}). "
            f"The Global Mobility team acknowledges within one business day, or the "
            f"same business day for time-critical cases."
        ),
        meta={"ticket_id": ticket_id, "category": category, "urgency": urgency},
    )


def escalate_to_human(reason: str) -> ToolResult:
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO hr_tickets (ticket_id, category, summary, urgency) "
            "VALUES (?, ?, ?, ?)",
            (ticket_id, "escalation", reason, "high"),
        )
        conn.commit()
    finally:
        conn.close()

    return ToolResult(
        display=(
            f"Escalated to a Global Mobility adviser (reference {ticket_id}). "
            f"Reason recorded: {reason}"
        ),
        meta={"ticket_id": ticket_id, "reason": reason},
    )


IMPLEMENTATIONS: dict[str, Callable[..., ToolResult]] = {
    "search_policy_kb": search_policy_kb,
    "lookup_visa_requirements": lookup_visa_requirements,
    "generate_document_checklist": generate_document_checklist,
    "get_relocation_timeline": get_relocation_timeline,
    "create_hr_ticket": create_hr_ticket,
    "escalate_to_human": escalate_to_human,
}

TOOL_NAMES = tuple(IMPLEMENTATIONS)


def build_tool_schemas(version: str | None = None) -> list[dict[str, Any]]:
    """Tool definitions in the function-calling format, for the given version."""
    descriptions = _descriptions(version)
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": _SCHEMAS[name],
            },
        }
        for name in TOOL_NAMES
    ]


def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Execute a tool call, converting failures into a result the model can read."""
    impl = IMPLEMENTATIONS.get(name)
    if impl is None:
        return ToolResult(
            display=f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}.",
            meta={"error": "unknown_tool"},
        )
    try:
        return impl(**arguments)
    except TypeError as exc:
        return ToolResult(
            display=f"Invalid arguments for {name}: {exc}",
            meta={"error": "bad_arguments", "detail": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001 - the model should see the failure
        return ToolResult(
            display=f"Tool {name} failed: {exc}",
            meta={"error": "exception", "detail": str(exc)},
        )
