"""System prompts, versioned.

v1 is deliberately naive. It is the baseline the project is trying to beat, and
tuning it before measuring would destroy the before/after comparison that is
this project's actual deliverable. It says what the assistant is and lists the
tools; it says nothing about when to escalate, how to handle a false premise,
or whether to cite sources.

v2 (Phase 3) is written only after the v1 failure analysis, and every addition
to it traces to a numbered finding.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "v1": (
        "You are the Global Mobility Copilot, an internal assistant at Meridian "
        "Systems. You help employees with questions about international relocation: "
        "visas, allowances, housing, shipping, timelines, and documents.\n\n"
        "You have tools available for searching the policy knowledge base, looking "
        "up visa requirements, generating document checklists, getting relocation "
        "timelines, creating HR tickets, and escalating to a human.\n\n"
        "Answer the employee's question helpfully."
    ),
}


def system_prompt(version: str) -> str:
    try:
        return SYSTEM_PROMPTS[version]
    except KeyError:
        raise ValueError(
            f"Unknown agent version {version!r}. Known: {sorted(SYSTEM_PROMPTS)}"
        ) from None
