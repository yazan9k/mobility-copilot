# Policy Corpus Outline

> **All content described here is fabricated** for the fictional Meridian Systems. Nothing is scraped, quoted, or adapted from real company policies or government immigration sources. Fabricated data is deliberately signposted as such inside the documents themselves.

Ten documents, 300–800 words each, authored in Phase 1 into `backend/rag/corpus/`.

---

## Shared fabricated facts

Every document must agree on these. They are the spine of the corpus: the golden set asserts against them, the personas' failure modes depend on them, and any drift between documents will surface as a retrieval or faithfulness failure that is an artefact of sloppy authoring rather than a real agent defect.

### Destination country tiers

| Tier | Countries | Characteristics |
|---|---|---|
| **Tier 1** | Netherlands, Germany, Ireland, Canada, Australia | Established Meridian entity, in-house immigration support, fastest processing |
| **Tier 2** | Singapore, UAE, Japan, Switzerland, UK | Established entity, external immigration counsel, moderate processing |
| **Tier 3** | India, Brazil, Poland, Mexico, South Africa | Smaller entity or partner-of-record, longest processing, most document requirements |

### Employee bands

| Band | Level |
|---|---|
| 1–2 | Early career |
| 3–4 | Mid / senior individual contributor |
| 5–6 | Director and above |

### Assignment types

| Type | Duration | Key consequence |
|---|---|---|
| **Short-term assignment** | Under 6 months | Per-diem, no household shipping, no tax equalization |
| **Long-term assignment** | 6–24 months | Housing stipend, partial shipping, tax equalization applies |
| **Permanent transfer** | Indefinite | Full relocation allowance, full shipping, tax equalization for first 24 months |

### Load-bearing thresholds

These recur across documents and are the source of several intended failure modes:

- **The 6-month line.** Crossing it changes assignment type, visa class, tax treatment, and shipping entitlement simultaneously. (P2 Tom.)
- **90 days before permit expiry.** Renewal must be initiated by then. (P3 Sofia.)
- **24 months.** Tax equalization support ends. (P4 David.)
- **Band 5.** Senior allowance schedule and additional entitlements begin. (P4 David.)

### Human-escalation categories

Fixed list, stated identically in `08-escalation-criteria.md` and referenced from every other document. The agent must escalate on all five:

1. Individual tax advice or tax position questions
2. Visa refusals, appeals, or a lapsed/expired permit
3. Spousal or dependent **work** authorization (dependent *residence* is answerable)
4. Any request for a policy exception or discretionary approval
5. Questions about another employee's package or personal circumstances

---

## The ten documents

| # | File | Covers | Primary personas | Deliberate difficulty |
|---|---|---|---|---|
| 01 | `01-relocation-allowance.md` | Lump-sum relocation allowance by band and assignment type; what it may and may not be spent on; payment timing | P1, P4 | Band 5+ schedule differs — an agent quoting the standard table for a Band 6 employee is wrong |
| 02 | `02-visa-sponsorship.md` | Who Meridian sponsors, sponsorship by tier, employee vs. dependent permits, what sponsorship excludes | P1, P2, P3 | Employee permit and dependent permit are separate processes; easy to conflate |
| 03 | `03-housing-stipend.md` | Temporary accommodation, ongoing housing stipend by tier and band, duration limits, per-diem for short-term | P2, P3 | Short-term assignees get per-diem, **not** the housing stipend — directly targets P2's false premise |
| 04 | `04-shipping-and-moving.md` | Shipping allowance by assignment type and household size, air vs. sea freight, excluded items, insurance | P1, P2 | Short-term assignees get no household shipping; allowance scales with household size, not band |
| 05 | `05-tax-equalization.md` | What tax equalization is, who qualifies, the 24-month limit, what the company covers in tax advisory | P4 | Explains the policy while **refusing individual tax advice** — the document itself models the escalation boundary |
| 06 | `06-relocation-timeline.md` | Milestones from offer to first month in-country, per tier; typical durations; what blocks what | P1, P2, P3 | Tier 3 timelines are roughly double Tier 1; a tier-blind answer is wrong |
| 07 | `07-document-requirements.md` | Required documents by tier and employee type; dependent documents; apostille/legalization; validity windows | P1, P3 | Renewals need *reissued* documents, not the originals from the first application — targets P3 directly |
| 08 | `08-escalation-criteria.md` | The five escalation categories, why each exists, what the employee should expect after escalation | P4, P5 | Also states what is **not** an escalation, to give the agent grounds to answer rather than over-refuse |
| 09 | `09-dependent-and-family-support.md` | Dependent residence permits, school allowance by tier, spousal support services, healthcare registration | P1 | Spousal *work* authorization escalates; spousal *residence* is answerable. The document draws this line explicitly |
| 10 | `10-repatriation-and-assignment-end.md` | End-of-assignment process, return shipping, what happens when a short-term assignment extends past 6 months, resignation during assignment | P2, P3 | The extension-past-6-months path is the other half of P2's trap and lives here, not in the assignment-type doc — a genuine multi-document retrieval case |

---

## Authoring constraints

1. **Signpost the fabrication.** Each document opens with a one-line notice that it describes a fictional company.
2. **Cross-reference by filename.** When a document defers to another, name the file. This is what makes multi-document retrieval cases legitimate rather than accidental.
3. **State escalation inline.** Where a document touches an escalation category, it says so at that point rather than relying on the agent to remember doc 08.
4. **Use consistent headings.** `##`-level sections carry meaning; the ingester chunks on them, so a section is the unit of retrieval. Sections should be independently comprehensible.
5. **Bury the traps in prose, not in bold warnings.** The 6-month threshold and the renewal-reissue rule should read like normal policy text. Flagging them typographically would make retrieval artificially easy and inflate the baseline.
6. **Include one genuine near-duplicate pair.** Docs 03 and 04 both discuss short-term assignee entitlements from different angles. This creates a real retrieval-precision challenge rather than a synthetic one.
