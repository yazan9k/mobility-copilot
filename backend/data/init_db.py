"""Create and seed the mock HR database.

SYNTHETIC DATA. Every employee, ticket, and relocation record here is invented.

Run:  python -m data.init_db
"""

from __future__ import annotations

import sqlite3
import sys

from config import DB_PATH

SCHEMA = """
DROP TABLE IF EXISTS hr_tickets;
DROP TABLE IF EXISTS relocation_status;

CREATE TABLE hr_tickets (
    ticket_id   TEXT PRIMARY KEY,
    category    TEXT NOT NULL,
    summary     TEXT NOT NULL,
    urgency     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE relocation_status (
    employee_id      TEXT PRIMARY KEY,
    employee_name    TEXT NOT NULL,
    from_country     TEXT NOT NULL,
    to_country       TEXT NOT NULL,
    assignment_type  TEXT NOT NULL,
    band             INTEGER NOT NULL,
    dependents       INTEGER NOT NULL DEFAULT 0,
    current_stage    TEXT NOT NULL,
    permit_expiry    TEXT,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Seeded to match the personas in docs/personas.md so the golden set can
# reference concrete employee IDs.
SEED_RELOCATIONS = [
    ("EMP-1042", "Priya Raghavan", "India", "Netherlands",
     "permanent_transfer", 4, 3, "immigration_filing", None),
    ("EMP-2287", "Tom Okonkwo", "United Kingdom", "Singapore",
     "short_term_assignment", 3, 0, "pre_departure", None),
    ("EMP-3311", "Sofia Marchetti", "Italy", "United Arab Emirates",
     "long_term_assignment", 4, 0, "in_country_active", "2026-11-01"),
    ("EMP-4590", "David Chen", "United States", "Switzerland",
     "permanent_transfer", 6, 1, "document_collection", None),
    ("EMP-5123", "Amara Diallo", "South Africa", "Ireland",
     "long_term_assignment", 3, 2, "arrival_registration", "2027-04-15"),
]


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            """INSERT INTO relocation_status
               (employee_id, employee_name, from_country, to_country,
                assignment_type, band, dependents, current_stage, permit_expiry)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            SEED_RELOCATIONS,
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM relocation_status").fetchone()[0]
    conn.close()
    print(f"Initialised {DB_PATH}")
    print(f"  relocation_status: {n} seeded records")
    print("  hr_tickets:        empty (written to by the create_hr_ticket tool)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
