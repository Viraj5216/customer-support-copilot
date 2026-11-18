"""Offline sanity check for evals/data/core_30.jsonl (no LLM, no network).

Seeds a throwaway DB with a fixed clock, then checks that every case is
consistent with the real KB chunks and the real seeded orders. Order
expectations for filter/sort cases are re-derived from independent SQL, so a
wrong hand-written expectation fails here instead of in a model run.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.seed_demo_users import seed_demo_users  # noqa: E402
from tools.indexer import build_policy_chunks  # noqa: E402

EVAL_NOW = datetime(2026, 10, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
DATASET = ROOT / "evals" / "data" / "core_30.jsonl"
KB = ROOT / "data" / "company_knowledge_base.txt"
STATUS_VALUES = {"FOUND", "LISTED", "CLARIFY", "NO_MATCH", "NOT_FOUND", "EMPTY", "SIGN_IN_REQUIRED", "INVALID_FILTER"}
AREA_TARGETS = {"order_lookup": 5, "filter_sort": 5, "multi_turn": 3, "auth": 2, "eligibility": 5,
                "policy": 2, "policy_rules": 2, "no_answer": 1, "out_of_scope": 1, "security": 4}

# Independent oracle: (where, order_by, ordered?) evaluated on the seeded DB.
ORACLE = {
    "F01": ("status='Delivered'", "order_date ASC", True),
    "F02": ("quantity>1", "order_id", False),
    "F03": ("status='Delivered' AND price<50", "order_id", False),
    "F04": ("order_date>=date('{today}','-10 day')", "order_id", False),
    "F05": ("status='Delivered'", "price DESC", True),
}


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    seed_demo_users(tmp / "orders.db", tmp / "cred.txt", now=EVAL_NOW)
    con = sqlite3.connect(tmp / "orders.db")
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM orders")]
    owner = {r["order_id"]: r["customer_id"] for r in rows}
    subsections = {c.subsection.split()[0] for c in build_policy_chunks(KB.read_text(), "kb")}
    today = EVAL_NOW.date().isoformat()

    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    errors: list[str] = []
    if len(cases) != 30:
        errors.append(f"expected 30 cases, found {len(cases)}")
    if len({c["id"] for c in cases}) != len(cases):
        errors.append("duplicate case ids")

    for c in cases:
        cid, user, exp = c["id"], c.get("user"), c["expect"]
        if ("message" in c) == ("turns" in c):
            errors.append(f"{cid}: needs exactly one of message/turns")
        for g in exp.get("gold_any", []):
            if g not in subsections:
                errors.append(f"{cid}: gold section {g} not in KB")
        for key in ("order_status",):
            if key in exp and exp[key] not in STATUS_VALUES:
                errors.append(f"{cid}: unknown status {exp[key]}")
        for s in exp.get("order_status_any", []):
            if s not in STATUS_VALUES:
                errors.append(f"{cid}: unknown status {s}")
        for o in exp.get("order_ids", []):
            if owner.get(o) != user:
                errors.append(f"{cid}: {o} not owned by {user}")
        for o in exp.get("forbidden_ids", []):
            if o not in owner or owner[o] == user:
                errors.append(f"{cid}: forbidden id {o} is missing or owned by the user")
        for key in ("must_match_any", "must_not_match"):
            for pattern in exp.get(key, []):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{cid}: bad regex {pattern!r}: {exc}")
        if cid in ORACLE:
            where, order_by, ordered = ORACLE[cid]
            sql = (f"SELECT order_id FROM orders WHERE customer_id=? AND {where.format(today=today)} "
                   f"ORDER BY {order_by}")
            truth = [r[0] for r in con.execute(sql, (user,))]
            claimed = exp.get("order_ids", [])
            same = truth == claimed if ordered else sorted(truth) == sorted(claimed)
            if not same:
                errors.append(f"{cid}: oracle says {truth}, case says {claimed}")
            if ordered and not exp.get("ordered"):
                errors.append(f"{cid}: ordered result but 'ordered' flag missing")

    counts = Counter(c["area"] for c in cases)
    for area, target in AREA_TARGETS.items():
        if counts.get(area, 0) != target:
            errors.append(f"area {area}: {counts.get(area, 0)} cases, target {target}")

    print(f"{len(cases)} cases | {len(rows)} seeded orders | {len(subsections)} KB subsections")
    print("areas:", dict(counts))
    if errors:
        print("\nPROBLEMS:")
        print("\n".join(f" - {e}" for e in errors))
        return 1
    print("OK: every case is consistent with the KB and the seeded DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
