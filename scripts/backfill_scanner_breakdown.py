#!/usr/bin/env python3
"""Backfill scans_cv2 / scans_zxing / scans_qreader for existing candidates.

Migration 009 added these columns as NULL for pre-existing rows. This
script walks every candidate where scans_cv2 IS NULL, opens the on-disk
PNG, runs all three scanners independently, and writes the booleans back.

Idempotent — re-running is a no-op for rows already populated. Safe to
run while the server is up (write lock is acquired per-row).

Run with the project's venv:
  venv/bin/python scripts/backfill_scanner_breakdown.py [options]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from qrart.scanner import scan_breakdown  # noqa: E402

DB_PATH = ROOT / "qrart.db"


def url_to_fs(url_path: str) -> Path:
    if url_path.startswith("/outputs/"):
        return ROOT / "outputs" / url_path[len("/outputs/"):]
    return Path(url_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap how many rows to backfill (default: unlimited)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be updated without writing")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find candidates that need backfill.
    query = (
        "SELECT c.id, c.image_path, j.data AS target "
        "FROM candidates c JOIN jobs j ON c.job_id = j.id "
        "WHERE c.scans_cv2 IS NULL "
        "ORDER BY c.id"
    )
    if args.limit:
        query += f" LIMIT {int(args.limit)}"
    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} candidate(s) needing scanner backfill")
    if not rows:
        return 0

    t0 = time.time()
    counts = {"universal": 0, "phone": 0, "ios": 0, "soft": 0, "none": 0, "missing": 0}
    for i, row in enumerate(rows, 1):
        fs = url_to_fs(row["image_path"])
        if not fs.exists():
            print(f"  [{i:>4}/{len(rows)}] {row['id'][:8]}: SKIP (file missing: {fs})")
            counts["missing"] += 1
            continue
        try:
            img = Image.open(fs)
            br = scan_breakdown(img)
        except Exception as e:
            print(f"  [{i:>4}/{len(rows)}] {row['id'][:8]}: SKIP ({type(e).__name__}: {e})")
            counts["missing"] += 1
            continue
        target = row["target"]
        cv2_ok = br["cv2"] == target
        zx_ok = br["zxing"] == target
        qr_ok = br["qreader"] == target

        # Bucket for the per-run summary.
        if cv2_ok and zx_ok and qr_ok:
            counts["universal"] += 1; tag = "🟢 universal"
        elif cv2_ok and zx_ok:
            counts["phone"] += 1; tag = "🟡 phone-ready"
        elif zx_ok or cv2_ok:
            counts["ios"] += 1; tag = "🟠 ios-class"
        elif qr_ok:
            counts["soft"] += 1; tag = "🔴 soft"
        else:
            counts["none"] += 1; tag = "⚫ none"

        if args.dry_run:
            print(f"  [{i:>4}/{len(rows)}] {row['id'][:8]}: {tag}")
            continue

        conn.execute(
            "UPDATE candidates SET scans_cv2=?, scans_zxing=?, scans_qreader=? WHERE id=?",
            (int(cv2_ok), int(zx_ok), int(qr_ok), row["id"]),
        )
        if i % 25 == 0:
            conn.commit()
            elapsed = round(time.time() - t0, 1)
            print(f"  [{i:>4}/{len(rows)}] checkpoint · {elapsed}s elapsed", flush=True)

    if not args.dry_run:
        conn.commit()
    conn.close()

    print()
    print("Summary:")
    print(f"  🟢 universal:   {counts['universal']}")
    print(f"  🟡 phone-ready: {counts['phone']}")
    print(f"  🟠 ios-class:   {counts['ios']}")
    print(f"  🔴 soft:        {counts['soft']}")
    print(f"  ⚫ none:        {counts['none']}")
    print(f"  ⊘  missing/err: {counts['missing']}")
    print(f"  total in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
