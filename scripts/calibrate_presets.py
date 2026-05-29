#!/usr/bin/env python3
"""Calibrate every preset against a running QR Art Studio server.

Walks the PRESETS list, generates each one, and if no candidate scans,
bumps controlnet_scale (and qr_coverage on later attempts) until the
output reliably scans — or the preset is declared hopeless.

Writes results atomically to a JSON report. With --update-presets, also
appends a CALIBRATIONS overlay dict to qrart/presets.py so every future
load of the preset uses the calibrated values.

Assumes the server is already running. Default base-url targets the
loopback dev server. Use --fast-mode for a rapid sanity-check pass (LCM,
6 steps); use the default for a real calibration run.

Run with the project's venv:
  venv/bin/python scripts/calibrate_presets.py [options]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as _urllib_error
from urllib import request as _urllib_request

# Project root on sys.path so we can import qrart without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qrart.presets import PRESETS, PRESETS_BY_SLUG  # noqa: E402


# ── Tuning knobs (mirror what the server enforces) ────────────────────────────
ESCALATE_CAP = 1.90          # matches app.ESCALATE_CAP
SCORE_FLOOR = 0.55           # below this on attempt 1 → hopeless prompt
SCALE_BUMP = 0.05            # per failed attempt
COVERAGE_BUMP_AT_ATTEMPT = 3
COVERAGE_BUMP_AMOUNT = 0.05
COVERAGE_BUMP_CEILING = 0.85

# How long to wait per job before giving up.
JOB_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 2.5


# ── HTTP via stdlib (no `requests` dependency) ────────────────────────────────
def _http_request(method: str, url: str, body: dict | None = None, timeout: float = 15.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = _urllib_request.Request(url, data=data, method=method, headers=headers)
    try:
        with _urllib_request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except _urllib_error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except Exception:
            return e.code, {"error": raw.decode(errors="replace")[:200]}


def http_post(url: str, body: dict, timeout: float = 15.0) -> tuple[int, dict]:
    return _http_request("POST", url, body, timeout)


def http_get(url: str, timeout: float = 15.0) -> tuple[int, dict]:
    return _http_request("GET", url, None, timeout)


# ── Helpers ───────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"run_started": now_iso(), "presets": {}}
    return json.loads(path.read_text())


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Atomic write: dump to .tmp, then rename. Survives mid-write crashes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=False))
    tmp.replace(path)


def enqueue_job(base_url: str, body: dict[str, Any]) -> str:
    """POST /api/generate, returning job_id. Backs off on 503 (queue full)."""
    delay = 2.0
    for _ in range(8):
        status, payload = http_post(f"{base_url}/api/generate", body)
        if status == 200:
            return payload["job_id"]
        if status == 503:
            time.sleep(delay)
            delay = min(delay * 1.5, 30)
            continue
        raise RuntimeError(f"HTTP {status} from /api/generate: {payload}")
    raise RuntimeError("queue full after 8 backoff attempts")


def poll_job(base_url: str, job_id: str, timeout: float = JOB_TIMEOUT_S) -> dict[str, Any]:
    """Poll /api/jobs/{id} until status is terminal."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, j = http_get(f"{base_url}/api/jobs/{job_id}")
        if status != 200:
            raise RuntimeError(f"HTTP {status} polling job {job_id}: {j}")
        if j.get("status") in ("completed", "failed", "cancelled"):
            return j
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"job {job_id} didn't finish in {timeout}s")


def best_scannability(job: dict[str, Any]) -> float:
    cands = job.get("candidate_list", [])
    return max(
        (float(c.get("scannability") or 0) for c in cands),
        default=0.0,
    )


# ── Calibration loop ──────────────────────────────────────────────────────────
def build_request_body(preset, settings: dict[str, Any], data: str) -> dict[str, Any]:
    subject = (
        preset.placeholder_subjects[0]
        if preset.placeholder_subjects
        else "a striking subject"
    )
    body = {
        "data": data,
        "prompt": preset.prompt.replace("{SUBJECT}", subject),
        "negative_prompt": preset.negative_override,
        **settings,
    }
    # Strip any None values to avoid sending null over the wire.
    return {k: v for k, v in body.items() if v is not None or k == "negative_prompt"}


def calibrate_one(
    preset,
    base_url: str,
    data: str,
    max_attempts: int,
    candidates: int,
    fast_mode: bool,
) -> dict[str, Any]:
    """Run the calibration loop for a single preset. Returns the record dict."""
    base_settings = dict(preset.settings)
    start = time.time()
    scale = base_settings["controlnet_scale"]
    coverage = base_settings["qr_coverage"]
    last_score = 0.0
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        settings = dict(base_settings)
        settings["controlnet_scale"] = scale
        settings["qr_coverage"] = coverage
        settings["candidates"] = candidates
        if fast_mode:
            settings["fast_mode"] = True
            settings["steps"] = 6
            settings["refine"] = False
            settings["hires_fix"] = False

        body = build_request_body(preset, settings, data)
        try:
            job_id = enqueue_job(base_url, body)
            job = poll_job(base_url, job_id)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            break

        last_score = best_scannability(job)
        elapsed = round(time.time() - start, 1)

        if job.get("scans"):
            return {
                "status": "success",
                "attempts": attempt,
                "calibrated_scale": round(scale, 3),
                "calibrated_coverage": round(coverage, 3),
                "best_score": round(last_score, 3),
                "elapsed_s": elapsed,
            }

        # First-attempt hopeless gate — prompt is fundamentally bad for QR.
        if attempt == 1 and last_score < SCORE_FLOOR:
            return {
                "status": "hopeless",
                "attempts": attempt,
                "calibrated_scale": round(scale, 3),
                "calibrated_coverage": round(coverage, 3),
                "best_score": round(last_score, 3),
                "elapsed_s": elapsed,
                "reason": f"best score {last_score:.2f} below floor {SCORE_FLOOR}",
            }

        # Bump for next attempt.
        next_scale = round(scale + SCALE_BUMP, 3)
        if next_scale > ESCALATE_CAP:
            return {
                "status": "cap_hit",
                "attempts": attempt,
                "calibrated_scale": round(scale, 3),
                "calibrated_coverage": round(coverage, 3),
                "best_score": round(last_score, 3),
                "elapsed_s": elapsed,
                "reason": f"would exceed ESCALATE_CAP {ESCALATE_CAP}, still no scan",
            }
        scale = next_scale
        if attempt == COVERAGE_BUMP_AT_ATTEMPT and coverage < COVERAGE_BUMP_CEILING:
            coverage = round(
                min(coverage + COVERAGE_BUMP_AMOUNT, COVERAGE_BUMP_CEILING), 3
            )

    return {
        "status": "failed",
        "attempts": max_attempts,
        "calibrated_scale": round(scale, 3),
        "calibrated_coverage": round(coverage, 3),
        "best_score": round(last_score, 3),
        "elapsed_s": round(time.time() - start, 1),
        "reason": last_error or f"exhausted {max_attempts} attempts without scan",
    }


# ── Optional: write calibrations back to qrart/presets.py ─────────────────────
CALIBRATIONS_HEADER = "\n\n# ── Auto-calibrated overrides ───────────────────────────────────────────────"
CALIBRATIONS_FOOTER = "# End auto-calibrated overrides\n"


def update_presets_file(report: dict[str, Any]) -> int:
    """Append (or replace) a CALIBRATIONS dict at the end of qrart/presets.py
    that overlays calibrated controlnet_scale + qr_coverage per slug.

    The overlay is applied at import time via a small for-loop, so PRESETS
    and PRESETS_BY_SLUG always reflect the calibrated values without us
    needing to rewrite each Preset() construction by hand.

    Returns the number of overrides written. Idempotent: re-running replaces
    the existing block."""
    presets_file = ROOT / "qrart" / "presets.py"
    contents = presets_file.read_text()

    # Strip any previous calibration block (everything between the header
    # and footer markers, inclusive).
    pattern = re.compile(
        re.escape(CALIBRATIONS_HEADER) + r".*?" + re.escape(CALIBRATIONS_FOOTER),
        re.DOTALL,
    )
    contents = pattern.sub("", contents).rstrip() + "\n"

    # Build the overlay entries. Only emit when the calibrated value
    # actually differs from the default — keeps the overlay minimal.
    overrides: list[str] = []
    for slug in sorted(report.get("presets", {}).keys()):
        rec = report["presets"][slug]
        if rec.get("status") != "success":
            continue
        preset = PRESETS_BY_SLUG.get(slug)
        if preset is None:
            continue
        cur_scale = preset.settings["controlnet_scale"]
        cur_coverage = preset.settings["qr_coverage"]
        new_scale = rec["calibrated_scale"]
        new_coverage = rec["calibrated_coverage"]
        deltas: list[str] = []
        if abs(new_scale - cur_scale) >= 0.005:
            deltas.append(f'"controlnet_scale": {new_scale}')
        if abs(new_coverage - cur_coverage) >= 0.005:
            deltas.append(f'"qr_coverage": {new_coverage}')
        if not deltas:
            continue
        overrides.append(f'    "{slug}": {{{", ".join(deltas)}}},')

    block_lines = [
        CALIBRATIONS_HEADER,
        "# Generated by scripts/calibrate_presets.py. Each entry overrides the",
        "# base preset's controlnet_scale (and qr_coverage if needed) with the",
        "# empirically-confirmed-scannable value. Re-run the calibration script",
        "# with --update-presets to regenerate. Edit by hand only if you've",
        "# verified the new value still scans.",
        "CALIBRATIONS: dict[str, dict] = {",
        *overrides,
        "}",
        "",
        "# Apply calibrations at import time so PRESETS/PRESETS_BY_SLUG always",
        "# reflect them — no separate fetch needed.",
        "for _slug, _overrides in CALIBRATIONS.items():",
        "    _p = PRESETS_BY_SLUG.get(_slug)",
        "    if _p is not None:",
        "        _p.settings.update(_overrides)",
        CALIBRATIONS_FOOTER,
    ]
    new_contents = contents + "\n".join(block_lines)
    presets_file.write_text(new_contents)
    return len(overrides)


# ── Tier-range filtering ──────────────────────────────────────────────────────
TIER_BOUNDS = [
    (1, 0, 30),    # TIER1: slots 0..29
    (2, 30, 60),
    (3, 60, 90),
    (4, 90, 150),
    (5, 150, 200),
]


def filter_by_tier(presets: list, tier: int) -> list:
    for n, lo, hi in TIER_BOUNDS:
        if n == tier:
            return presets[lo:hi]
    return presets


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data", default="https://github.com/briankwest/qrart",
                    help="payload to encode (default: https://github.com/briankwest/qrart)")
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="cap retries per preset (default 4)")
    ap.add_argument("--candidates", type=int, default=3,
                    help="candidates per attempt (default 3, fewer = faster)")
    ap.add_argument("--fast-mode", action="store_true",
                    help="use LCM 6-step for ~3x speedup (less accurate)")
    ap.add_argument("--only-tier", type=int, choices=[1, 2, 3, 4, 5],
                    help="restrict to one tier")
    ap.add_argument("--only-slug",
                    help="comma-separated slugs to run")
    ap.add_argument("--skip-passing", action="store_true",
                    help="resume — skip presets already marked success in the report")
    ap.add_argument("--update-presets", action="store_true",
                    help="after run, write CALIBRATIONS overlay back into qrart/presets.py")
    ap.add_argument("--report", default="presets_calibration.json",
                    help="JSON report path (default presets_calibration.json)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="server URL (default http://127.0.0.1:8000)")
    args = ap.parse_args()

    report_path = Path(args.report)
    report = load_report(report_path)
    report.setdefault("presets", {})

    # Sanity-check the server is up.
    try:
        status, _ = http_get(f"{args.base_url}/api/health", timeout=5)
        if status != 200:
            print(f"ERROR: /api/health returned {status} — is the server running?")
            return 1
    except Exception as e:
        print(f"ERROR: can't reach {args.base_url}: {e}")
        return 1

    presets = PRESETS
    if args.only_slug:
        wanted = set(s.strip() for s in args.only_slug.split(","))
        presets = [p for p in presets if p.slug in wanted]
    if args.only_tier:
        presets = filter_by_tier(presets, args.only_tier)

    print(f"calibrating {len(presets)} preset(s)")
    print(f"  data:         {args.data}")
    print(f"  max attempts: {args.max_attempts}")
    print(f"  candidates:   {args.candidates}")
    print(f"  fast mode:    {args.fast_mode}")
    print(f"  base url:     {args.base_url}")
    print(f"  report:       {report_path}")
    print()

    try:
        for i, preset in enumerate(presets, 1):
            if preset.requires_init:
                print(f"[{i:>3}/{len(presets)}] {preset.slug}: SKIP (needs init image)")
                report["presets"][preset.slug] = {"status": "skipped_init_required"}
                save_report(report_path, report)
                continue

            if args.skip_passing:
                prev = report["presets"].get(preset.slug)
                if prev and prev.get("status") == "success":
                    print(f"[{i:>3}/{len(presets)}] {preset.slug}: SKIP (already success)")
                    continue

            print(f"[{i:>3}/{len(presets)}] {preset.slug}: calibrating...", flush=True)
            rec = calibrate_one(
                preset, args.base_url, args.data,
                args.max_attempts, args.candidates, args.fast_mode,
            )
            report["presets"][preset.slug] = rec
            tag = rec["status"].upper()
            scale = rec.get("calibrated_scale")
            score = rec.get("best_score")
            tries = rec.get("attempts")
            elapsed = rec.get("elapsed_s")
            print(
                f"            → {tag} · scale={scale} · score={score} · "
                f"attempts={tries} · {elapsed}s",
                flush=True,
            )
            save_report(report_path, report)

    except KeyboardInterrupt:
        print("\ninterrupted — saving partial report")
        save_report(report_path, report)
        return 130

    # Summary
    counts: dict[str, int] = {}
    for rec in report["presets"].values():
        s = rec.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    print("\nSummary:")
    for s in ("success", "hopeless", "cap_hit", "failed", "skipped_init_required"):
        if s in counts:
            print(f"  {s:24s} {counts[s]}")

    if args.update_presets:
        n = update_presets_file(report)
        print(f"\nwrote {n} calibration override(s) into qrart/presets.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
