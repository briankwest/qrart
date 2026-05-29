#!/usr/bin/env python3
"""Build GALLERY.md from every scanning candidate in the DB.

Filters candidates whose decoded value matches a target URL and whose
scannability score meets a minimum threshold. For each surviving
candidate, attempts to identify which preset produced it via three
fallbacks (exact prompt match → prefix match → settings fingerprint).
Picks the best-scoring candidate per preset, copies the images into a
gallery/ directory with stable filenames, and writes a categorized
markdown gallery suitable for committing to a GitHub repo.

Run with the project's venv:
  venv/bin/python scripts/build_gallery.py [options]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qrart.presets import PRESETS, PRESETS_BY_SLUG  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "qrart.db"

# Categories whose presets are technical recipes (Fast mode, print-ready,
# branded-logo) rather than aesthetic ones. The prompts are generic
# placeholders ("a beautiful landscape") that can land on anything,
# including content not appropriate for a public gallery. Skip by default.
DEFAULT_EXCLUDE_CATEGORIES = {
    "⚡ Quick & Special",
}


# ── DB ────────────────────────────────────────────────────────────────────────
def url_to_fs(url_path: str) -> Path:
    """Convert /outputs/{...} URL path to filesystem path."""
    if url_path.startswith("/outputs/"):
        return ROOT / "outputs" / url_path[len("/outputs/"):]
    return Path(url_path)


def fetch_scanning_candidates(
    db_path: Path,
    target_url: str,
    min_score: float,
) -> list[dict[str, Any]]:
    """Return scanning candidates that decode to target_url with score >= min_score.
    Each row carries both candidate columns and joined job context."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
          c.id           AS cand_id,
          c.idx          AS cand_idx,
          c.seed         AS cand_seed,
          c.scans        AS cand_scans,
          c.decoded      AS cand_decoded,
          c.scannability AS cand_score,
          c.image_path   AS cand_image_path,
          c.pass1_image_path AS cand_pass1_image_path,
          c.controlnet_scale AS cand_scale,
          j.id           AS job_id,
          j.created_at   AS job_created,
          j.prompt       AS job_prompt,
          j.negative_prompt AS job_negative,
          j.style        AS job_style,
          j.model        AS job_model,
          j.composition  AS job_composition,
          j.qr_monster_version AS job_qr_version,
          j.controlnet_scale AS job_scale,
          j.qr_coverage  AS job_coverage,
          j.tile_scale   AS job_tile,
          j.control_start AS job_control_start,
          j.control_end  AS job_control_end,
          j.steps        AS job_steps,
          j.candidates   AS job_num_cands,
          j.qr_image_path AS job_qr_image,
          j.elapsed_s    AS job_elapsed
        FROM candidates c
        JOIN jobs j ON c.job_id = j.id
        WHERE c.scans = 1
          AND c.decoded = ?
          AND (c.scannability IS NULL OR c.scannability >= ?)
        ORDER BY c.scannability DESC, c.id ASC
        """,
        (target_url, min_score),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Preset identification ─────────────────────────────────────────────────────
def build_prompt_index() -> dict[str, str]:
    """{canonical_prompt_string: preset_slug} for tier-1 exact + tier-2 prefix
    matching. Each preset contributes prompt × N placeholder_subjects entries."""
    idx: dict[str, str] = {}
    for p in PRESETS:
        subjects = p.placeholder_subjects or [""]
        for subj in subjects:
            canonical = p.prompt.replace("{SUBJECT}", subj) if subj else p.prompt
            # If two presets share a canonical (rare), last-write wins — fine
            # because identical prompts produce identical preset behavior.
            idx[canonical] = p.slug
    return idx


def fingerprint(row_or_settings: dict[str, Any]) -> tuple:
    """Identity tuple for preset matching: the parts of the request that are
    most distinctive across presets."""
    def get(k1: str, k2: str | None = None, default=None):
        if k1 in row_or_settings:
            return row_or_settings[k1]
        if k2 and k2 in row_or_settings:
            return row_or_settings[k2]
        return default
    scale = get("job_scale", "controlnet_scale") or 0.0
    coverage = get("job_coverage", "qr_coverage") or 0.0
    return (
        get("job_model", "model"),
        get("job_qr_version", "qr_monster_version"),
        get("job_composition", "composition"),
        get("job_style", "style"),
        round(float(scale), 2),
        round(float(coverage), 2),
    )


def build_fingerprint_index() -> dict[tuple, str]:
    """Only return fingerprints that uniquely identify one preset — ambiguous
    fingerprints are dropped so we don't mistag jobs."""
    counts: dict[tuple, int] = {}
    mapping: dict[tuple, str] = {}
    for p in PRESETS:
        fp = fingerprint(p.settings)
        counts[fp] = counts.get(fp, 0) + 1
        mapping[fp] = p.slug
    return {fp: slug for fp, slug in mapping.items() if counts[fp] == 1}


def identify_preset(
    row: dict[str, Any],
    prompt_idx: dict[str, str],
    prefix_list: list[tuple[str, str]],
    fp_idx: dict[tuple, str],
) -> str | None:
    prompt = row.get("job_prompt")
    if prompt:
        # Tier 1: exact match
        if prompt in prompt_idx:
            return prompt_idx[prompt]
        # Tier 2: prefix match (try longer canonical prompts first)
        for canonical, slug in prefix_list:
            if prompt.startswith(canonical):
                return slug
    # Tier 3: settings fingerprint
    return fp_idx.get(fingerprint(row))


# ── Gallery assembly ──────────────────────────────────────────────────────────
def slug_safe(s: str) -> str:
    """Filesystem-safe slug; falls back to a hash-like short id if empty."""
    s = re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")
    return s or "untagged"


def short_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())[:8] or "unknown"


def truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n].rstrip() + "…"


def md_escape(s: str) -> str:
    """Minimal escape for markdown inside HTML tags. The HTML grid wraps each
    cell, so the prompt becomes <sub> content — angle brackets need escaping."""
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--target-url", default="https://github.com/briankwest/qrart",
                    help="filter candidates whose decoded value matches "
                    "(default: https://github.com/briankwest/qrart)")
    ap.add_argument("--min-score", type=float, default=0.90,
                    help="only include candidates with scannability ≥ X (default 0.90)")
    ap.add_argument("--gallery-dir", default="gallery",
                    help="where to copy images (default gallery/)")
    ap.add_argument("--output", default="GALLERY.md",
                    help="markdown output path (default GALLERY.md)")
    ap.add_argument("--all", dest="per_preset", action="store_false", default=True,
                    help="emit every matching candidate (default: best per preset)")
    ap.add_argument("--include-qr", action="store_true",
                    help="also copy each entry's source qr.png")
    ap.add_argument("--columns", type=int, default=3,
                    help="thumbnails per row in the HTML grid (default 3)")
    ap.add_argument("--max-prompt-len", type=int, default=140,
                    help="truncate the displayed prompt (default 140)")
    ap.add_argument("--append", action="store_true",
                    help="don't blow away gallery/ before run (default: clean rebuild)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't copy or write — just print what would happen")
    ap.add_argument("--db", default=str(DB_PATH),
                    help=f"path to qrart.db (default {DB_PATH})")
    ap.add_argument("--exclude-category", action="append", default=[],
                    help="exclude an entire category by substring match "
                    "(repeatable). Default skips: " + ", ".join(repr(c) for c in DEFAULT_EXCLUDE_CATEGORIES))
    ap.add_argument("--include-all-categories", action="store_true",
                    help="override the default category exclusion list")
    args = ap.parse_args()

    # Build the effective exclusion set.
    exclude: set[str] = set()
    if not args.include_all_categories:
        exclude |= DEFAULT_EXCLUDE_CATEGORIES
    exclude |= set(args.exclude_category)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 1

    print(f"Building gallery")
    print(f"  db:         {db_path}")
    print(f"  target URL: {args.target_url}")
    print(f"  min score:  {args.min_score}")
    print(f"  per-preset: {args.per_preset}")
    print()

    rows = fetch_scanning_candidates(db_path, args.target_url, args.min_score)
    print(f"  found {len(rows)} scanning candidate(s) decoding to target URL")
    if not rows:
        print("  nothing to do — exiting cleanly")
        return 0

    prompt_idx = build_prompt_index()
    prefix_list = sorted(prompt_idx.items(), key=lambda x: -len(x[0]))
    fp_idx = build_fingerprint_index()

    # Tag every row with a preset slug (or None for uncategorized).
    tagged: list[tuple[str | None, dict[str, Any]]] = []
    for row in rows:
        slug = identify_preset(row, prompt_idx, prefix_list, fp_idx)
        tagged.append((slug, row))

    matched = sum(1 for s, _ in tagged if s)
    print(f"  matched {matched}/{len(tagged)} to a preset "
          f"({len(tagged) - matched} uncategorized)")

    # Reduce to one-per-preset if requested.
    if args.per_preset:
        best_by_slug: dict[str, tuple[str, dict]] = {}
        uncategorized: list[tuple[None, dict]] = []
        for slug, row in tagged:
            if slug is None:
                uncategorized.append((None, row))
                continue
            prev = best_by_slug.get(slug)
            if prev is None or (row["cand_score"] or 0) > (prev[1]["cand_score"] or 0):
                best_by_slug[slug] = (slug, row)
        # Dedupe uncategorized by (prompt-prefix, model) so identical custom
        # runs only contribute one entry.
        seen: set = set()
        keep_uncat: list[tuple[None, dict]] = []
        for _, row in uncategorized:
            key = ((row.get("job_prompt") or "")[:80], row.get("job_model"))
            if key in seen:
                continue
            seen.add(key)
            keep_uncat.append((None, row))
        entries = list(best_by_slug.values()) + keep_uncat
    else:
        entries = tagged

    print(f"  selected {len(entries)} entries for the gallery")
    print()

    # Group by category (preset.category for matched, fallback bucket for not).
    groups: dict[str, list[tuple[str | None, dict]]] = {}
    excluded_count = 0
    for slug, row in entries:
        if slug:
            preset = PRESETS_BY_SLUG.get(slug)
            cat = preset.category if preset else "📦 Other"
        else:
            cat = "📦 Uncategorized — custom runs"
        # Drop excluded categories. Substring match so partial names work
        # ("Quick" filters out "⚡ Quick & Special").
        if any(ex.lower() in cat.lower() for ex in exclude):
            excluded_count += 1
            continue
        groups.setdefault(cat, []).append((slug, row))
    if exclude:
        print(f"  excluded {excluded_count} entries from filtered categories: "
              f"{', '.join(repr(e) for e in sorted(exclude))}")

    # Maintain preset order within each category, then descending by score.
    preset_order = {p.slug: i for i, p in enumerate(PRESETS)}
    for cat, items in groups.items():
        items.sort(
            key=lambda e: (
                preset_order.get(e[0], 1_000_000),
                -((e[1]["cand_score"] or 0)),
            )
        )

    if args.dry_run:
        for cat, items in groups.items():
            print(f"\n{cat}: {len(items)} entries")
            for slug, row in items[:5]:
                tag = slug or "(uncategorized)"
                print(f"  {tag} · score={row['cand_score']:.2f} · {row['cand_image_path']}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
        return 0

    # Prepare gallery dir.
    gallery_dir = ROOT / args.gallery_dir
    if not args.append and gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(exist_ok=True)

    # Copy each entry's image (and optionally qr.png) into the gallery dir,
    # building the per-cell metadata as we go.
    md_groups: dict[str, list[dict[str, Any]]] = {}
    copied = 0
    skipped = 0
    for cat, items in groups.items():
        md_groups[cat] = []
        for slug, row in items:
            src = url_to_fs(row["cand_image_path"])
            if not src.exists():
                print(f"  warning: missing file {src} (cand {row['cand_id']}), skipping",
                      file=sys.stderr)
                skipped += 1
                continue
            if slug:
                base = slug_safe(slug)
                if not args.per_preset:
                    base = f"{base}-{short_id(row['cand_id'])}"
            else:
                base = f"custom-{short_id(row['job_id'])}"
            dest = gallery_dir / f"{base}.png"
            shutil.copy2(src, dest)
            copied += 1
            qr_rel = None
            if args.include_qr and row.get("job_qr_image"):
                qr_src = url_to_fs(row["job_qr_image"])
                if qr_src.exists():
                    qr_dest = gallery_dir / f"{base}.qr.png"
                    shutil.copy2(qr_src, qr_dest)
                    qr_rel = f"{args.gallery_dir}/{base}.qr.png"

            preset = PRESETS_BY_SLUG.get(slug) if slug else None
            name = preset.name if preset else f"Custom · job {row['job_id'][:8]}"
            scale = row["cand_scale"] or row["job_scale"] or 0
            md_groups[cat].append({
                "name": name,
                "slug": slug,
                "icon": preset.icon if preset else "📦",
                "image_rel": f"{args.gallery_dir}/{base}.png",
                "qr_rel": qr_rel,
                "prompt": truncate(row.get("job_prompt") or "", args.max_prompt_len),
                "model": row.get("job_model"),
                "score": row["cand_score"] or 0,
                "seed": row["cand_seed"],
                "scale": float(scale),
                "great_fit": bool(preset.great_fit) if preset else False,
            })

    print(f"  copied {copied} image(s) to {gallery_dir}/" + (f" (skipped {skipped})" if skipped else ""))

    # Build the markdown.
    lines: list[str] = []
    lines.append("# QR Art Gallery")
    lines.append("")
    lines.append(f"Scannable QR art outputs that decode to `{args.target_url}`.")
    lines.append("Every image below is a real QR code — point your phone at it.")
    lines.append("")
    lines.append(
        f"_Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"{copied} entries · {len(groups)} categories · "
        f"min scan score {args.min_score}_"
    )
    lines.append("")
    lines.append("Generated with [QR Art Studio](https://github.com/briankwest/qrcode) — "
                 f"a local Stable Diffusion 1.5 + QR Monster ControlNet generator with "
                 f"200 one-click presets, multi-scanner verification, and an automated "
                 f"calibration tool.")
    lines.append("")

    cols = args.columns
    cell_width_pct = 100 // cols
    for cat, items in md_groups.items():
        if not items:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("<table>")
        for row_start in range(0, len(items), cols):
            row_items = items[row_start:row_start + cols]
            lines.append("  <tr>")
            for it in row_items:
                fit = ' <sub title="great QR fit">★</sub>' if it["great_fit"] else ""
                qr_link = (
                    f'<br><a href="{it["qr_rel"]}"><sub>view source QR ↗</sub></a>'
                    if it.get("qr_rel") else ""
                )
                lines.append(
                    f'    <td align="center" width="{cell_width_pct}%" valign="top">\n'
                    f'      <a href="{it["image_rel"]}">'
                    f'<img src="{it["image_rel"]}" width="240" alt="{md_escape(it["name"])}" />'
                    f'</a>\n'
                    f'      <br><b>{md_escape(it["name"])}</b>{fit}\n'
                    f'      <br><sub>{md_escape(it["prompt"])}</sub>\n'
                    f'      <br><sub>★ {it["score"]:.2f} · seed {it["seed"]} · '
                    f'scale {it["scale"]:.2f} · {md_escape(it["model"] or "?")}</sub>'
                    f'{qr_link}\n'
                    f'    </td>'
                )
            for _ in range(cols - len(row_items)):
                lines.append('    <td></td>')
            lines.append("  </tr>")
        lines.append("</table>")
        lines.append("")

    output_path = ROOT / args.output
    output_path.write_text("\n".join(lines))
    print(f"  wrote {output_path}")
    print()
    print("Done. Review GALLERY.md and the gallery/ dir, then:")
    print(f"  git add {args.output} {args.gallery_dir}/")
    print(f"  git commit -m 'Update QR art gallery'")
    print(f"  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
