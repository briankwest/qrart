from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

import asyncio
import json as _json
from dataclasses import replace as _replace

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import hashlib
import io

from qrart import COMPOSITIONS, Generator, GenerationRequest, STYLE_PRESETS
from qrart.db import get_db, new_job_id
from qrart.generator import Progress
from qrart.pipeline import (
    MODELS,
    QR_MONSTER_VERSIONS,
    QR_MONSTER_DEFAULT,
    QRART_SAMPLER,
    CancelledByUser,
)
from qrart.worker import Job, MAX_QUEUED, QueueFull, Worker

# A2: auto-escalation tuning. When the user opts in (require_scan=True,
# auto_escalate=True), all-fail jobs spawn a follow-up at scale +0.1, capped.
# best-score floor avoids escalating on hopeless prompts where the QR will
# never resolve regardless of scale (e.g. dark cosmic scenes that fight QR
# luminance fundamentally).
ESCALATE_STEP = 0.10
ESCALATE_CAP = 1.50
ESCALATE_MIN_SCORE = 0.70

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
# Content-addressed asset store for user uploads (init images, future
# IP-Adapter references, logos). Files are named by SHA256 so the same
# image uploaded twice deduplicates automatically. Served via the
# existing /outputs/_assets/* static mount.
ASSETS_DIR = OUTPUT_DIR / "_assets"
ASSETS_DIR.mkdir(exist_ok=True)
ASSET_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
ASSET_FORMATS = {"PNG", "JPEG", "WEBP"}

app = FastAPI(title="QR Art Studio")


# Optional shared-password auth. If QRART_AUTH=user:pass is set in the env,
# every /api/* request (and /outputs/* + the index) is gated behind HTTP
# Basic auth using a timing-safe compare. /api/health stays open so external
# probes don't have to know the password.
import base64 as _b64
import os as _os
import secrets as _secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as _Response

_AUTH = _os.environ.get("QRART_AUTH")


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not _AUTH:
            return await call_next(request)
        path = request.url.path
        # Allow open access to the health endpoint so probes don't need creds.
        if path == "/api/health":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        ok = False
        if header.lower().startswith("basic "):
            try:
                creds = _b64.b64decode(header.split(" ", 1)[1]).decode()
                ok = _secrets.compare_digest(creds, _AUTH)
            except Exception:
                ok = False
        if not ok:
            return _Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="QR Art Studio"'},
            )
        return await call_next(request)


if _AUTH:
    app.add_middleware(_BasicAuthMiddleware)
    print("[auth] basic auth enabled via QRART_AUTH env var", flush=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


RETENTION_KEEP = int(__import__("os").environ.get("QRART_RETENTION_KEEP", "1000"))


def _cleanup_evicted_files(evicted_ids: list[str]) -> int:
    """Best-effort rm -rf of the outputs/{id}/ directories for evicted jobs.
    Returns the number of directories actually removed."""
    import shutil as _shutil
    removed = 0
    for jid in evicted_ids:
        d = OUTPUT_DIR / jid
        if d.exists():
            try:
                _shutil.rmtree(d)
                removed += 1
            except OSError:
                pass
    return removed


@app.on_event("startup")
def _startup() -> None:
    # Initialize SQLite + run migrations + mark orphans before the first
    # request arrives. Runs once per process.
    db = get_db()
    evicted = db.evict_old_jobs(keep=RETENTION_KEEP)
    if evicted:
        removed = _cleanup_evicted_files(evicted)
        print(
            f"[retention] kept newest {RETENTION_KEEP} jobs, "
            f"evicted {len(evicted)} (removed {removed} output dirs)",
            flush=True,
        )
    _worker.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _worker.stop()

# One Generator per model, lazily created on first use. Models that haven't
# been used in this session don't consume memory. Switching back to a model
# you used earlier is instant.
_generators: dict[str, Generator] = {}
_gen_lock = threading.Lock()


def get_generator(model: str | None = None) -> Generator:
    key = model if model in MODELS else "photoreal"
    with _gen_lock:
        if key not in _generators:
            _generators[key] = Generator(base_model=key)
    return _generators[key]


class GenerateBody(BaseModel):
    data: str = Field(..., description="URL or text to encode")
    prompt: str
    style: str = "photoreal"
    model: str = "photoreal"
    negative_prompt: str | None = None
    candidates: int = 5
    steps: int = 32
    controlnet_scale: float = 1.10
    tile_scale: float = 0.0
    # Control window — when ControlNet conditions diffusion. Fractions of
    # total denoising steps. (0.30, 0.95) is the v2 community sweet spot:
    # txt2img paints the scene from steps 0-30%, QR Monster shapes it
    # from 30-95%, the last 5% finishes naturally without QR pull.
    control_start: float = 0.30
    control_end: float = 0.95
    guidance: float = 7.5
    refine: bool = True
    refine_strength: float = 0.30
    refine_steps: int = 20
    size: int = 768
    composition: str = "standalone"
    seed: int | None = None
    require_scan: bool = True
    # A2: when require_scan is on and zero candidates pass, auto-resubmit with
    # scale +0.1 (capped at 1.5) until one passes or we hit the cap. Off → user
    # gets the failed result and can manually retry.
    auto_escalate: bool = True
    # QR Monster ControlNet version: 'v1' (default) or 'v2'. Both are loaded
    # at warm time and swapped per-request without a model reload.
    qr_monster_version: str = "v1"
    # Fraction of the diffusion canvas the QR occupies. 1.0 = QR fills the
    # canvas (legacy). <1.0 leaves a #808080 gray margin where the prompt's
    # scene grows naturally. 0.70-0.80 is the sweet spot for "QR as a
    # feature inside a larger scene."
    qr_coverage: float = 1.0
    # User-uploaded init image (URL path, e.g. /outputs/_assets/<sha>.png).
    # When set: standalone uses it as the img2img init for pass-1;
    # compositions use it as the scene the QR art is composited into.
    init_image_path: str | None = None
    # 0.0 = preserve init unchanged (no diffusion happens); 1.0 = reimagine
    # entirely (equivalent to no init). Useful band: 0.5-0.85 for "QR-ify
    # this photo", 0.25-0.45 for "barely touch it, just embed the QR."
    init_strength: float = 0.65
    # Canny ControlNet weight for the init image's structure. 0 = ignore
    # structure (img2img alone); 0.5-0.8 typical for logo-shaped QR codes
    # where the modules should cluster along the logo's outlines.
    canny_scale: float = 0.0
    fast_mode: bool = False
    hires_fix: bool = False
    hires_target: int = 1024
    hires_strength: float = 0.20
    adetailer: bool = False
    adetailer_strength: float = 0.35
    # Phase 1: link a remix back to the source job. When the UI loads a past
    # job's settings into the form and the user resubmits, it sets this so the
    # history can show "remixed from <id>" lineage.
    parent_job_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/health")
def health() -> dict[str, Any]:
    g = get_generator()
    loaded_models = [k for k, gen in _generators.items() if gen.pipeline._pipe is not None]
    state = _worker.state()
    return {
        "ok": True,
        "device": g.pipeline.device,
        "loaded": g.pipeline._pipe is not None,
        "base_model": g.pipeline.base_model,
        "qr_monster_default": QR_MONSTER_DEFAULT,
        "qr_monster_versions": list(QR_MONSTER_VERSIONS),
        "sampler": QRART_SAMPLER,
        "styles": list(STYLE_PRESETS.keys()),
        "compositions": list(COMPOSITIONS.keys()),
        "models": list(MODELS.keys()),
        "loaded_models": loaded_models,
        **state,  # busy, active_model, active_job_id, active_elapsed_s, queue_depth, queued_ids, max_queued
    }


class WarmBody(BaseModel):
    model: str = "photoreal"


@app.post("/api/warm")
def warm(body: WarmBody | None = None) -> dict[str, Any]:
    t0 = time.time()
    model = body.model if body else "photoreal"
    get_generator(model).warm()
    return {"ok": True, "elapsed_s": round(time.time() - t0, 2), "model": model}


@app.post("/api/assets")
async def upload_asset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Content-addressed upload for init images / reference assets.

    SHA256-hashes the bytes, stores at /outputs/_assets/{sha}.png after
    re-encoding through PIL (normalizes color profile, strips metadata,
    enforces a real image format). Idempotent — uploading the same file
    twice returns the existing record without rewriting.
    """
    raw = await file.read()
    if len(raw) > ASSET_MAX_BYTES:
        raise HTTPException(413, f"asset too large (max {ASSET_MAX_BYTES // (1024 * 1024)} MB)")
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # parses header, raises on garbage
        img = Image.open(io.BytesIO(raw))  # verify() consumes the buffer
        fmt = (img.format or "").upper()
    except Exception:
        raise HTTPException(400, "unreadable image")
    if fmt not in ASSET_FORMATS:
        raise HTTPException(415, f"unsupported format {fmt}; use PNG/JPEG/WEBP")

    sha = hashlib.sha256(raw).hexdigest()
    path = ASSETS_DIR / f"{sha}.png"
    url = f"/outputs/_assets/{sha}.png"
    if not path.exists():
        img.convert("RGB").save(path, "PNG", optimize=True)
    width, height = img.size
    return {
        "hash": sha,
        "url": url,
        "width": width,
        "height": height,
        "format": "PNG",
        "bytes": path.stat().st_size,
    }


def _run_job(job: Job, cancelled: bool) -> None:
    """Worker callback. Drives the pipeline + persistence for one job.

    Cancelled jobs (cancelled while queued) get marked 'cancelled' in DB and
    never invoke the pipeline. Otherwise we update to 'running', generate,
    save outputs + candidates, and finish to 'completed' (or 'failed').
    """
    db = get_db()
    if cancelled:
        db.finish_job(job.job_id, status="cancelled", elapsed_s=0.0)
        return

    db.mark_running(job.job_id)
    t0 = time.time()

    # Set up the output dir + early QR-image save so the UI can render the
    # source QR before any candidates land. The QR image is rebuilt
    # identically inside Generator.generate() — building it twice is cheap.
    job_dir = OUTPUT_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        from qrart.canvas import build_composition as _build_comp
        _comp = _build_comp(
            job.request.data, job.request.composition,
            job.request.qr_monster_version, job.request.qr_coverage,
        )
        qr_path = job_dir / "qr.png"
        _comp.qr_image.save(qr_path)
    except Exception as _e:
        print(f"[worker] early QR save failed: {_e}", flush=True)
        qr_path = None

    # Track candidate DB ids in completion order (== idx for the main loop;
    # any C1 rescue gets appended). Used after generate() for best-candidate
    # selection and for post-processing updates on the winner.
    candidate_ids: list[str] = []

    def _save_candidate(idx: int, cand) -> dict[str, Any]:
        """Persist a candidate's image AND DB row the instant the generator
        finishes it. This is what lets the UI render candidates
        incrementally — and also what lets a cancelled job retain the
        partial output the user already saw (the post-generate fallback
        loop never runs on cancellation).
        Returns the URL fields that get merged into the candidate_done
        SSE event payload."""
        cand_path = job_dir / f"cand{idx}.png"
        cand.image.save(cand_path)
        url = f"/outputs/{job.job_id}/cand{idx}.png"
        pass1_url: str | None = None
        if cand.pass1_image is not None:
            pass1_path = job_dir / f"cand{idx}.pass1.png"
            cand.pass1_image.save(pass1_path)
            pass1_url = f"/outputs/{job.job_id}/cand{idx}.pass1.png"
        cid = db.insert_candidate(
            job_id=job.job_id,
            idx=idx,
            seed=cand.seed,
            controlnet_scale=cand.controlnet_scale,
            refine_strength=cand.refine_strength,
            scans=cand.scans,
            decoded=cand.decoded,
            image_path=url,
            pass1_image_path=pass1_url,
            scannability=cand.scannability,
        )
        candidate_ids.append(cid)
        return {"url": url, "pass1_url": pass1_url, "candidate_id": cid}

    progress = Progress(
        publish=lambda type_, payload: db.insert_event(job.job_id, type_, payload),
        is_cancelled=lambda: _worker.is_cancelled(job.job_id),
        on_candidate_ready=_save_candidate,
    )
    progress.emit("started", model=job.model)
    if qr_path is not None:
        progress.emit("qr_ready", url=f"/outputs/{job.job_id}/qr.png")

    # Path the early QR save into a URL we can hand to finish_job — both
    # the success and the cancel/fail paths benefit from having the QR
    # image visible in history for partial-output jobs.
    qr_url = f"/outputs/{job.job_id}/qr.png" if qr_path is not None and qr_path.exists() else None

    try:
        result = get_generator(job.model).generate(job.request, progress=progress)
    except CancelledByUser:
        elapsed = round(time.time() - t0, 2)
        db.finish_job(
            job.job_id,
            status="cancelled",
            elapsed_s=elapsed,
            qr_image_path=qr_url,
            best_candidate_id=candidate_ids[0] if candidate_ids else None,
        )
        progress.emit("cancelled")
        print(f"[worker] {job.job_id} CANCELLED after {elapsed}s · partial candidates: {len(candidate_ids)}", flush=True)
        return
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        db.finish_job(
            job.job_id,
            status="failed",
            elapsed_s=elapsed,
            error=f"{e}\n\n{traceback.format_exc()}",
            qr_image_path=qr_url,
            best_candidate_id=candidate_ids[0] if candidate_ids else None,
        )
        progress.emit("failed", error=str(e))
        print(f"[worker] {job.job_id} FAILED: {e}", flush=True)
        return

    elapsed = round(time.time() - t0, 2)

    # Candidates were saved incrementally (image + DB row) by _save_candidate
    # during generation. Backstop only: if the rescue path or any future
    # control flow appended a candidate without going through the callback,
    # save+insert it now. Idempotent — skips any idx that's already in
    # candidate_ids.
    for i, c in enumerate(result.candidates):
        if i < len(candidate_ids):
            continue
        cand_file = job_dir / f"cand{i}.png"
        c.image.save(cand_file)
        pass1_url: str | None = None
        if c.pass1_image is not None:
            pass1_file = job_dir / f"cand{i}.pass1.png"
            c.pass1_image.save(pass1_file)
            pass1_url = f"/outputs/{job.job_id}/cand{i}.pass1.png"
        cid = db.insert_candidate(
            job_id=job.job_id,
            idx=i,
            seed=c.seed,
            controlnet_scale=c.controlnet_scale,
            refine_strength=c.refine_strength,
            scans=c.scans,
            decoded=c.decoded,
            image_path=f"/outputs/{job.job_id}/cand{i}.png",
            pass1_image_path=pass1_url,
            scannability=c.scannability,
        )
        candidate_ids.append(cid)

    # hires_fix / adetailer may have replaced the winner candidate's
    # .image in-place. Re-save the file so the URL points at the
    # post-processed pixels, and patch the DB row to match.
    best_idx_for_save = next(
        (i for i, c in enumerate(result.candidates) if c.image is result.image), -1
    )
    if best_idx_for_save >= 0:
        winner = result.candidates[best_idx_for_save]
        winner.image.save(job_dir / f"cand{best_idx_for_save}.png")
        if best_idx_for_save < len(candidate_ids):
            db.update_candidate(
                candidate_ids[best_idx_for_save],
                scans=winner.scans,
                decoded=winner.decoded,
                scannability=winner.scannability,
            )

    if qr_path is not None and not qr_path.exists():
        result.qr_image.save(qr_path)

    best_idx = next(
        (i for i, c in enumerate(result.candidates) if c.image is result.image), 0
    )
    db.finish_job(
        job.job_id,
        status="completed",
        elapsed_s=elapsed,
        scans=result.scans,
        decoded=result.decoded,
        qr_image_path=f"/outputs/{job.job_id}/qr.png",
        best_candidate_id=candidate_ids[best_idx] if candidate_ids else None,
    )
    progress.emit(
        "completed",
        elapsed_s=elapsed,
        scans=result.scans,
        decoded=result.decoded,
        best_candidate_id=candidate_ids[best_idx] if candidate_ids else None,
    )
    print(f"[worker] {job.job_id} done in {elapsed}s · scans={result.scans}", flush=True)

    _maybe_escalate(job, result, db)


def _maybe_escalate(job: Job, result, db) -> None:
    """A2: when require_scan + auto_escalate are on and zero candidates scanned
    but the best score is salvageable, enqueue a follow-up at scale +0.1.

    The best-score floor avoids burning compute on hopeless prompts where
    QR will never resolve at any scale (cosmic galaxies, plain skies, etc.).
    """
    if not job.body.get("require_scan"):
        return
    if not job.body.get("auto_escalate", True):
        return
    if result.scans:
        return
    # Don't escalate cancelled-mid-run results (the worker would have raised
    # CancelledByUser, so we wouldn't reach here — but defensive).
    if _worker.is_cancelled(job.job_id):
        return

    current_scale = float(job.request.controlnet_scale)
    new_scale = round(current_scale + ESCALATE_STEP, 2)
    if new_scale > ESCALATE_CAP:
        db.insert_event(job.job_id, "escalation_skipped", {
            "reason": f"already at cap ({ESCALATE_CAP})",
            "scale": current_scale,
        })
        return

    best_score = max(
        (float(c.scannability) for c in result.candidates), default=0.0
    )
    if best_score < ESCALATE_MIN_SCORE:
        db.insert_event(job.job_id, "escalation_skipped", {
            "reason": f"best score {best_score:.2f} below floor {ESCALATE_MIN_SCORE}",
            "scale": current_scale,
        })
        return

    new_req = _replace(job.request, controlnet_scale=new_scale, seed=None)
    new_body = {
        **job.body,
        "controlnet_scale": new_scale,
        "seed": None,
        "parent_job_id": job.job_id,
    }
    new_jid = new_job_id()
    try:
        db.insert_job(new_jid, new_body)
        new_job = Job(job_id=new_jid, model=job.model, request=new_req, body=new_body)
        _worker.enqueue(new_job)
        db.insert_event(job.job_id, "auto_escalated", {
            "child_job_id": new_jid,
            "from_scale": current_scale,
            "to_scale": new_scale,
            "best_score": round(best_score, 3),
        })
        print(f"[escalate] {job.job_id} -> {new_jid} (scale {current_scale} -> {new_scale}, score {best_score:.2f})", flush=True)
    except QueueFull:
        db.finish_job(new_jid, status="failed", elapsed_s=0.0,
                      error="queue full during auto-escalation")


_worker = Worker(_run_job)


@app.post("/api/generate")
def generate(body: GenerateBody, request: Request) -> dict[str, Any]:
    """Enqueue a generation job. Returns immediately — poll /api/jobs/{id}
    until status flips to completed / failed / cancelled."""
    if not body.data.strip() or not body.prompt.strip():
        raise HTTPException(400, "data and prompt are required")

    # Fast mode swaps in LCM-LoRA which needs different sampling defaults.
    # Override step/guidance regardless of what the client sent.
    if body.fast_mode:
        steps = 6
        refine_steps = 12
        guidance = 1.5
        # LCM at low CFG (1.5) leaves ControlNet relatively dominant, so the
        # same scale slider produces a much QR-ier image than in Quality mode.
        # Apply a transparent multiplier so the slider's intent matches output.
        controlnet_scale = body.controlnet_scale * 0.75
    else:
        steps = max(10, min(body.steps, 60))
        refine_steps = max(10, min(body.refine_steps, 50))
        guidance = body.guidance
        controlnet_scale = body.controlnet_scale

    print(
        f"[generate] model={body.model} prompt={body.prompt[:60]!r}... "
        f"scale={body.controlnet_scale} tile={body.tile_scale} "
        f"refine={body.refine}/{body.refine_strength} "
        f"composition={body.composition} candidates={body.candidates} "
        f"fast={body.fast_mode} hires={body.hires_fix} adetailer={body.adetailer} "
        f"seed={body.seed}",
        flush=True,
    )

    composition = body.composition if body.composition in COMPOSITIONS else "standalone"

    req = GenerationRequest(
        data=body.data,
        prompt=body.prompt,
        style=body.style,
        negative_prompt=body.negative_prompt,
        candidates=max(1, min(body.candidates, 8)),
        steps=steps,
        controlnet_scale=controlnet_scale,
        tile_scale=max(0.0, min(body.tile_scale, 1.0)),
        control_start=max(0.0, min(body.control_start, 0.9)),
        control_end=max(0.1, min(body.control_end, 1.0)),
        guidance=guidance,
        refine=body.refine,
        refine_strength=max(0.05, min(body.refine_strength, 0.6)),
        refine_steps=refine_steps,
        size=body.size,
        composition=composition,
        seed=body.seed,
        require_scan=body.require_scan,
        auto_escalate=body.auto_escalate,
        qr_monster_version=(
            body.qr_monster_version if body.qr_monster_version in QR_MONSTER_VERSIONS
            else QR_MONSTER_DEFAULT
        ),
        qr_coverage=max(0.40, min(body.qr_coverage, 1.0)),
        init_image_path=body.init_image_path,
        init_strength=max(0.05, min(body.init_strength, 0.95)),
        canny_scale=max(0.0, min(body.canny_scale, 1.5)),
        fast_mode=body.fast_mode,
        hires_fix=body.hires_fix,
        hires_target=max(768, min(body.hires_target, 1536)),
        hires_strength=max(0.05, min(body.hires_strength, 0.45)),
        adetailer=body.adetailer,
        adetailer_strength=max(0.1, min(body.adetailer_strength, 0.6)),
    )

    db = get_db()
    job_id = new_job_id()

    # Snapshot post-clamp values so the DB row reflects what diffused, not
    # the raw request body.
    persisted = {
        **body.model_dump(),
        "candidates": req.candidates,
        "steps": req.steps,
        "controlnet_scale": req.controlnet_scale,
        "tile_scale": req.tile_scale,
        "control_start": req.control_start,
        "control_end": req.control_end,
        "guidance": req.guidance,
        "refine_strength": req.refine_strength,
        "refine_steps": req.refine_steps,
        "hires_target": req.hires_target,
        "hires_strength": req.hires_strength,
        "adetailer_strength": req.adetailer_strength,
        "composition": composition,
        "qr_monster_version": req.qr_monster_version,
        "qr_coverage": req.qr_coverage,
        "init_image_path": req.init_image_path,
        "init_strength": req.init_strength,
        "canny_scale": req.canny_scale,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    db.insert_job(job_id, persisted)
    db.touch_prompt(body.prompt)

    job = Job(job_id=job_id, model=body.model, request=req, body=persisted)
    try:
        position = _worker.enqueue(job)
    except QueueFull:
        # Roll the row to 'failed' so the user sees why and the queue stays
        # consistent with what the DB shows.
        db.finish_job(
            job_id,
            status="failed",
            elapsed_s=0.0,
            error=f"queue at capacity ({MAX_QUEUED}); try again in a moment",
        )
        raise HTTPException(
            503,
            f"Queue full ({MAX_QUEUED} jobs). Wait for one to finish, then retry.",
        )

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": position,
        "queue_depth": _worker.state()["queue_depth"],
    }


@app.post("/api/jobs/{job_id}/rerun")
def rerun_job(
    job_id: str,
    request: Request,
    keep_seed: bool = False,
) -> dict[str, Any]:
    """One-click "Run again" — clones a past job's settings and enqueues it.

    keep_seed=true reproduces the exact run; default keep_seed=false nulls
    the seed so the worker rolls a fresh one and produces variations on the
    same recipe. The new job links back via parent_job_id for lineage.

    Distinct from "Remix" (a UI flow): Remix loads settings into the form
    so the user can edit before submitting; rerun is fire-and-forget.
    """
    db = get_db()
    src = db.get_job(job_id)
    if not src:
        raise HTTPException(404, f"job {job_id} not found")

    body = GenerateBody(
        data=src["data"],
        prompt=src["prompt"],
        style=src["style"],
        model=src["model"],
        negative_prompt=src.get("negative_prompt"),
        candidates=src["candidates"],
        steps=src["steps"],
        controlnet_scale=src["controlnet_scale"],
        tile_scale=src["tile_scale"],
        control_start=src.get("control_start") if src.get("control_start") is not None else 0.30,
        control_end=src["control_end"],
        guidance=src["guidance"],
        refine=bool(src["refine"]),
        refine_strength=src["refine_strength"],
        refine_steps=src["refine_steps"],
        size=src["size"],
        composition=src["composition"],
        seed=src["seed"] if keep_seed else None,
        require_scan=bool(src["require_scan"]),
        auto_escalate=bool(src.get("auto_escalate", 1)),
        qr_monster_version=src.get("qr_monster_version") or QR_MONSTER_DEFAULT,
        qr_coverage=src.get("qr_coverage") or 1.0,
        init_image_path=src.get("init_image_path"),
        init_strength=src.get("init_strength") or 0.65,
        canny_scale=src.get("canny_scale") or 0.0,
        fast_mode=bool(src["fast_mode"]),
        hires_fix=bool(src["hires_fix"]),
        hires_target=src["hires_target"],
        hires_strength=src["hires_strength"],
        adetailer=bool(src["adetailer"]),
        adetailer_strength=src["adetailer_strength"],
        parent_job_id=job_id,
    )
    return generate(body, request)


@app.delete("/api/jobs/{job_id}")
def cancel_or_delete_job(job_id: str) -> dict[str, Any]:
    """Smart DELETE: cancel if the job is in flight, hard-delete if terminal.

    - queued / running -> cancel (worker's cancel set; running jobs are
      caught at the next diffusion step boundary).
    - completed / failed / cancelled -> remove the row + cascade candidates
      and events, plus rm -rf outputs/{id}/ on disk.
    """
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")

    state = _worker.cancel(job_id)
    if state in ("queued", "running"):
        return {"job_id": job_id, "cancelled": True, "was": state}

    # Terminal state — hard delete row + files.
    db.delete_job(job_id)
    _cleanup_evicted_files([job_id])
    return {"job_id": job_id, "deleted": True, "was": job["status"]}


@app.get("/api/prompts/recent")
def list_recent_prompts(limit: int = 20, favorites_only: bool = False) -> dict[str, Any]:
    db = get_db()
    return {
        "prompts": db.list_prompts(
            limit=max(1, min(limit, 100)),
            favorites_only=favorites_only,
        ),
    }


class FavoriteBody(BaseModel):
    favorited: bool = True


@app.post("/api/prompts/{prompt_id}/favorite")
def set_favorite(prompt_id: int, body: FavoriteBody) -> dict[str, Any]:
    db = get_db()
    ok = db.set_prompt_favorite(prompt_id, body.favorited)
    if not ok:
        raise HTTPException(404, f"prompt {prompt_id} not found")
    return {"prompt_id": prompt_id, "favorited": body.favorited}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    db = get_db()
    return db.stats()


@app.post("/api/admin/cleanup")
def admin_cleanup(keep: int = 1000) -> dict[str, Any]:
    """Manual retention sweep. Defaults to the same retention as startup.
    Returns the evicted job count + how many output dirs were removed."""
    db = get_db()
    evicted = db.evict_old_jobs(keep=max(1, keep))
    removed = _cleanup_evicted_files(evicted)
    return {"evicted": len(evicted), "removed_dirs": removed, "keep": keep}


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE feed of job_events. The worker writes events into SQLite as the
    pipeline ticks; this handler polls the table at ~250 ms and forwards each
    new row as a server-sent event. Stream closes after a terminal event
    (completed/failed/cancelled) or when the client disconnects.
    """
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")

    async def gen():
        last_id = 0
        terminal = {"completed", "failed", "cancelled"}
        # Replay any events already on disk (e.g. UI reconnects mid-job).
        seen_terminal = False
        while not seen_terminal:
            if await request.is_disconnected():
                return
            events = db.events_since(job_id, after_id=last_id, limit=200)
            for ev in events:
                last_id = ev["id"]
                payload = {**ev["payload"], "ts": ev["ts"]}
                yield f"event: {ev['type']}\ndata: {_json.dumps(payload)}\n\n"
                if ev["type"] in terminal:
                    seen_terminal = True
                    break
            if seen_terminal:
                break
            # Also short-circuit if the DB row says the job is already done
            # (e.g. completed before the SSE handler attached).
            row = db.get_job(job_id)
            if row and row["status"] in terminal and not events:
                yield f"event: {row['status']}\ndata: {_json.dumps({'status': row['status']})}\n\n"
                seen_terminal = True
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs")
def list_jobs(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    model: str | None = None,
    scans: bool | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """Browse history. Filters: status, model, scans (bool), q (substring of
    prompt or decoded)."""
    db = get_db()
    rows = db.list_jobs(
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        status=status,
        model=model,
        scans=scans,
        q=q,
    )
    return {"jobs": rows, "count": len(rows)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Full job row + candidates. URL paths in the response point at the
    /outputs static mount so the UI can render them directly."""
    db = get_db()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")
    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
