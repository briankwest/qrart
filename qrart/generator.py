from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable
from PIL import Image

from .canvas import (
    build_composition,
    composite_qr_into_scene,
    is_standalone,
    reinforce_finders,
)
from .pipeline import QRArtPipeline
from .scannability import score as scannability_score
from .scanner import scan, scan_breakdown
from .styles import compose


@dataclass
class Progress:
    """Per-job progress emitter. The worker passes one to Generator.generate()
    so pipeline step callbacks can emit phase-aware events and check for
    cancellation. publish(type, payload) is the only side-effect — the worker
    wires it to db.insert_event so SSE subscribers can poll it out.

    on_candidate_ready (optional) lets the worker save the candidate image
    AS SOON as it's finished, so the UI can render candidates one-by-one
    instead of waiting for the whole job. Returns a dict merged into the
    candidate_done event payload (typically {"url": "/outputs/...", "pass1_url": ...}).
    """
    publish: Callable[[str, dict[str, Any]], None] | None = None
    is_cancelled: Callable[[], bool] | None = None
    on_candidate_ready: Callable[[int, Any], dict[str, Any]] | None = None
    total_candidates: int = 1
    candidate_idx: int = 0

    def emit(self, type_: str, **payload: Any) -> None:
        if self.publish is None:
            return
        self.publish(type_, payload)

    def step_cb(self, phase: str, total: int) -> Callable[[int], None]:
        def cb(step: int) -> None:
            self.emit(
                "step",
                phase=phase,
                candidate=self.candidate_idx,
                total_candidates=self.total_candidates,
                step=step,
                total_steps=total,
            )
        return cb

    @property
    def cancel_check(self) -> Callable[[], bool] | None:
        return self.is_cancelled


@dataclass
class GenerationRequest:
    data: str  # the URL/text to encode
    prompt: str
    style: str = "photoreal"
    negative_prompt: str | None = None  # overrides the style's default negative
    candidates: int = 5
    steps: int = 32
    guidance: float = 7.5
    # Photo-dominant band: 1.05–1.20. Below 1.05 the QR usually doesn't decode;
    # above 1.20 the grid pattern starts to dominate. With 5+ candidates, scan
    # rate at 1.10 is high enough to reliably get a winner.
    controlnet_scale: float = 1.10
    # Tile ControlNet stacked alongside QR Monster. 0 = off; 0.3-0.5 nudges
    # toward photo coherence at the cost of slightly weakened QR signal.
    tile_scale: float = 0.0
    # Control window: ControlNet only conditions diffusion between these
    # fractions of total steps. control_start=0.30 lets the first 30% of
    # denoising run as pure txt2img (paints the scene from the prompt
    # before any QR pull); control_end=0.95 releases ControlNet in the
    # final 5% so the diffusion can finish painting cleanly. This is the
    # QR Monster v2 community sweet spot — produces scenes that look
    # natural with the QR woven *into* them rather than forced *over* them.
    control_start: float = 0.30
    control_end: float = 0.95
    refine: bool = True
    refine_strength: float = 0.30  # polishes pass-1 without erasing the QR
    refine_steps: int = 20
    size: int = 768  # ignored when composition != "standalone" (canvas drives size)
    composition: str = "standalone"
    seed: int | None = None
    require_scan: bool = True
    # A2: see app.GenerateBody. Persisted on the request so reruns inherit it.
    auto_escalate: bool = True
    # QR Monster ControlNet version: 'v1' or 'v2'. v2 carries stronger QR
    # signal at the same scale, so users running v2 typically dial scale
    # down ~0.10 from their v1 settings.
    qr_monster_version: str = "v1"
    # Fraction of the diffusion canvas the QR code occupies. 1.0 = QR fills
    # the canvas (legacy default). <1.0 centers the QR with #808080 gray
    # padding around it; the diffusion paints prompt content in the margin
    # (the "QR-as-feature-in-a-scene" workflow). 0.70-0.80 produces the
    # tree-canopy / temple-cluster aesthetic.
    qr_coverage: float = 1.0
    # Init image (user upload). When set, pass-1 becomes ControlNet img2img
    # seeded with this image at (1 - init_strength) preservation. For
    # non-standalone compositions, the init image replaces the
    # auto-generated scene that the QR art is pasted into.
    init_image_path: str | None = None
    init_strength: float = 0.65
    # Canny ControlNet scale. > 0 with an init image present stacks Canny
    # edge conditioning into the multi-controlnet — useful for logo-shaped
    # QR codes where modules should cluster along the logo's outlines.
    # 0.5–0.8 typical; ≥ 1.0 starts to dominate over QR Monster.
    canny_scale: float = 0.0
    # Fast mode: swaps in LCM-LoRA + LCMScheduler. ~3–4x faster, slight fidelity drop.
    fast_mode: bool = False
    # Hi-res fix: upscale best candidate via Lanczos and run a low-strength
    # img2img pass for sharper detail. Only runs once on the winning candidate.
    hires_fix: bool = False
    hires_target: int = 1024
    hires_strength: float = 0.20
    hires_steps: int = 18
    # ADetailer: detect faces, re-render at 512x512 each, paste back. Same
    # "winner only" semantics as hi-res fix.
    adetailer: bool = False
    adetailer_strength: float = 0.35
    adetailer_steps: int = 20


@dataclass
class Candidate:
    image: Image.Image  # final image (refined if refine, else pass1)
    pass1_image: Image.Image | None  # pre-refine image, None when refine=False
    seed: int
    scans: bool
    decoded: str | None
    controlnet_scale: float
    refine_strength: float | None  # None when refine=False
    scannability: float = 0.0       # 0.0-1.0, fraction of correctly-resolved QR modules
    # Per-scanner decode results — used to derive the compatibility tier
    # (universal / phone-ready / ios-class / soft / none) for downstream UI
    # and gallery filtering. None means we never measured (shouldn't happen
    # on fresh candidates; existing rows from before migration 009 get
    # populated by scripts/backfill_scanner_breakdown.py).
    scans_cv2: bool | None = None
    scans_zxing: bool | None = None
    scans_qreader: bool | None = None


@dataclass
class GenerationResult:
    image: Image.Image
    qr_image: Image.Image
    seed: int
    scans: bool
    decoded: str | None
    controlnet_scale: float
    refine_strength: float | None
    candidates: list[Candidate] = field(default_factory=list)


def _refine_strengths(target: float) -> list[float]:
    return [target] if target <= 0.18 else [target, max(0.15, target - 0.1)]


def phone_scans(c) -> bool:
    """A candidate is "phone-readable" if either cv2 OR zxing decoded it.
    qreader-only successes are tolerable but don't count as phone-class —
    stock iOS Camera and most Android scanners use zxing-class decoders, so
    a candidate that only YOLO+libzbar reads will fail in users' hands.
    """
    return bool(getattr(c, "scans_cv2", False)) or bool(getattr(c, "scans_zxing", False))


def _load_init_image(url_path: str) -> Image.Image:
    """Resolve a /outputs/* URL (or absolute filesystem path) to a PIL
    image. Used to materialize user-uploaded init images at generation
    time. Returns RGB."""
    from pathlib import Path
    if url_path.startswith("/outputs/"):
        fs_path = Path(__file__).parent.parent / "outputs" / url_path[len("/outputs/"):]
    else:
        fs_path = Path(url_path)
    return Image.open(fs_path).convert("RGB")


def _canny_edges(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Run cv2.Canny on the image and return a 3-channel RGB edge map.
    Output is white edges on black — the Canny ControlNet's expected input
    format. low/high thresholds tuned for typical logos / cleaned photos;
    bump them up to drop minor edges, down to keep more detail."""
    import cv2
    import numpy as np
    arr = np.array(image.convert("L"))
    edges = cv2.Canny(arr, low, high)
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb)


def _fit_to(img: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop the image to the target aspect ratio, then resize to
    exactly (w, h). Preserves the visual subject when the user's image
    doesn't match the canvas aspect."""
    target_ratio = w / h
    iw, ih = img.size
    src_ratio = iw / ih
    if src_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    elif src_ratio < target_ratio:
        new_h = int(iw / target_ratio)
        top = (ih - new_h) // 2
        img = img.crop((0, top, iw, top + new_h))
    return img.resize((w, h), Image.LANCZOS)


def _score_for(image: Image.Image, data: str, comp) -> float:
    """Compute scannability against the inner QR rectangle defined by comp.
    Always passes qr_pos + qr_size now; with coverage < 1.0 they point to
    the inner QR inside a gray-padded diffusion canvas."""
    return scannability_score(
        image, data, qr_pos=comp.qr_pos, qr_size=comp.qr_size,
    )


class Generator:
    def __init__(self, base_model: str | None = None):
        self.pipeline = QRArtPipeline(base_model=base_model)

    def warm(self) -> None:
        self.pipeline.load()

    def generate(
        self,
        req: GenerationRequest,
        progress: Progress | None = None,
    ) -> GenerationResult:
        # Composition scaffold goes in BEFORE style preset so the style suffix
        # (RAW photo, 8k, etc.) lands at the very end where SD weighs it most.
        comp = build_composition(
            req.data, req.composition, req.qr_monster_version, req.qr_coverage,
        )
        full_prompt = req.prompt + comp.scaffold
        prompt, negative = compose(full_prompt, req.style, req.negative_prompt)

        # Apply Fast/Quality mode once per request so all candidates use the
        # same scheduler/LoRA state. Pin the QR Monster version too — both
        # are loaded so this is just a pointer swap.
        self.pipeline.set_qr_monster_version(req.qr_monster_version)
        self.pipeline.set_fast_mode(req.fast_mode)
        candidates: list[Candidate] = []
        rng = random.Random(req.seed)

        progress = progress or Progress()
        progress.total_candidates = max(1, req.candidates)

        for i in range(max(1, req.candidates)):
            seed = rng.randrange(2**31)
            progress.candidate_idx = i
            progress.emit("candidate_started", idx=i, seed=seed)
            cand = self._make_candidate(req, comp, seed, prompt, negative, progress)
            candidates.append(cand)
            # Hand the candidate to the worker for immediate persistence
            # so the UI can render it. Failures here MUST NOT abort the
            # job — log and continue (the worker's end-of-job save path
            # will still cover them).
            extra: dict[str, Any] = {}
            if progress.on_candidate_ready is not None:
                try:
                    extra = progress.on_candidate_ready(i, cand) or {}
                except Exception as e:
                    print(f"[gen] on_candidate_ready failed for {i}: {e}", flush=True)
            progress.emit(
                "candidate_done",
                idx=i,
                seed=cand.seed,
                scans=cand.scans,
                decoded=cand.decoded,
                controlnet_scale=cand.controlnet_scale,
                refine_strength=cand.refine_strength,
                scannability=cand.scannability,
                scans_cv2=cand.scans_cv2,
                scans_zxing=cand.scans_zxing,
                scans_qreader=cand.scans_qreader,
                **extra,
            )

        # Best-candidate sort, prioritized:
        #   1. phone_scans (cv2 OR zxing) — what stock phones can actually read
        #   2. any-scanner scans (covers qreader-only fallbacks)
        #   3. scannability score (closer to scannable = better near-miss)
        #   4. lowest controlnet_scale (= least visible QR)
        # A qreader-only candidate now loses to any phone-readable one
        # regardless of score, fixing the "looks like it scans but doesn't"
        # bug where outputs were tagged ✓ but consumer scanners couldn't read.
        best = sorted(
            candidates,
            key=lambda c: (
                0 if phone_scans(c) else 1,
                0 if c.scans else 1,
                -c.scannability,
                c.controlnet_scale,
            ),
        )[0]
        best_idx = candidates.index(best)

        # C1: cheap in-generation rescue. Fires when the best candidate
        # isn't PHONE-READABLE (cv2/zxing), even if qreader decoded it —
        # otherwise the user gets an output that looks scannable but their
        # iPhone Camera can't read. score >= 0.70 means there's something
        # to nudge over the line; below that, no scale bump will help.
        if not phone_scans(best) and best.scannability >= 0.70 and req.controlnet_scale < 1.5:
            rescue_scale = round(req.controlnet_scale + 0.10, 2)
            progress.emit(
                "rescue_started",
                from_score=round(best.scannability, 3),
                from_scale=req.controlnet_scale,
                to_scale=rescue_scale,
            )
            rescue_req = GenerationRequest(**{
                **req.__dict__,
                "controlnet_scale": rescue_scale,
                "candidates": 1,
                "seed": best.seed + 1,
            })
            rescue_progress = Progress(
                publish=progress.publish,
                is_cancelled=progress.is_cancelled,
                # Show the rescue as one extra candidate beyond the original
                # set so the UI's progress bar makes sense.
                total_candidates=len(candidates) + 1,
                candidate_idx=len(candidates),
            )
            rescue = self._make_candidate(
                rescue_req, comp, best.seed + 1, prompt, negative, rescue_progress,
            )
            candidates.append(rescue)
            # Persist the rescue immediately so it's saved + DB-inserted like
            # the regular candidates. Without this it'd only be saved via
            # the post-generate backstop loop, which doesn't run on cancel.
            if progress.on_candidate_ready is not None:
                try:
                    progress.on_candidate_ready(len(candidates) - 1, rescue)
                except Exception as e:
                    print(f"[gen] on_candidate_ready (rescue) failed: {e}", flush=True)
            progress.emit(
                "rescue_done",
                scans=rescue.scans,
                score=round(rescue.scannability, 3),
            )
            # Re-pick best including the rescue candidate. Same phone-first
            # priority as the initial sort.
            best = sorted(
                candidates,
                key=lambda c: (
                    0 if phone_scans(c) else 1,
                    0 if c.scans else 1,
                    -c.scannability,
                    c.controlnet_scale,
                ),
            )[0]
            best_idx = candidates.index(best)

        # Finishing passes — only run on the winner so we don't pay 3x for
        # them. Each pass re-scans; if it kills scannability we keep the
        # post-processed image but report scans=False so the UI can flag it.
        if req.hires_fix or req.adetailer:
            final = best.image
            if req.hires_fix:
                progress.emit("phase", phase="hires", candidate=best_idx)
                final = self.pipeline.hires_fix(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    target_size=req.hires_target,
                    strength=req.hires_strength,
                    steps=req.hires_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                    step_callback=progress.step_cb("hires", req.hires_steps),
                    cancel_check=progress.cancel_check,
                )
            if req.adetailer:
                progress.emit("phase", phase="adetailer", candidate=best_idx)
                final = self.pipeline.adetailer_faces(
                    image=final,
                    prompt=prompt,
                    negative_prompt=negative,
                    strength=req.adetailer_strength,
                    steps=req.adetailer_steps,
                    guidance=req.guidance,
                    seed=best.seed,
                    step_callback=progress.step_cb("adetailer", req.adetailer_steps),
                    cancel_check=progress.cancel_check,
                )
            br = scan_breakdown(final)
            decoded = next(
                (v for v in br.values() if v == req.data),
                next((v for v in br.values() if v), None),
            )
            best = Candidate(
                image=final,
                pass1_image=best.pass1_image,
                seed=best.seed,
                scans=decoded == req.data,
                decoded=decoded,
                controlnet_scale=best.controlnet_scale,
                refine_strength=best.refine_strength,
                scannability=_score_for(final, req.data, comp),
                scans_cv2=br["cv2"] == req.data,
                scans_zxing=br["zxing"] == req.data,
                scans_qreader=br["qreader"] == req.data,
            )
            candidates[best_idx] = best

        return GenerationResult(
            image=best.image,
            qr_image=comp.qr_image,
            seed=best.seed,
            scans=best.scans,
            decoded=best.decoded,
            controlnet_scale=best.controlnet_scale,
            refine_strength=best.refine_strength,
            candidates=candidates,
        )

    def _make_candidate(
        self,
        req: GenerationRequest,
        comp,
        seed: int,
        prompt: str,
        negative: str,
        progress: Progress,
    ) -> Candidate:
        # Pre-load the user's init image once (used by both pass-1 init and
        # the composition scene replacement below). Center-crop+resize to the
        # respective target rectangle so it matches the diffusion canvas.
        init_image: Image.Image | None = None
        if req.init_image_path:
            try:
                init_image = _load_init_image(req.init_image_path)
            except Exception as e:
                progress.emit("init_image_failed", reason=str(e))
                init_image = None

        # Compute Canny edges once (qr_size resolution — pipeline resizes
        # to width/height per call). Only when both an init image is loaded
        # AND canny_scale > 0 so we skip the cv2 work on pure txt2img runs.
        canny_image: Image.Image | None = None
        canny_scale = max(0.0, float(req.canny_scale))
        if init_image is not None and canny_scale > 0:
            sized = _fit_to(init_image, comp.diffusion_size, comp.diffusion_size)
            canny_image = _canny_edges(sized)

        # Pass-1: either txt2img + ControlNet (default), or img2img +
        # ControlNet when the user supplied an init image. For non-standalone
        # compositions the init image becomes the scene (below), NOT the QR
        # art's init — the QR art has a dedicated qr_size×qr_size canvas
        # where the user's image would be cropped to nothing meaningful.
        progress.emit("phase", phase="pass1", candidate=progress.candidate_idx)
        use_init_for_pass1 = init_image is not None and is_standalone(req.composition)
        if use_init_for_pass1:
            assert init_image is not None
            init_for_pass1 = _fit_to(init_image, comp.diffusion_size, comp.diffusion_size)
            qr_pass1 = self.pipeline.generate_pass1_from_init(
                init_image=init_for_pass1,
                qr_image=comp.qr_image,
                prompt=prompt,
                negative_prompt=negative,
                steps=req.steps,
                guidance=req.guidance,
                controlnet_scale=req.controlnet_scale,
                tile_scale=req.tile_scale,
                strength=req.init_strength,
                seed=seed,
                width=comp.diffusion_size,
                height=comp.diffusion_size,
                control_start=req.control_start,
                control_end=req.control_end,
                canny_image=canny_image,
                canny_scale=canny_scale,
                step_callback=progress.step_cb("pass1", req.steps),
                cancel_check=progress.cancel_check,
            )
        else:
            qr_pass1 = self.pipeline.generate_pass1(
                qr_image=comp.qr_image,
                prompt=prompt,
                negative_prompt=negative,
                steps=req.steps,
                guidance=req.guidance,
                controlnet_scale=req.controlnet_scale,
                tile_scale=req.tile_scale,
                control_start=req.control_start,
                control_end=req.control_end,
                seed=seed,
                width=comp.diffusion_size,
                height=comp.diffusion_size,
                canny_image=canny_image,
                canny_scale=canny_scale,
                step_callback=progress.step_cb("pass1", req.steps),
                cancel_check=progress.cancel_check,
            )

        # Scene: either user-supplied (composition + init), or generated.
        # Different seed offset so the scene RNG isn't correlated with the
        # QR art RNG.
        scene: Image.Image | None = None
        if not is_standalone(req.composition):
            if init_image is not None:
                scene = _fit_to(init_image, comp.canvas_w, comp.canvas_h)
                progress.emit("phase", phase="scene_from_init", candidate=progress.candidate_idx)
            else:
                progress.emit("phase", phase="scene", candidate=progress.candidate_idx)
                scene = self.pipeline.generate_scene(
                    prompt=prompt,
                    negative_prompt=negative,
                    steps=req.steps,
                    guidance=req.guidance,
                    seed=seed + 9001,
                    width=comp.canvas_w,
                    height=comp.canvas_h,
                    step_callback=progress.step_cb("scene", req.steps),
                    cancel_check=progress.cancel_check,
                )

        def composite(qr_art: Image.Image) -> Image.Image:
            """Build the un-reinforced final image. Standalone returns the
            raw QR art; compositions paste it into the scene with the
            finder-aware mask + quiet-zone pad but WITHOUT pasting the
            ground-truth finder squares on top — that's the rescue path."""
            if scene is None:
                return qr_art
            return composite_qr_into_scene(
                scene, qr_art, req.composition, data=req.data,
                reinforce_finders_flag=False,
                diffusion_pos=comp.diffusion_pos,
                diffusion_size=comp.diffusion_size,
                qr_pos=comp.qr_pos,
                qr_size=comp.qr_size,
            )

        def composite_and_scan(
            qr_art: Image.Image,
        ) -> tuple[Image.Image, str | None, dict[str, str | None]]:
            """Composite, scan with the full 3-way breakdown, and rescue with
            finder reinforcement when needed. Returns (image, decoded,
            breakdown). breakdown is {"cv2": .., "zxing": .., "qreader": ..}
            with each entry being the decoded URL or None.
            """
            final = composite(qr_art)
            br = scan_breakdown(final)
            # Match if ANY scanner decoded the right URL (existing behavior).
            decoded = next(
                (v for v in br.values() if v == req.data),
                next((v for v in br.values() if v), None),
            )
            if decoded != req.data:
                rescued = reinforce_finders(
                    final, req.data, comp.qr_pos, comp.qr_size,
                )
                br_rescued = scan_breakdown(rescued)
                rescued_decoded = next(
                    (v for v in br_rescued.values() if v == req.data),
                    next((v for v in br_rescued.values() if v), None),
                )
                if rescued_decoded == req.data:
                    return rescued, rescued_decoded, br_rescued
            return final, decoded, br

        if not req.refine:
            final, decoded, br = composite_and_scan(qr_pass1)
            return Candidate(
                image=final,
                pass1_image=None,
                seed=seed,
                scans=decoded == req.data,
                decoded=decoded,
                controlnet_scale=req.controlnet_scale,
                refine_strength=None,
                scannability=_score_for(final, req.data, comp),
                scans_cv2=br["cv2"] == req.data,
                scans_zxing=br["zxing"] == req.data,
                scans_qreader=br["qreader"] == req.data,
            )

        # Refine the QR art (not the composite — scene doesn't need it). Try
        # each strength in the ladder; first one that scans wins.
        last: Candidate | None = None
        pass1_composite = composite(qr_pass1)
        for strength in _refine_strengths(req.refine_strength):
            progress.emit(
                "phase", phase="refine", candidate=progress.candidate_idx,
                strength=strength,
            )
            qr_refined = self.pipeline.refine(
                image=qr_pass1,
                qr_image=comp.qr_image,
                controlnet_scale=req.controlnet_scale,
                tile_scale=req.tile_scale,
                prompt=prompt,
                negative_prompt=negative,
                strength=strength,
                steps=req.refine_steps,
                guidance=req.guidance,
                seed=seed,
                control_start=req.control_start,
                control_end=req.control_end,
                canny_image=canny_image,
                canny_scale=canny_scale,
                step_callback=progress.step_cb("refine", req.refine_steps),
                cancel_check=progress.cancel_check,
            )
            final, decoded, br = composite_and_scan(qr_refined)
            ok = decoded == req.data
            cand = Candidate(
                image=final,
                pass1_image=pass1_composite,
                seed=seed,
                scans=ok,
                decoded=decoded,
                controlnet_scale=req.controlnet_scale,
                refine_strength=strength,
                scannability=_score_for(final, req.data, comp),
                scans_cv2=br["cv2"] == req.data,
                scans_zxing=br["zxing"] == req.data,
                scans_qreader=br["qreader"] == req.data,
            )
            if ok:
                return cand
            last = cand

        assert last is not None
        return last
