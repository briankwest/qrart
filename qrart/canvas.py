"""Composition canvases — generate the QR as a feature in a larger scene by:

  1. Generating the scene at canvas dims via txt2img (no ControlNet).
  2. Generating the QR art separately at QR-region dims using the standalone
     ControlNet path (full-canvas QR pattern as control image — matches
     QR Monster's training distribution).
  3. Compositing the QR art into the scene with a finder-aware alpha mask
     (corners hard, BR + interior edges feathered) plus an alpha-blended
     reinforcement of the three ground-truth finder patterns to lock in
     scannability.

The earlier two-stage inpaint approach failed because QR Monster ControlNet
was trained on full-canvas QR patterns, not partial ones — feeding it a
scene-sized canvas with QR-in-corner produced a weak conditioning signal
that the prompt + scene-init overrode, so the masked region rendered as
"more scene" instead of QR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .qr import make_qr, qr_modules


class Composition(TypedDict):
    canvas_size: tuple[int, int]  # (width, height)
    qr_size: int                  # square QR side length, px
    qr_pos: tuple[int, int]       # top-left corner inside canvas
    scaffold: str                 # appended to user's prompt to bias both scene + QR-art


# Canvas dims sized for SD 1.5 finetunes. Going much past 1024 in any axis on
# MPS is slow and degrades quality. qr_size bumped from 600 → 720 in
# scene-landscape and garment so module-pixel density doesn't fall below
# scanner thresholds.
COMPOSITIONS: dict[str, Composition] = {
    "standalone": {
        "canvas_size": (768, 768),
        "qr_size": 768,
        "qr_pos": (0, 0),
        "scaffold": "",
    },
    "subject-portrait": {
        # Subject above, QR-as-feature below. Astronaut on a patterned platform.
        "canvas_size": (768, 1024),
        "qr_size": 640,
        "qr_pos": (64, 320),
        "scaffold": ", with an intricate decorative patterned ornament featured below, cinematic photorealistic composition",
    },
    "scene-landscape": {
        # Wide scene with a QR-as-stone-monolith feature on the right.
        "canvas_size": (1024, 768),
        "qr_size": 720,
        "qr_pos": (288, 24),
        "scaffold": ", featuring an ancient stone monolith covered in intricate carved geometric patterns on the right, dramatic landscape composition, photorealistic",
    },
    "garment": {
        # Fashion shot — the QR is the patterned outfit centerpiece.
        "canvas_size": (768, 1024),
        "qr_size": 720,
        "qr_pos": (24, 280),
        "scaffold": ", wearing an intricate richly patterned ornate ceremonial garment with bold geometric design, fashion photograph",
    },
}


# B3 reinforcement: blend strength of the ground-truth finder pattern overlay.
# 0.85 keeps the texture readable underneath while giving scanners a pristine
# corner. Tuned empirically — anything < 0.7 doesn't reliably push borderline
# candidates over the threshold, anything > 0.92 starts looking pasted-on.
FINDER_BLEND_ALPHA = 0.85


@dataclass
class CompositionInfo:
    canvas_w: int
    canvas_h: int
    # Inner QR rectangle inside the scene canvas — used by scannability score
    # and finder reinforcement (where the ACTUAL QR modules live).
    qr_size: int
    qr_pos: tuple[int, int]
    # Diffusion canvas — what gets fed to ControlNet and what gets pasted
    # into the scene. When qr_coverage < 1.0 this is larger than qr_size:
    # the actual QR sits in the center with #808080 gray padding around it,
    # giving the diffusion freedom to paint prompt content in the margin.
    # When coverage = 1.0 (default), diffusion_size == qr_size.
    diffusion_size: int
    diffusion_pos: tuple[int, int]
    qr_image: Image.Image  # diffusion-canvas-sized control input (may include gray margin)
    scaffold: str


QR_MARGIN_GRAY = (128, 128, 128)  # #808080 — see build_composition()


def build_composition(
    data: str,
    name: str,
    qr_monster_version: str = "v1",
    qr_coverage: float = 1.0,
) -> CompositionInfo:
    """Build the composition layout + ground-truth QR control image.

    qr_coverage < 1.0 renders the QR at (cfg.qr_size * coverage) and centers
    it on a (cfg.qr_size × cfg.qr_size) #808080 gray canvas. The diffusion
    sees strong QR conditioning in the central region and a weak/neutral
    signal in the gray margin — so the prompt's scene content can grow
    naturally around the QR (sky, ground, buildings, sub-foreground) instead
    of being forced into QR-pattern shapes everywhere.

    This is the "QR as a feature in a scene" workflow the QR Monster v2
    community uses. It's what produces the iconic tree-canopy-as-QR and
    temple-domes-as-QR aesthetics that pure full-canvas QR can't reach.

    qr_pos and qr_size are updated to point to the inner QR rectangle so
    that finder reinforcement, scannability scoring, and composition paste
    logic all operate on the actual QR area — not the padded canvas.
    """
    cfg = COMPOSITIONS.get(name, COMPOSITIONS["standalone"])
    cw, ch = cfg["canvas_size"]

    coverage = max(0.40, min(qr_coverage, 1.0))
    diffusion_size = cfg["qr_size"]
    inner_qr_size = int(diffusion_size * coverage)
    # Snap to multiple of 8 — keeps the gray border aligned and avoids
    # sub-pixel resize artifacts on the QR's modules.
    inner_qr_size = max(64, (inner_qr_size // 8) * 8)
    inner_offset = (diffusion_size - inner_qr_size) // 2

    qr_inner = make_qr(data, size=inner_qr_size)
    if inner_qr_size == diffusion_size:
        qr_image = qr_inner
    else:
        qr_image = Image.new("RGB", (diffusion_size, diffusion_size), QR_MARGIN_GRAY)
        qr_image.paste(qr_inner, (inner_offset, inner_offset))

    diffusion_pos = cfg["qr_pos"]
    inner_qr_pos = (diffusion_pos[0] + inner_offset, diffusion_pos[1] + inner_offset)

    return CompositionInfo(
        canvas_w=cw,
        canvas_h=ch,
        qr_size=inner_qr_size,
        qr_pos=inner_qr_pos,
        diffusion_size=diffusion_size,
        diffusion_pos=diffusion_pos,
        qr_image=qr_image,
        scaffold=cfg["scaffold"],
    )


def is_standalone(name: str) -> bool:
    return name == "standalone"


def _finder_aware_mask(qsz: int, feather_px: int) -> Image.Image:
    """Alpha mask sized (qsz, qsz). The three finder-pattern corners
    (TL/TR/BL) are kept fully opaque with NO feather — those corners are
    the most fragile QR feature; even a 4-px blur can break detection.
    The bottom-right and the interior edges get the feather.

    Implementation: start with a fully-opaque rectangle, blur it (this
    feathers all four edges equally), then paint hard-opaque squares back
    over the three finder corners.
    """
    if feather_px <= 0:
        return Image.new("L", (qsz, qsz), 255)

    # Start with an inset rectangle so the blur produces a gradient.
    alpha = Image.new("L", (qsz, qsz), 0)
    ImageDraw.Draw(alpha).rectangle(
        (feather_px, feather_px, qsz - feather_px - 1, qsz - feather_px - 1),
        fill=255,
    )
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Re-impose hard opacity over the three finder-pattern regions. The
    # finder is 7 modules; a QR with default border=1 places it 1 module
    # in. A typical QR is 25-49 modules, so each module is qsz / (n+2)
    # pixels. Conservatively reserve 9 modules' worth around each corner
    # (7 finder + 2 buffer) so we cover the separator too.
    # We don't know n at this point but ~50 px is safe for qsz >= 600.
    margin = max(50, qsz // 8)
    draw = ImageDraw.Draw(alpha)
    # Top-left
    draw.rectangle((0, 0, margin - 1, margin - 1), fill=255)
    # Top-right
    draw.rectangle((qsz - margin, 0, qsz - 1, margin - 1), fill=255)
    # Bottom-left
    draw.rectangle((0, qsz - margin, margin - 1, qsz - 1), fill=255)
    return alpha


def reinforce_finders(
    image: Image.Image,
    data: str,
    qr_pos: tuple[int, int],
    qr_size: int,
    alpha: float = FINDER_BLEND_ALPHA,
) -> Image.Image:
    """Alpha-blend the ground-truth finder patterns over the three corners.

    The QR finder is a 7×7-module square. We render those three regions
    from the ground-truth grid at the same scale as the QR art and blend
    them in at `alpha` opacity. This is mostly imperceptible at viewing
    distance but gives camera scanners a pristine 1:1:3:1:1 ratio to lock
    onto — the single most reliable feature for QR detection.
    """
    modules = qr_modules(data)
    n = modules.shape[0]
    border = 1
    total = n + 2 * border
    px_per_module = qr_size / total
    finder_modules = 7

    # Module-grid positions of the three finders (top-left module, in the
    # n×n grid without border). Ordered TL, TR, BL.
    finder_corners = [(0, 0), (0, n - finder_modules), (n - finder_modules, 0)]

    pixels_per_finder = int(round(finder_modules * px_per_module))
    qx, qy = qr_pos

    out = image.copy().convert("RGB")
    out_arr = np.array(out, dtype=np.float32)

    for fi, fj in finder_corners:
        # Pixel position of the finder's top-left within the QR region.
        py = int(round((border + fi) * px_per_module)) + qy
        px = int(round((border + fj) * px_per_module)) + qx

        # Render the 7×7 finder pattern at the right pixel size.
        finder_grid = modules[fi:fi + finder_modules, fj:fj + finder_modules]
        finder_img = Image.fromarray(
            np.where(finder_grid, 0, 255).astype(np.uint8)
        ).resize((pixels_per_finder, pixels_per_finder), Image.NEAREST)
        finder_arr = np.array(finder_img.convert("RGB"), dtype=np.float32)

        # Clip to image bounds (in case a corner is right at the edge).
        h, w = out_arr.shape[:2]
        y1 = min(h, py + pixels_per_finder)
        x1 = min(w, px + pixels_per_finder)
        fy = y1 - py
        fx = x1 - px
        if fy <= 0 or fx <= 0:
            continue

        # Blend: out = (1-alpha)*out + alpha*finder
        target = out_arr[py:y1, px:x1]
        finder_slice = finder_arr[:fy, :fx]
        out_arr[py:y1, px:x1] = (1 - alpha) * target + alpha * finder_slice

    return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))


QUIET_ZONE_PX = 8  # B2: light-ring pad between scene and QR modules


def _quiet_zone_pad(qr_art: Image.Image, pad: int) -> Image.Image:
    """Wrap the QR art in a `pad`-pixel light ring (sampled from the QR's own
    light-module luminance so the ring matches the art's tonal palette, not
    pure white which would look pasted-on).
    """
    if pad <= 0:
        return qr_art
    src = qr_art.convert("RGB")
    arr = np.array(src)
    # Sample the brightest 5% of pixels in the QR art for the ring color —
    # these are the "light modules" so the ring tonally matches them.
    luma = arr.mean(axis=2)
    if luma.size == 0:
        ring_color = (255, 255, 255)
    else:
        thr = np.quantile(luma, 0.95)
        bright = arr[luma >= thr]
        ring_color = tuple(int(v) for v in bright.mean(axis=0)) if bright.size else (255, 255, 255)
    w, h = src.size
    out = Image.new("RGB", (w + 2 * pad, h + 2 * pad), ring_color)
    out.paste(src, (pad, pad))
    return out


def composite_qr_into_scene(
    scene: Image.Image,
    qr_art: Image.Image,
    composition: str,
    feather_px: int = 4,
    data: str | None = None,
    reinforce_finders_flag: bool = True,
    quiet_zone_px: int = QUIET_ZONE_PX,
    diffusion_pos: tuple[int, int] | None = None,
    diffusion_size: int | None = None,
    qr_pos: tuple[int, int] | None = None,
    qr_size: int | None = None,
) -> Image.Image:
    """Paste qr_art into scene with a finder-aware alpha mask, a quiet-zone
    pad ring, and optional finder reinforcement.

    The diffusion canvas (qr_art, which is generated at diffusion_size and
    may include a gray margin around the actual QR) is pasted at
    diffusion_pos. Finder reinforcement, when enabled, targets the INNER
    qr_pos/qr_size — the actual QR rectangle. With coverage = 1.0 the inner
    rectangle equals the diffusion rectangle.

    For backward compatibility, when the explicit overrides aren't passed
    the function falls back to the COMPOSITIONS dict (which is correct only
    when coverage = 1.0). Callers using the QR coverage feature MUST pass
    the explicit values from CompositionInfo.
    """
    cfg = COMPOSITIONS.get(composition, COMPOSITIONS["standalone"])
    if diffusion_pos is None:
        diffusion_pos = cfg["qr_pos"]
    if diffusion_size is None:
        diffusion_size = cfg["qr_size"]
    if qr_pos is None:
        qr_pos = diffusion_pos
    if qr_size is None:
        qr_size = diffusion_size

    out = scene.copy().convert("RGB")
    # Quiet-zone pad applies to the WHOLE diffusion canvas (including any
    # gray margin) before paste; the inner QR's own quiet zone is handled
    # by the gray margin when coverage < 1.0.
    padded = _quiet_zone_pad(qr_art, quiet_zone_px)
    qr_resized = padded.resize((diffusion_size, diffusion_size)).convert("RGB")
    out.paste(qr_resized, diffusion_pos, _finder_aware_mask(diffusion_size, feather_px))

    if reinforce_finders_flag and data is not None:
        out = reinforce_finders(out, data, qr_pos, qr_size)

    return out
