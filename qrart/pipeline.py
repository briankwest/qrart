from __future__ import annotations

import os
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
)
from PIL import Image
from typing import Callable

import os as _os

# QR Monster ControlNet — trained for hiding QR codes inside generated images.
# The repo ships both v1 (root) and v2 (subfolder). Both are loaded at warm
# time and swappable per-request: v2 produces slightly stronger QR signal at
# the same scale slider, so at v2 you can run scale ~0.95-1.05 for the same
# scan rate v1 needs 1.10-1.20 for. The active version is selected on the
# generate request (qr_monster_version) and swapped into pipe.controlnet
# without rebuilding the pipe.
#
# QRART_MONSTER_VERSION sets the *default* when no version is requested.
_MONSTER_REPO = "monster-labs/control_v1p_sd15_qrcode_monster"
QR_MONSTER_VERSIONS = ("v1", "v2")
QR_MONSTER_DEFAULT = _os.environ.get("QRART_MONSTER_VERSION", "v1").lower()
if QR_MONSTER_DEFAULT not in QR_MONSTER_VERSIONS:
    QR_MONSTER_DEFAULT = "v1"
# Back-compat alias for callers that imported the old name.
QR_MONSTER_VERSION = QR_MONSTER_DEFAULT
CONTROLNET_ID = _MONSTER_REPO


def _qr_monster_subfolder(version: str) -> str | None:
    return "v2" if version == "v2" else None


# Sampler choice. Euler-Ancestral is the default — numerically stable on MPS
# even in fp16. DPM++ 2M Karras (non-SDE) is the community standard for QR
# Monster outputs; produces sharper detail at the same step count. Should be
# MPS-safe but hasn't been battle-tested on every model in our list, so it's
# opt-in via env. SDE variants of DPM++ are known NaN sources on MPS — we
# don't expose those.
QRART_SAMPLER = _os.environ.get("QRART_SAMPLER", "euler_a").lower()
SAMPLER_CHOICES = ("euler_a", "dpm_karras")
if QRART_SAMPLER not in SAMPLER_CHOICES:
    QRART_SAMPLER = "euler_a"


def _build_scheduler(config):
    if QRART_SAMPLER == "dpm_karras":
        return DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    return EulerAncestralDiscreteScheduler.from_config(config)

# Tile ControlNet — when stacked alongside QR Monster, it adds a coherence /
# detail-preservation signal that pushes outputs toward photo (less QR-noisy)
# at no significant runtime cost. Default tile_scale=0 keeps it dormant.
CONTROLNET_TILE_ID = "lllyasviel/control_v11f1e_sd15_tile"

# Canny edge ControlNet — used when the user supplies an init image and
# wants its structure preserved (e.g. logo-shaped QR codes). Compute
# cv2.Canny() of the init image at request time, feed as the third
# control input. When canny_scale=0 it's effectively off; a blank image
# is fed so the multi-controlnet still has a valid third slot.
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"

# External VAE that produces sharper, higher-contrast outputs than the bundled one.
VAE_ID = "stabilityai/sd-vae-ft-mse"

# Latent Consistency Model LoRA — turns SD 1.5 into a 4–8 step generator.
# ~3–4x speedup at the cost of some fidelity. Used by the Fast mode toggle.
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

MODELS = {
    # All SD 1.5 finetunes — same VAE + QR Monster ControlNet. Differences are
    # aesthetic: warm/cool palette, soft/sharp focus, photo-y vs stylized.
    # Models ordered roughly by photo-dominance (top = strongest photo prior,
    # most resistant to ControlNet override at low scales).
    "photoreal": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "photoreal-v51": "SG161222/Realistic_Vision_V5.1_noVAE",
    "majicmix": "digiplay/majicMIX_realistic_v6",
    "epicphoto": "Yntec/epiCPhotoGasm",
    "cyberrealistic": "Yntec/CyberRealistic",
    "hyperrealism": "Yntec/HyperRealism",
    "absolute-v18": "digiplay/AbsoluteReality_v1.8.1",
    "photon": "digiplay/Photon_v1",
    "epic": "emilianJR/epiCRealism",
    "absolute": "Lykon/AbsoluteReality",
    "analog": "wavymulder/Analog-Diffusion",
    "dreamlike": "dreamlike-art/dreamlike-photoreal-2.0",
    "openjourney": "prompthero/openjourney-v4",
    "dreamshaper": "Lykon/dreamshaper-8",
}
DEFAULT_BASE_MODEL = MODELS["photoreal"]


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


class CancelledByUser(Exception):
    """Raised by the diffusers step callback when the worker has flagged the
    job for cancellation. _run_job catches this specifically and marks the
    DB row 'cancelled' instead of 'failed'."""


StepCallback = Callable[[int], None]
CancelCheck = Callable[[], bool]


def _make_diffusers_callback(
    step_cb: StepCallback | None,
    cancel_check: CancelCheck | None,
):
    """Adapt our simple (step,) callback to diffusers' callback_on_step_end
    signature (pipe, step_index, timestep, kwargs) -> kwargs.

    Returns None if neither callback is provided so we can omit the param.
    """
    if step_cb is None and cancel_check is None:
        return None

    def adapter(pipe, step_index, timestep, callback_kwargs):
        if cancel_check is not None and cancel_check():
            raise CancelledByUser()
        if step_cb is not None:
            try:
                step_cb(step_index + 1)
            except CancelledByUser:
                raise
            except Exception:
                # Never let a publishing error abort the diffusion run.
                pass
        return callback_kwargs

    return adapter


def _prep_canny_input(
    image: Image.Image | None,
    width: int,
    height: int,
) -> Image.Image:
    """Return an (width, height) image suitable for the Canny ControlNet slot.

    If `image` is None, returns a black image — combined with canny_scale=0
    on the caller side, this is effectively a no-op slot. If `image` is
    provided, it's assumed to already be Canny edges (computed once by the
    generator) and just gets resized."""
    if image is None:
        return Image.new("RGB", (width, height), (0, 0, 0))
    return image.convert("RGB").resize((width, height))


def pick_device() -> tuple[str, torch.dtype]:
    # MPS defaults to fp32 — Realistic Vision V6 produces NaN in fp16 on Apple
    # Silicon. Set QRART_MPS_FP16=1 to opt in if your model tolerates it.
    if torch.backends.mps.is_available():
        if os.environ.get("QRART_MPS_FP16") == "1":
            return "mps", torch.float16
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class QRArtPipeline:
    """Two-pass diffusion pipeline.

    Pass 1: ControlNet pipe plants the QR pattern into a generated image.
    Pass 2: Img2Img pipe refines that QR-textured output toward photorealism
    while preserving enough structure to keep the code scannable.

    For non-standalone compositions, the scene_pipe (txt2img, no ControlNet)
    generates a full-canvas scene, the QR art is generated via the standalone
    Pass 1 path at QR-region size, and the two are composited in canvas.py.
    """

    def __init__(
        self,
        base_model: str | None = None,
        use_external_vae: bool = True,
    ) -> None:
        self.device, self.dtype = pick_device()
        self.base_model = resolve_model(base_model) if base_model else DEFAULT_BASE_MODEL
        self.use_external_vae = use_external_vae
        self._pipe: StableDiffusionControlNetPipeline | None = None
        self._refiner: StableDiffusionImg2ImgPipeline | None = None
        # ControlNet-aware refine pipe. Shares all weights with _pipe so it
        # costs zero extra memory — img2img + the same multi-controlnet
        # (QR + Tile) keeps the QR pattern locked while img2img polishes
        # toward photoreal. Without this, the plain _refiner erodes the QR
        # pattern (~5-15% of pixels lost to the prompt each refine pass).
        self._cn_refiner: StableDiffusionControlNetImg2ImgPipeline | None = None
        # Scene pipe (txt2img, no ControlNet) for stage A of paste-composite
        # compositions — shares weights with the main pipe; created lazily.
        self._scene_pipe: StableDiffusionPipeline | None = None
        self._lcm_loaded = False
        self._fast_mode = False
        self._default_scheduler_config: dict | None = None
        # Both QR Monster ControlNets — loaded once at warm time and swapped
        # by rebinding pipe.controlnet.nets[0]. Keeping both resident costs
        # ~700MB RAM but avoids a 5-10s reload per version switch.
        self._qr_controlnets: dict[str, ControlNetModel] = {}
        self._active_qr_version: str = QR_MONSTER_DEFAULT

    def _load_qr_controlnet(self, version: str) -> ControlNetModel:
        if version in self._qr_controlnets:
            return self._qr_controlnets[version]
        qr_kwargs: dict = {"torch_dtype": self.dtype}
        sub = _qr_monster_subfolder(version)
        if sub:
            qr_kwargs["subfolder"] = sub
        cn = ControlNetModel.from_pretrained(CONTROLNET_ID, **qr_kwargs)
        if self.device != "cpu":
            cn = cn.to(self.device)
        self._qr_controlnets[version] = cn
        return cn

    def load(self) -> None:
        if self._pipe is not None:
            return
        # Load BOTH versions up-front so the user can switch between them
        # without a model reload between jobs.
        for v in QR_MONSTER_VERSIONS:
            self._load_qr_controlnet(v)
        qr_controlnet = self._qr_controlnets[self._active_qr_version]
        tile_controlnet = ControlNetModel.from_pretrained(CONTROLNET_TILE_ID, torch_dtype=self.dtype)
        canny_controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_ID, torch_dtype=self.dtype)
        # Multi-ControlNet: three slots — QR Monster (always), Tile (photo
        # coherence), Canny (init-image structure). Tile + Canny default to
        # scale 0 so they're dormant unless the request opts in.
        controlnets = [qr_controlnet, tile_controlnet, canny_controlnet]

        kwargs: dict = {
            "torch_dtype": self.dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if self.use_external_vae:
            kwargs["vae"] = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=self.dtype)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnets, **kwargs
        )
        # Scheduler choice — Euler-Ancestral by default; DPM++ 2M Karras
        # available via QRART_SAMPLER=dpm_karras. The original scheduler
        # config is captured BEFORE the swap so set_fast_mode can restore
        # the right Quality-mode scheduler later.
        pipe.scheduler = _build_scheduler(pipe.scheduler.config)
        self._default_scheduler_config = dict(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        if self.device != "cpu":
            pipe.enable_attention_slicing()
        self._pipe = pipe

        # Plain img2img refiner — used by hires_fix and adetailer_faces, which
        # operate on the final winner image where the QR pattern is already
        # locked. Light strength (0.15-0.25) avoids erasing it.
        self._refiner = StableDiffusionImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=pipe.feature_extractor,
            requires_safety_checker=False,
        )

        # ControlNet-aware img2img refiner — used by the main refine pass.
        # Shares the SAME multi-controlnet instance as _pipe, so when we
        # swap pipe.controlnet.nets[0] for v1/v2 selection, the refiner
        # sees it automatically.
        self._cn_refiner = StableDiffusionControlNetImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            controlnet=pipe.controlnet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=pipe.feature_extractor,
            requires_safety_checker=False,
        )

        # Scene generation pipe — txt2img with no ControlNet. Used for stage A
        # of paste-composite compositions to produce the surrounding scene.
        self._scene_pipe = StableDiffusionPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=pipe.feature_extractor,
            requires_safety_checker=False,
        )

    def ensure_lcm(self) -> None:
        """Lazily download + apply the LCM LoRA the first time Fast mode is used."""
        self.load()
        if self._lcm_loaded:
            return
        assert self._pipe is not None
        # Load the LoRA. Earlier code passed adapter_name="lcm" then called
        # set_adapters(["lcm"]) — in diffusers 0.37 the adapter name doesn't
        # register reliably when the pipeline has multi-controlnet attached,
        # producing "Adapter name(s) {'lcm'} not in the list of present
        # adapters: set()". enable_lora()/disable_lora() works regardless of
        # naming, so use that for the toggle.
        self._pipe.load_lora_weights(LCM_LORA_ID)
        self._pipe.disable_lora()  # default off; set_fast_mode flips it on
        self._lcm_loaded = True

    def set_qr_monster_version(self, version: str) -> None:
        """Swap the active QR Monster ControlNet on the multi-controlnet.

        Both v1 and v2 are pre-loaded by load(); this is just a pointer
        rebind on pipe.controlnet.nets[0] (the QR slot — the Tile
        ControlNet stays at index 1). Idempotent.
        """
        if version not in QR_MONSTER_VERSIONS:
            raise ValueError(f"unknown QR Monster version: {version}")
        if version == self._active_qr_version and self._pipe is not None:
            return
        self.load()
        assert self._pipe is not None
        cn = self._load_qr_controlnet(version)
        # MultiControlNetModel exposes its members via .nets (list-like).
        # Rebinding [0] is enough — the pipe holds a reference to the
        # MultiControlNet wrapper, not the inner net directly.
        self._pipe.controlnet.nets[0] = cn
        self._active_qr_version = version

    def set_fast_mode(self, fast: bool) -> None:
        """Switch between Quality (Euler-Ancestral, LoRA disabled) and Fast
        (LCM scheduler + LCM LoRA enabled). Idempotent — calling twice is a no-op."""
        if fast == self._fast_mode and self._pipe is not None:
            return
        self.load()
        assert self._pipe is not None and self._refiner is not None
        if fast:
            self.ensure_lcm()
            self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
            self._pipe.enable_lora()
        else:
            assert self._default_scheduler_config is not None
            self._pipe.scheduler = _build_scheduler(self._default_scheduler_config)
            if self._lcm_loaded:
                self._pipe.disable_lora()
        # Keep all sibling pipes (plain refiner, ControlNet refiner, scene)
        # on the same scheduler so they stay in sync with Fast/Quality mode.
        self._refiner.scheduler = self._pipe.scheduler
        if self._cn_refiner is not None:
            self._cn_refiner.scheduler = self._pipe.scheduler
        if self._scene_pipe is not None:
            self._scene_pipe.scheduler = self._pipe.scheduler
        self._fast_mode = fast

    def generate_pass1(
        self,
        qr_image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        steps: int,
        guidance: float,
        controlnet_scale: float,
        tile_scale: float,
        control_start: float,
        control_end: float,
        seed: int | None,
        width: int,
        height: int,
        canny_image: Image.Image | None = None,
        canny_scale: float = 0.0,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        self.load()
        assert self._pipe is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        qr = qr_image.resize((width, height))
        canny = _prep_canny_input(canny_image, width, height)
        out = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[qr, qr, canny],
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=[controlnet_scale, tile_scale, canny_scale],
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            generator=gen,
            width=width,
            height=height,
            callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
        )
        return out.images[0]

    def generate_pass1_from_init(
        self,
        init_image: Image.Image,
        qr_image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        steps: int,
        guidance: float,
        controlnet_scale: float,
        tile_scale: float,
        strength: float,
        seed: int | None,
        width: int,
        height: int,
        control_start: float = 0.0,
        control_end: float = 1.0,
        canny_image: Image.Image | None = None,
        canny_scale: float = 0.0,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        """Pass-1 variant that uses a user-supplied init image instead of
        starting from noise. ControlNet img2img with the same multi-controlnet
        (QR Monster + Tile + optional Canny) drives the QR pattern into the
        init image while `strength` controls how much the original is preserved.

        canny_scale > 0 with a Canny-edge image stacks structural conditioning
        — useful for logo-shaped QR codes where you want the modules to
        cluster along the logo's edges.

        strength=1.0 ignores the init (equivalent to txt2img); 0.0 returns
        the init unchanged. Useful band: 0.5–0.85 for "QR-ify this photo,"
        0.25–0.45 for "barely modify it, just embed the QR."
        """
        self.load()
        assert self._cn_refiner is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        init = init_image.convert("RGB").resize((width, height))
        qr = qr_image.resize((width, height))
        canny = _prep_canny_input(canny_image, width, height)
        out = self._cn_refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=[qr, qr, canny],
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=[controlnet_scale, tile_scale, canny_scale],
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            generator=gen,
            callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
        )
        return out.images[0]

    def generate_scene(
        self,
        prompt: str,
        *,
        negative_prompt: str,
        steps: int,
        guidance: float,
        seed: int | None,
        width: int,
        height: int,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        """Stage A: clean scene generation with no ControlNet. Used as the
        base for inpaint compositions. Output is what the canvas would look
        like before any QR is added."""
        self.load()
        assert self._scene_pipe is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )
        out = self._scene_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            width=width,
            height=height,
            callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
        )
        return out.images[0]

    def refine(
        self,
        image: Image.Image,
        prompt: str,
        *,
        qr_image: Image.Image,
        controlnet_scale: float,
        tile_scale: float,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
        control_start: float = 0.0,
        control_end: float = 1.0,
        canny_image: Image.Image | None = None,
        canny_scale: float = 0.0,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        """img2img refine with the same multi-controlnet (QR + Tile + optional
        Canny) so the QR pattern AND any structural conditioning are preserved
        while the prompt polishes the photo.

        Without ControlNet during refine, plain img2img at strength 0.30
        erodes the QR by 5-15% per pass — the prompt steers pixels back
        toward "just a photo." With ControlNet, the QR signal is re-imposed
        each step, so we can polish more aggressively without losing scan.

        Canny is propagated through refine at the same scale as pass-1 so
        logo-shaped outputs don't have their structure smoothed away.
        """
        self.load()
        assert self._cn_refiner is not None
        gen = (
            torch.Generator(device="cpu").manual_seed(seed + 7919)
            if seed is not None
            else None
        )
        qr = qr_image.resize(image.size)
        canny = _prep_canny_input(canny_image, image.size[0], image.size[1])
        out = self._cn_refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=[qr, qr, canny],
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=[controlnet_scale, tile_scale, canny_scale],
            control_guidance_start=control_start,
            control_guidance_end=control_end,
            generator=gen,
            callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
        )
        return out.images[0]

    def hires_fix(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        target_size: int,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        """Lanczos-upscale to target_size on the longest dim, then img2img at
        low strength. Adds detail/sharpness without redrawing — strength must
        stay low (~0.15-0.25) or the QR pattern washes out.
        """
        self.load()
        assert self._refiner is not None

        w, h = image.size
        if w >= h:
            new_w = target_size
            new_h = int(round(h * (target_size / w)))
        else:
            new_h = target_size
            new_w = int(round(w * (target_size / h)))
        # SD requires multiple-of-8 dims.
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        if new_w <= w and new_h <= h:
            return image

        upscaled = image.resize((new_w, new_h), Image.LANCZOS)
        gen = (
            torch.Generator(device="cpu").manual_seed(seed + 31337)
            if seed is not None
            else None
        )
        out = self._refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
        )
        return out.images[0]

    def adetailer_faces(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance: float,
        seed: int | None,
        step_callback: StepCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Image.Image:
        """Detect faces with cv2 Haar cascade, re-render each at 512x512 via
        img2img, paste back. No-op if no faces detected. Uses lower strength
        than full refine so the underlying QR structure isn't disrupted.
        """
        import cv2
        import numpy as np

        self.load()
        assert self._refiner is not None

        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )
        if len(faces) == 0:
            return image

        result = image.copy()
        face_prompt = (
            prompt
            + ", detailed realistic face, sharp focus, photorealistic skin texture, "
              "natural eyes, symmetric features"
        )
        face_negative = (
            (negative_prompt + ", ") if negative_prompt else ""
        ) + "deformed, asymmetric eyes, extra eyes, distorted face, blurry face"

        for (x, y, w, h) in faces:
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.width, x + w + pad_x)
            y2 = min(image.height, y + h + pad_y)
            face_crop = result.crop((x1, y1, x2, y2))
            orig_size = face_crop.size
            face_resized = face_crop.resize((512, 512), Image.LANCZOS)

            face_seed = (seed + int(x) * 7 + int(y) * 13) if seed is not None else None
            gen = (
                torch.Generator(device="cpu").manual_seed(face_seed)
                if face_seed is not None
                else None
            )
            out = self._refiner(
                prompt=face_prompt,
                negative_prompt=face_negative,
                image=face_resized,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                callback_on_step_end=_make_diffusers_callback(step_callback, cancel_check),
            )
            face_rendered = out.images[0].resize(orig_size, Image.LANCZOS)
            result.paste(face_rendered, (x1, y1))

        return result
