# QR Art Studio

Local AI QR art generator. Produces photorealistic (or stylized) images that secretly encode a QR code, using Stable Diffusion 1.5 + the Monster Labs QR ControlNet. Web UI, async queue, live progress, history, and a multi-scanner verification stack.

Runs on Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

```
                  ┌────────────────────────────────────────────────────────┐
   URL  ────►     │  Multi-ControlNet pipe                                 │
                  │   • QR Monster (v1 or v2) — drives the code            │
                  │   • Tile           — photo coherence (optional)        │
                  │   • Canny          — init-image structure (optional)   │
                  └──────────┬──────────────────────────┬──────────────────┘
                             │ Pass 1 (txt2img or       │ Refine (img2img +
                             │ img2img w/ init image)   │ same multi-CN)
                             ▼                          ▼
                  ┌──────────────────┐         ┌──────────────────┐
                  │  Candidate i     │ ──────► │  Candidate i'    │
                  └──────────────────┘         └──────────────────┘
                                                        │
                  ┌─────────────────────────────────────┴──────────┐
                  │  Multi-scanner ensemble                        │
                  │   • cv2.QRCodeDetector                         │
                  │   • zxing-cpp                                  │
                  │   • qreader (YOLO + libzbar, iOS-class)        │
                  │   • Per-module scannability score (0.0–1.0)    │
                  └────────────────────────────────────────────────┘
```

If zero of N candidates scan, an **in-generation rescue pass** retries one candidate at scale +0.10 with a new seed. If that still misses and `auto_escalate` is on, the worker spawns a follow-up job at the higher scale. The single best candidate is selected by `(scans, scannability, controlnet_scale)`.

---

## Setup

Requires Python 3.12 (3.11 works too).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### First-run model downloads (~7–8 GB, cached under `~/.cache/huggingface`)

| Asset | Purpose | Size |
|---|---|---|
| `SG161222/Realistic_Vision_V6.0_B1_noVAE` | Default photoreal SD 1.5 base | ~4 GB |
| `monster-labs/control_v1p_sd15_qrcode_monster` | QR Monster v1 + v2 (both loaded) | ~1.4 GB |
| `lllyasviel/control_v11f1e_sd15_tile` | Tile ControlNet (photo coherence) | ~700 MB |
| `lllyasviel/sd-controlnet-canny` | Canny ControlNet (logo structure) | ~700 MB |
| `stabilityai/sd-vae-ft-mse` | Sharper VAE | ~330 MB |
| `latent-consistency/lcm-lora-sdv1-5` | LCM-LoRA for Fast mode | ~67 MB |

Additional SD 1.5 finetune models (~4 GB each) download lazily the first time they're requested via the UI.

---

## Running the server

```bash
source venv/bin/activate
python app.py
# open http://127.0.0.1:8000
```

The first generation triggers the model load (~30 s on M-series). Subsequent generations run ~30–60 s for a 768×768 image with refine on.

---

## Form controls

The form is grouped top-to-bottom into: payload, init image, prompt, model & version, style & composition, **Tuning** (ControlNet + diffusion parameters), and toggles (Fast / Auto-escalate / Refine / Hi-res fix / ADetailer).

### Payload

- **QR type** — `URL`, `Plain text`, `Email`, `Phone`, `SMS`, `WiFi`, `Contact (vCard)`, `Location`. Each variant builds a properly formatted payload (e.g. `mailto:`, `WIFI:T:WPA;S:...;P:...;;`, `BEGIN:VCARD...`) before encoding.

### Init image (optional)

- **Drop / click to upload.** PNG / JPEG / WebP up to 10 MB. Hashed (SHA256) and content-addressed under `outputs/_assets/{sha}.png` — uploading the same file twice deduplicates.
- **Preserve input** *(slider, 0.05–0.95, default 0.35)* — how much of your original image survives the diffusion. Higher = more original, lower = more reimagined.
  - `0.65–0.85` for "QR-ify this exact photo"
  - `0.25–0.45` for "use this as a starting point, but reimagine it"
- **Logo structure** *(slider, 0.00–1.20, default 0.00)* — Canny edge ControlNet weight. Stacks structural conditioning so the result's silhouette follows the init image's edges. Use for **logo-shaped QR codes**:
  - `0.50–0.80` typical — modules cluster along the logo's outlines
  - `≥ 1.00` starts to dominate over QR Monster and scans break

> **Composition behavior**: in `standalone` mode the init image becomes the img2img seed for pass-1. In non-standalone modes (subject-portrait / scene-landscape / garment) the init image *replaces* the auto-generated scene that the QR art is composited into.

### Prompt

The main text-to-image prompt. The selected **Style** preset is appended automatically (RAW photo / 8k / DSLR for photoreal, etc.) — you don't need to add those yourself.

### Model

14 SD 1.5 finetunes, listed below. See the [Models](#models) section for when to pick which.

### QR Monster ControlNet

- **v1** *(default)* — original. Scale sweet spot **1.10–1.20**.
- **v2** — stronger conditioning. Scale sweet spot **0.95–1.05** (drop scale ~0.10 from v1).

Both versions load at warm time and swap instantly between jobs. The QR strength slider auto-adjusts to the new sweet-spot midpoint when you change the dropdown.

### Style

| Style | Positive suffix | Negative defaults |
|---|---|---|
| `photoreal` | RAW photo / 8k / DSLR / film grain / Kodak Portra 400 | illustration, painting, render, CGI, plastic, … |
| `cinematic` | dramatic lighting / octane render / masterpiece / 8k / vivid | low quality, ugly, jpeg artifacts |
| `illustration` | digital illustration / concept art / trending on artstation | photo, photographic, jpeg artifacts |
| `custom` | *(none — your prompt is sent verbatim)* | low quality, deformed, watermark |

All styles include `"low contrast, washed out, foggy, hazy"` in the negative — these tokens push the model away from soft tonal regions where QR modules get mushy.

### Composition

| Composition | Canvas | QR region | Use for |
|---|---|---|---|
| `standalone` | 768×768 | full canvas | Most outputs. QR fills the frame; the prompt fills around it via QR coverage. |
| `subject-portrait` | 768×1024 (portrait) | 640×640 lower-center | A subject *above*, the QR woven into a patterned ornament below |
| `scene-landscape` | 1024×768 (wide) | 720×720 right side | Wide scene with the QR as a stone-monolith feature on the right |
| `garment` | 768×1024 (portrait) | 720×720 center | Fashion shot — the QR is the patterned outfit centerpiece |

Non-standalone modes use a **paste-composite**: scene + QR art are generated separately, then alpha-feathered and finder-pattern-reinforced together. This sidesteps the "QR Monster trained on full-canvas QRs" problem that breaks naive inpaint compositions.

---

## Tuning sliders

The most consequential section — these are the ControlNet and diffusion knobs.

### QR strength *(0.80–2.00, default 1.10)*

The QR Monster ControlNet conditioning scale. **The main lever for "how hidden vs. how visible is the QR."**

| Value | Effect |
|---|---|
| `0.80–1.00` | Subtle. QR may not scan reliably; great when "photo first, code second" is the goal. |
| `1.05–1.20` | **Photo-dominant sweet spot for v1.** Most outputs scan; QR is hidden in texture. |
| `0.95–1.05` | **Sweet spot for v2.** |
| `1.25–1.40` | QR-dominant. Grid pattern becomes part of the visible composition (think dome-temples). |
| `≥ 1.50` | Output reads as "a QR with photo-flavored texture." Last-resort retry territory. |

Auto-adjusts when you change the QR Monster version dropdown.

### QR coverage *(0.40–1.00, default 1.00)*

Fraction of the diffusion canvas the QR occupies. `1.00` = QR fills the canvas (legacy behavior). `<1.00` centers a smaller QR with `#808080` gray padding around it; the diffusion paints the prompt's scene content in the gray margin while QR Monster only conditions the central region.

| Value | Effect |
|---|---|
| `1.00` | QR fills the frame. Best for dense-content prompts where the whole canvas is the canvas. |
| `0.85–0.90` | Subtle margin. Mostly QR with a sliver of "real scene" around the edges. |
| **`0.70–0.80`** | **Community v2 sweet spot.** QR as a feature in a scene — tree-canopy / temple-cluster / cherry-blossom aesthetic. |
| `0.55–0.65` | Small QR in a large scene. Risk of module density getting too small to scan. |
| `< 0.55` | Usually breaks scanning. |

### Tile guidance *(0.00–0.80, default 0.00)*

Stacks Tile ControlNet alongside QR Monster. Adds a coherence / detail-preservation signal that pushes outputs toward "photo-like" structure.

| Value | Effect |
|---|---|
| `0.00` | Off (default). |
| `0.30–0.50` | Recommended for photoreal scenes that look too noisy or QR-grid-y. |
| `≥ 0.60` | Starts to soften the QR signal — may reduce scan rate. |

### Control start *(0.00–0.90, default 0.30)* + Control end *(0.10–1.00, default 0.95)*

The **ControlNet active window**. Fractions of total diffusion steps. Outside this window the diffusion runs as pure txt2img / img2img with no QR pull.

| Window | Effect |
|---|---|
| `0.00 → 1.00` | Legacy "full-window" behavior. QR Monster conditions every step. Best for "QR fills the canvas" outputs. |
| **`0.30 → 0.95`** | **Default / community v2 sweet spot.** First 30% of steps paint the scene freely from the prompt; QR Monster shapes the result from step 30–95%; the final 5% finishes naturally without QR pull. Produces scenes that look composed naturally with the QR woven in, not built out of QR pattern from step 0. |
| `0.50 → 0.95` | Prompt-dominant. Scene composition is fully baked before QR Monster starts. QR is faint. |
| `0.00 → 0.80` | QR-dominant during early steps, free at the end. Crisp QR that's been "polished" into something photo-like. |

### Steps *(15–60, default 32)*

Diffusion iterations. More steps = more detail and cleaner edges, at the cost of linearly more time. 28–40 is the practical range. Fast mode overrides to 6.

### Candidates *(1–8, default 5)*

How many seeds to try in parallel within a single job. The best one (by `(scans, scannability, scale)`) becomes the final result; the rest are saved alongside for browsing. More candidates = higher chance one scans, but linearly more compute.

### Seed *(int, blank = random)*

Deterministic seed for reproducibility. With `keep_seed` on Remix, identical settings + seed produce identical output.

### Size *(512–1024, default 768)*

Pixel side length. Currently only affects standalone outputs. Larger = sharper at the cost of generation time; 768 is the SD 1.5 sweet spot.

---

## Toggles

### Fast mode

Swaps in [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) + LCMScheduler. **~3–4× faster** per candidate (6 steps instead of 32) at the cost of some fidelity. Steps slider auto-jumps to 6; CFG is overridden server-side to ~1.5. Great for prompt iteration; not great for final outputs.

### Auto-escalate

If `require_scan` is on AND zero of N candidates scan AND the best score ≥ `0.70` AND current scale `< 1.50`, the worker enqueues a follow-up job at `controlnet_scale + 0.10` with a new seed. Chains up to the cap. Cheap insurance against "I just got 5 unscannable candidates in a row."

### Refine pass

A second diffusion pass: ControlNet-aware img2img on top of pass-1. The multi-controlnet stays attached so the QR pattern is re-imposed each step while img2img polishes detail. Without this, pass-1 outputs are visibly QR-textured.

- **Refine strength** *(0.10–0.55, default 0.30)* — how much pixels are allowed to change. Lower = closer to pass-1 (preserves QR), higher = more photoreal (more risk of degrading scan rate, but now bounded by ControlNet re-imposition).

### Hi-res fix

Runs once on the winner candidate only. Lanczos-upscales the image to a larger square, then runs a low-strength img2img pass through the plain refiner (no ControlNet — by that point the QR is locked in). Output is print-ready.

- **Hi-res target** *(896–1536 px, default 1024)*
- **Hi-res strength** *(0.10–0.40, default 0.20)* — low values preserve detail; higher values redraw more aggressively.

### ADetailer (faces)

Detects faces with OpenCV's Haar cascade, crops + re-renders each at 512×512 via img2img, pastes back. Same winner-only semantics as hi-res. Use only when humans are in the frame — useless on wildlife / landscapes (Haar cascade is human-face-only).

- **Face strength** *(0.20–0.55, default 0.35)*

---

## Models

All are SD 1.5 finetunes — same VAE, same ControlNet stack. They differ in their **photo prior** (how aggressively they resist QR pattern bleed-through) and their **aesthetic palette**.

### Strongest photo prior — best for "hidden QR"

| Model | Strengths | When to use |
|---|---|---|
| `photoreal` (Realistic Vision V6) | All-around photoreal, strongest photo prior, good resistance to controlnet override | **Default for most outputs.** Nature, animals, architecture, portraits. |
| `photoreal-v51` (Realistic Vision V5.1) | Slightly softer focus than V6 | When V6 looks too clinical. |
| `majicmix` (majicMIX Realistic v6) | Warm cinematic skin tones, faces, fur | People, wildlife, golden-hour scenes. |
| `epicphoto` (epiCPhotoGasm) | Editorial-magazine sharpness, clean micro-detail | Architecture, products, anything with sharp geometry. |
| `cyberrealistic` (CyberRealistic) | Crisp modern photography, neon palettes hold up | Cyberpunk, night cities, technology, sci-fi. |
| `hyperrealism` (HyperRealism) | Pore-level micro-detail | Close-up portraits, dramatic faces. |
| `absolute-v18` (AbsoluteReality v1.8.1) | Balanced, fine mid-tones | Versatile alternate to `photoreal`. |

### Other photoreal

| Model | Strengths | When to use |
|---|---|---|
| `photon` (Photon V1) | Natural landscapes, warm/film feel | Outdoor scenes, sunsets, forests. |
| `epic` (epiCRealism) | Sharp, dramatic, high-detail | Cinematic compositions with strong lighting. |
| `absolute` (AbsoluteReality v1) | Versatile, balanced | When you want a neutral fallback. |
| `analog` (Analog Diffusion) | Film grain, 70s/80s aesthetic | Retro / nostalgic / muted-color outputs. |
| `dreamlike` (Dreamlike Photoreal 2.0) | Soft, golden, painterly photoreal | Fantasy-leaning realism. |

### Stylized — best for "QR as visible composition"

| Model | Strengths | When to use |
|---|---|---|
| `openjourney` (OpenJourney v4) | Midjourney-style, moody/cinematic | Concept art, dramatic atmospheres. |
| `dreamshaper` (DreamShaper 8) | Stylized/painterly | When you want the QR's grid to read as part of the composition (temple-cluster, cherry-blossom, neon fantasy). Pair with `illustration` style and v2 at scale 1.25+. |

> **Rule of thumb:** for "hide the QR in a photo" use a **photoreal** model + v1 + scale ~1.10 + coverage 0.75–0.85. For "let the QR be the visible structure of a stylized composition" use **`dreamshaper`** / **`openjourney`** + `illustration` style + v2 + scale 1.20+ + coverage 0.70 + refine **off**.

---

## Recipes

### Flagship photoreal (dense scenes — nature, animals, architecture)

Model `photoreal` / Style `photoreal` / **v1** / Scale **1.10** / Coverage **0.75** / Control window **0.30 → 0.95** / Tile **0.30** / Steps **38** / Candidates **5** / Refine **ON @ 0.40** / Hi-res **ON @ 1024 / 0.18**.

### Stylized "QR as composition" (Gooey / Reddit v2 aesthetic)

Model `dreamshaper` / Style `illustration` / **v2** / Scale **1.25** / Coverage **0.70** / Control window **0.30 → 0.95** / Tile **0.00** / Steps **40** / Refine **OFF** / Hi-res **OFF**.

### Logo-shaped QR (init image + Canny stack)

Model `photoreal` / Style `photoreal` / **v1** / Scale **1.10** / Coverage **0.75** / Preserve input **0.20** / **Logo structure 0.60** / Control window **0.30 → 0.95** / Refine **ON @ 0.30**.

### Cyberpunk night cities

Model `cyberrealistic` / Style `cinematic` / **v1** / Scale **1.10** / Coverage **0.80** / Tile **0.40** / Steps **40** / Refine **ON @ 0.40** / Hi-res **ON @ 1024 / 0.18**.

### Fast iteration

Anything above with **Fast mode ON** (6 steps, LCM scheduler). Use for prompt-tuning, then disable for the final.

---

## API

All endpoints are JSON unless noted.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Status, device, loaded models, queue depth, active QR Monster + sampler. Always open even with auth on. |
| `/api/warm` | POST `{model}` | Pre-load a model into VRAM/RAM. |
| `/api/assets` | POST (multipart) | Upload an init image. SHA256-deduped. Returns `{hash, url, width, height}`. |
| `/api/generate` | POST `<GenerateBody>` | Enqueue a generation. Returns `{job_id, queue_position}` immediately. |
| `/api/jobs/{id}/stream` | GET (SSE) | Live progress events: `started`, `qr_ready`, `phase`, `step`, `candidate_started`, `candidate_done`, `rescue_started`, `rescue_done`, `auto_escalated`, `completed`, `failed`, `cancelled`. |
| `/api/jobs/{id}` | GET | Full job record + candidate list. |
| `/api/jobs/{id}/rerun` | POST `{keep_seed?}` | Clone + re-queue an existing job's settings. |
| `/api/jobs/{id}` | DELETE | Cancel if queued/running; hard-delete (row + cascade + `rm -rf outputs/{id}/`) if terminal. |
| `/api/jobs` | GET `?status=&model=&scans=&q=&limit=&offset=` | List/filter jobs. |
| `/api/prompts/recent` | GET | Recently-used + favorited prompts. |
| `/api/prompts/{id}/favorite` | POST | Star/unstar a prompt. |
| `/api/stats` | GET | Aggregate stats (totals, top prompts, scan rate). |
| `/api/admin/cleanup` | POST | Run the retention pass (evict jobs beyond `QRART_RETENTION_KEEP`). |

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `QRART_MONSTER_VERSION` | `v1` | Default QR Monster version. UI dropdown still lets users pick per-request. |
| `QRART_SAMPLER` | `euler_a` | Scheduler. Set to `dpm_karras` to use `DPMSolverMultistepScheduler` with `use_karras_sigmas=True` (community standard, cleaner detail; non-SDE only — SDE variants NaN on MPS). |
| `QRART_RETENTION_KEEP` | `1000` | Number of recent jobs to retain. Older jobs are evicted (row + files) at startup. |
| `QRART_AUTH` | *(unset)* | Set to `user:pass` to gate all `/api/*` + `/outputs/*` + `/` behind HTTP Basic auth with a timing-safe compare. `/api/health` stays open for probes. |
| `QRART_MPS_FP16` | `0` | Set to `1` to opt into fp16 on Apple Silicon. Default is fp32 because Realistic Vision V6 NaNs in fp16 on MPS. |

---

## Codebase layout

```
qrcode/
├── app.py                    # FastAPI server: endpoints, worker callback, auth
├── qrart/
│   ├── pipeline.py           # SD pipes, multi-ControlNet, scheduler swaps
│   ├── generator.py          # Generation loop, candidate selection, C1 rescue
│   ├── canvas.py             # Compositions, QR coverage, paste-composite, finder reinforcement
│   ├── qr.py                 # QR encoding helpers (qrcode lib wrappers)
│   ├── scannability.py       # 0–1 per-module scan score
│   ├── scanner.py            # cv2 + zxing + qreader ensemble
│   ├── styles.py             # Positive/negative prompt presets
│   ├── worker.py             # Single-thread job queue
│   ├── db.py                 # SQLite (WAL), 8 migrations
│   └── migrations/           # 001 initial → 008 control_start
├── static/index.html         # Single-file UI
└── outputs/                  # Job dirs + _assets/ for user uploads
```

---

## Architecture notes

- **Single-thread worker.** MPS allows one diffusion process at a time; the worker queue has `MAX_QUEUED = 5` slots. Excess `/api/generate` calls return 503.
- **SSE polling.** The stream endpoint polls `job_events` every 250 ms and pushes new events to the EventSource. ~250 ms typical end-to-end latency from server-side emit to UI.
- **Schema versioning.** SQL files in `qrart/migrations/` run at startup in order; the `meta` table tracks the current version.
- **Content-addressed assets.** User uploads go to `outputs/_assets/{sha}.png`. The same file uploaded twice is a no-op.
- **Best-candidate selection.** Sort key is `(0 if scans else 1, -scannability, controlnet_scale)`. Scannability breaks ties when multiple candidates scan, and selects the closest near-miss when none do.

---

## Troubleshooting

- **First generation hangs ~30 s.** Normal — models are loading from HF cache to MPS.
- **NaN black outputs on MPS.** Check that `QRART_MPS_FP16` is unset / `0`. Some finetunes don't tolerate fp16 on Apple Silicon.
- **Queue full (503).** `MAX_QUEUED = 5`. Wait for a running job to finish or cancel it via the history's × button.
- **"No candidate scanned."** Auto-escalate should kick in if best score ≥ 0.70. If best score is below 0.70 the prompt is fundamentally hard for QR concealment (too much flat content — sky, snow, solid colors). Switch to a denser prompt or use QR coverage <1.0 to give the QR a defined region.
- **v2 produces over-driven QR pattern (buildings ARE modules).** You're running v1's scale on v2. Toggle the version dropdown so the scale auto-adjusts to v2's sweet spot (~0.95), or drop scale manually.
- **Init image looks like a faded overlay.** Switch from "img2img only" (init image alone) to **Canny stacked** (set Logo structure ≥ 0.5). img2img blends pixels by opacity; Canny conditions structure.
