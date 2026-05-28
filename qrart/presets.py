"""One-click preset recipes.

Each preset bundles every GenerateBody field except `data` and `prompt`,
plus a prompt template with a `{SUBJECT}` placeholder and a list of
example subjects. The UI substitutes the first example into the
template when the preset is applied; the user can edit before
generating.

Two tiers:
  • Tier 1 (1-30): subject-based "great hidden-QR" recipes built around
    specific scene types (wildlife, architecture, food, etc.).
  • Tier 2 (31-60): artist-style recipes built around famous painters
    and movements. These use the stylized recipe (v2, scale ~1.20,
    refine OFF, illustration style) where the QR pattern reads as part
    of the artistic composition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Preset:
    slug: str
    name: str
    category: str
    icon: str
    description: str
    prompt: str
    placeholder_subjects: list[str]
    settings: dict[str, Any]
    negative_override: str | None = None
    requires_init: bool = False
    great_fit: bool = False  # marks presets that are especially good QR substrates

    def summary(self) -> dict[str, Any]:
        """Lightweight payload for /api/presets (no settings — saves bandwidth
        when listing 60+ items)."""
        return {
            "slug": self.slug,
            "name": self.name,
            "category": self.category,
            "icon": self.icon,
            "description": self.description,
            "placeholder_subjects": self.placeholder_subjects,
            "requires_init": self.requires_init,
            "great_fit": self.great_fit,
        }

    def full(self) -> dict[str, Any]:
        """Full preset payload for /api/presets/{slug} — UI applies this
        verbatim to the form."""
        return {
            **asdict(self),
        }


# ── Base recipes ─────────────────────────────────────────────────────────────
# Most presets are a base + a few overrides. Defining the bases once keeps
# the file readable and ensures consistency across presets in the same tier.

_BASE = {
    "negative_prompt": None,
    "guidance": 7.5,
    "control_start": 0.30,
    "control_end": 0.95,
    "tile_scale": 0.30,
    "candidates": 5,
    "refine_steps": 20,
    "size": 768,
    "composition": "standalone",
    "fast_mode": False,
    "auto_escalate": True,
    "require_scan": True,
    "init_image_path": None,
    "init_strength": 0.65,
    "canny_scale": 0.0,
    "adetailer": False,
    "adetailer_strength": 0.35,
    "hires_fix": True,
    "hires_target": 1024,
    "hires_strength": 0.18,
}

# Photoreal hidden-QR — the "default" recipe for tier 1.
PHOTOREAL = {
    **_BASE,
    "style": "photoreal",
    "model": "photoreal",
    "qr_monster_version": "v1",
    "controlnet_scale": 1.10,
    "qr_coverage": 0.75,
    "steps": 38,
    "refine": True,
    "refine_strength": 0.30,
}

# Artist-style — the recipe for tier 2.
ARTIST = {
    **_BASE,
    "style": "illustration",
    "model": "dreamshaper",
    "qr_monster_version": "v2",
    "controlnet_scale": 1.20,
    "qr_coverage": 0.72,
    "steps": 40,
    "refine": False,
    "refine_strength": 0.30,
    "tile_scale": 0.0,
    "hires_fix": False,
}


def _p(base: dict, **overrides) -> dict:
    """Shorthand: base | overrides as a dict."""
    return {**base, **overrides}


# ── Tier 1: subject-based photoreal recipes (30) ─────────────────────────────

TIER1: list[Preset] = [
    # 🦁 Wildlife & Nature (8)
    Preset(
        slug="wildlife-portrait",
        name="Wildlife portrait",
        category="🦁 Wildlife & Nature",
        icon="🐯",
        description="Close-up animal with detailed fur + intense eyes",
        prompt="{SUBJECT} portrait, sharp focused eyes, detailed fur texture, intense expression, wildlife photography, blurred natural background",
        placeholder_subjects=["Malayan tiger", "bald eagle", "snow leopard", "majestic lion", "red panda"],
        settings=_p(PHOTOREAL, model="majicmix"),
    ),
    Preset(
        slug="underwater-reef",
        name="Underwater reef",
        category="🦁 Wildlife & Nature",
        icon="🐠",
        description="Coral reef teeming with vibrant fish",
        prompt="vibrant saltwater coral reef teeming with {SUBJECT}, swaying soft purple sea fans, brain coral and staghorn coral textures, sunbeams piercing crystal clear turquoise water, refracted caustic light patterns, hyperdetailed marine ecosystem, National Geographic underwater photography",
        placeholder_subjects=["yellow tangs and clownfish", "schools of butterflyfish", "blue tangs grazing", "a curious sea turtle"],
        settings=PHOTOREAL,
    ),
    Preset(
        slug="rainforest-interior",
        name="Rainforest interior",
        category="🦁 Wildlife & Nature",
        icon="🌿",
        description="Dense jungle with god rays + bromeliads",
        prompt="lush dense rainforest interior at dawn featuring {SUBJECT}, towering kapok trees with buttress roots, thick green canopy filtering golden god rays through morning mist, vibrant red heliconia and bromeliad flowers, hanging vines, hyperdetailed nature photography, National Geographic rainforest",
        placeholder_subjects=["a scarlet macaw on a branch", "a jaguar prowling", "a tiny poison dart frog", "a blue morpho butterfly mid-flight"],
        settings=PHOTOREAL,
    ),
    Preset(
        slug="mountain-landscape",
        name="Mountain landscape",
        category="🦁 Wildlife & Nature",
        icon="⛰️",
        description="Peaks with dramatic light + atmospheric depth",
        prompt="{SUBJECT}, towering snow-capped mountain peaks, dramatic golden hour lighting, atmospheric haze and clouds, dense pine forest in the foreground, alpine meadow with wildflowers, ultra-detailed landscape photography, cinematic depth",
        placeholder_subjects=["Patagonian Torres del Paine", "Swiss Alps at sunrise", "Canadian Rockies", "Himalayan vista", "Dolomites at golden hour"],
        settings=_p(PHOTOREAL, model="photon", qr_coverage=0.78),
    ),
    Preset(
        slug="flower-field",
        name="Flower field",
        category="🦁 Wildlife & Nature",
        icon="🌻",
        description="Dense floral field with vivid color bands",
        prompt="vast vibrant flower field stretching to the horizon featuring {SUBJECT}, dense blooms with intricate petals and stems, sunlit summer afternoon, gentle breeze rippling through the field, dramatic landscape depth, hyperdetailed nature photography, sweeping countryside",
        placeholder_subjects=["bands of tulips and sunflowers", "rainbow rows of every color", "lavender rows of Provence", "red poppy fields"],
        settings=_p(PHOTOREAL, model="majicmix"),
    ),
    Preset(
        slug="forest-light-shafts",
        name="Forest light shafts",
        category="🦁 Wildlife & Nature",
        icon="🌳",
        description="Sunbeams through old-growth forest",
        prompt="{SUBJECT}, ancient old-growth forest interior with dramatic god rays piercing mist, towering pine and redwood trunks, moss-covered fallen logs, ferns carpeting the floor, drifting dust particles catching morning sunlight, atmospheric depth, cinematic nature photography",
        placeholder_subjects=["a deer at the forest edge", "a red fox crossing a clearing", "a single shaft of light hitting a clearing", "an ancient cedar grove"],
        settings=_p(PHOTOREAL, model="photon"),
    ),
    Preset(
        slug="aurora-night",
        name="Aurora night sky",
        category="🦁 Wildlife & Nature",
        icon="✨",
        description="Northern lights + warm-lit cabin",
        prompt="{SUBJECT} under vivid green and purple aurora borealis swirling across the night sky, deep snow blanketing the foreground, dense pine forest, frost-covered branches, dramatic long-exposure Lapland night photography, atmospheric shimmer",
        placeholder_subjects=["wooden alpine cabin glowing warm yellow", "reindeer in foreground", "a lone wanderer with lantern", "a frozen lake with reflections"],
        settings=_p(PHOTOREAL, controlnet_scale=1.12, qr_coverage=0.78),
    ),
    Preset(
        slug="bird-in-flight",
        name="Bird in flight",
        category="🦁 Wildlife & Nature",
        icon="🦅",
        description="Action shot of a bird mid-flight",
        prompt="{SUBJECT} mid-flight with wings fully spread, sharp talon and feather detail, dramatic sky background, motion frozen, wildlife photography, telephoto lens compression, golden hour rim lighting",
        placeholder_subjects=["bald eagle swooping", "red-tailed hawk soaring", "great horned owl", "peregrine falcon diving", "snowy owl gliding"],
        settings=_p(PHOTOREAL, model="majicmix"),
    ),

    # 🏙️ Architecture & Urban (5)
    Preset(
        slug="city-skyline-night",
        name="City skyline at night",
        category="🏙️ Architecture & Urban",
        icon="🌃",
        description="Skyscrapers + lit windows on indigo sky",
        prompt="{SUBJECT} at midnight, skyscrapers glowing with thousands of warm amber window lights against a deep indigo sky, dramatic moonlit cumulus clouds, distant red navigational lights blinking on antennas, long-exposure cinematic photography, navy and amber palette, anamorphic lens flares",
        placeholder_subjects=["Chicago skyline viewed from Lake Michigan", "Manhattan from Brooklyn Bridge", "Hong Kong harbor at night", "Tokyo Shinjuku district", "Dubai Burj towers"],
        settings=_p(PHOTOREAL, model="cyberrealistic", qr_coverage=0.80),
    ),
    Preset(
        slug="historic-landmark",
        name="Historic landmark",
        category="🏙️ Architecture & Urban",
        icon="🏛️",
        description="Iconic monument with dramatic light",
        prompt="{SUBJECT}, sweeping architectural detail, weathered stone texture, golden hour dramatic lighting, atmospheric haze, surrounding plaza and grounds, hyperdetailed editorial architectural photography, magazine-quality",
        placeholder_subjects=["Roman Colosseum at sunset", "Taj Mahal at dawn", "Angkor Wat sunrise", "Petra rose city facade", "Machu Picchu mountain"],
        settings=_p(PHOTOREAL, model="epicphoto"),
    ),
    Preset(
        slug="industrial-factory",
        name="Industrial / factory",
        category="🏙️ Architecture & Urban",
        icon="🏭",
        description="Pipes, machinery, steam, dramatic lighting",
        prompt="{SUBJECT}, industrial complex of pipes and machinery, dramatic shafts of light cutting through atmospheric steam and dust, weathered metal textures, glowing furnace light, cinematic depth of field, hyperdetailed industrial photography, gritty atmosphere",
        placeholder_subjects=["abandoned steel mill interior", "vintage steampunk workshop", "modern data center server rows", "oil refinery at twilight"],
        settings=_p(PHOTOREAL, model="cyberrealistic"),
    ),
    Preset(
        slug="opulent-interior",
        name="Opulent interior",
        category="🏙️ Architecture & Urban",
        icon="🛋️",
        description="Ornate room with dramatic lighting",
        prompt="{SUBJECT}, opulent ornate interior with carved details and decorative moldings, dramatic chandelier lighting, rich textures of velvet and marble and gold leaf, soft window light, intricate patterns on walls and ceiling, hyperdetailed architectural photography, magazine quality",
        placeholder_subjects=["Baroque palace ballroom", "Art Deco hotel lobby", "Gilded Age library", "Versailles Hall of Mirrors", "Moroccan riad courtyard"],
        settings=_p(PHOTOREAL, model="epicphoto", qr_coverage=0.78),
    ),
    Preset(
        slug="cyberpunk-megacity",
        name="Cyberpunk megacity",
        category="🏙️ Architecture & Urban",
        icon="🌆",
        description="Neon-lit rain-soaked dystopian street",
        prompt="{SUBJECT}, dense rain-soaked neon-lit cyberpunk megacity, holographic billboards reflecting in puddles, steam from street vendors, motorcycle chrome reflections, flying car silhouettes, atmospheric fog, Blade Runner 2049 cinematic, electric cyan and magenta palette",
        placeholder_subjects=["futuristic Tokyo alley at midnight", "Hong Kong Kowloon walled city", "neon-drenched dystopian downtown", "cyberpunk bazaar street"],
        settings=_p(PHOTOREAL, model="cyberrealistic", controlnet_scale=1.12, tile_scale=0.40),
    ),

    # 🍽️ Lifestyle & Culture (5)
    Preset(
        slug="food-culinary",
        name="Food / culinary",
        category="🍽️ Lifestyle & Culture",
        icon="🍝",
        description="Plate close-up + dramatic restaurant light",
        prompt="{SUBJECT}, gourmet plating with intricate garnish and texture, dramatic restaurant lighting, soft shadows and rim light, shallow depth of field, steam and sauce drizzle, hyperdetailed editorial food photography, Michelin-star presentation",
        placeholder_subjects=["fresh sushi platter", "rustic Italian pasta", "decadent chocolate dessert", "vibrant ramen bowl", "artisan charcuterie board"],
        settings=_p(PHOTOREAL, model="epicphoto", qr_coverage=0.80),
    ),
    Preset(
        slug="fashion-editorial",
        name="Fashion editorial",
        category="🍽️ Lifestyle & Culture",
        icon="👗",
        description="High-fashion subject + dramatic lighting",
        prompt="{SUBJECT}, high fashion editorial photograph, dramatic studio lighting, intricate fabric texture and pattern detail, confident pose, magazine-cover composition, Vogue editorial aesthetic, hyperdetailed beauty photography",
        placeholder_subjects=["model in flowing crimson gown", "haute couture sculptural dress", "androgynous editorial portrait", "fierce runway look"],
        settings=_p(PHOTOREAL, model="majicmix", adetailer=True),
    ),
    Preset(
        slug="travel-destination",
        name="Travel destination",
        category="🍽️ Lifestyle & Culture",
        icon="✈️",
        description="Iconic travel scene + atmospheric light",
        prompt="{SUBJECT}, iconic travel destination vista, dramatic golden hour lighting, atmospheric haze, surrounding landscape detail, hyperdetailed editorial travel photography, National Geographic feature, magazine cover quality",
        placeholder_subjects=["Santorini blue and white cliffs at sunset", "Kyoto bamboo grove path", "Iceland black sand beach", "Marrakech bazaar alley", "Cinque Terre coastal village"],
        settings=_p(PHOTOREAL, model="photon", qr_coverage=0.78),
    ),
    Preset(
        slug="pet-portrait",
        name="Pet portrait",
        category="🍽️ Lifestyle & Culture",
        icon="🐕",
        description="Studio-style pet close-up",
        prompt="{SUBJECT}, professional pet portrait, sharp focused eyes, detailed fur texture, expressive personality, soft studio lighting, shallow depth of field, hyperdetailed pet photography",
        placeholder_subjects=["golden retriever puppy", "regal black cat", "Australian shepherd in field", "French bulldog close-up", "Persian cat portrait"],
        settings=_p(PHOTOREAL, model="majicmix", adetailer=True),
    ),
    Preset(
        slug="holiday-seasonal",
        name="Holiday / seasonal",
        category="🍽️ Lifestyle & Culture",
        icon="🎄",
        description="Cozy holiday scene with warm lighting",
        prompt="{SUBJECT}, warm cozy holiday atmosphere, soft glowing string lights, rich seasonal color palette, atmospheric haze, intricate decorative details, hyperdetailed lifestyle photography, magazine cover quality",
        placeholder_subjects=["Christmas tree in country cottage", "autumn fall leaves country lane", "Halloween jack-o-lanterns and mist", "Easter brunch table", "Valentine's roses and candlelight"],
        settings=PHOTOREAL,
    ),

    # 📱 Tech & Brand (4)
    Preset(
        slug="logo-on-photo",
        name="Logo on photo background",
        category="📱 Tech & Brand",
        icon="🏷️",
        description="Your logo woven into a photoreal scene",
        prompt="{SUBJECT}, sharp clean logo composition, brand identity centerpiece, soft cinematic lighting, intricate background texture, hyperdetailed product photography",
        placeholder_subjects=["logo against weathered brick wall", "logo on polished marble", "logo etched in glass", "logo on industrial metal panel"],
        settings=_p(PHOTOREAL, controlnet_scale=1.10, canny_scale=0.60, init_strength=0.20),
        requires_init=True,
    ),
    Preset(
        slug="product-macro",
        name="Product macro",
        category="📱 Tech & Brand",
        icon="📱",
        description="Hyper-detailed product close-up",
        prompt="extreme close-up macro photograph of {SUBJECT}, sharp specular highlights catching every detail, intricate surface texture, dramatic product lighting, shallow depth of field, hyperdetailed editorial product photography",
        placeholder_subjects=["a vintage DTMF keypad", "an antique pocket watch", "a luxury wristwatch movement", "a polished chrome camera lens", "a diamond ring"],
        settings=_p(PHOTOREAL, model="cyberrealistic", qr_coverage=0.85),
    ),
    Preset(
        slug="vintage-telecom",
        name="Vintage telecom / retro tech",
        category="📱 Tech & Brand",
        icon="☎️",
        description="Nostalgic telecom + warm amber light",
        prompt="{SUBJECT}, vintage telecom equipment, polished chrome and signal-blue palette, warm amber backlight, dramatic shadow detail, hyperdetailed nostalgic product photography, 1980s industrial design, anamorphic lens flares, retro analog aesthetic",
        placeholder_subjects=["vintage BellSouth payphone booth", "retro rotary telephone", "rotary dial pulse phone", "Touch-Tone keypad close-up", "vintage telephone switchboard"],
        settings=_p(PHOTOREAL, model="cyberrealistic", controlnet_scale=1.12, qr_coverage=0.78),
    ),
    Preset(
        slug="modern-tech-circuits",
        name="Modern circuits / data",
        category="📱 Tech & Brand",
        icon="💾",
        description="Fiber optic + LEDs + neural patterns",
        prompt="{SUBJECT}, glowing cyan fiber optic cables, server racks with cascading LED indicators, holographic data flow visualizations, electric blue and signal cyan palette, atmospheric volumetric lighting, hyperdetailed cyberpunk technology photography",
        placeholder_subjects=["data center server room corridor", "fiber optic network hub", "neural network visualization sphere", "futuristic AI computing core", "telecom signal tower at night"],
        settings=_p(PHOTOREAL, model="cyberrealistic", controlnet_scale=1.12, qr_coverage=0.78, tile_scale=0.40),
    ),

    # 🎨 Stylized / Artistic (5) — bridge from photoreal to artist tier
    Preset(
        slug="cherry-blossom",
        name="Cherry blossom canopy",
        category="🎨 Stylized / Artistic",
        icon="🌸",
        description="Pink blossoms + blue sky — Reddit-style",
        prompt="{SUBJECT}, dense pink cherry blossom canopy filling the frame, bright blue sky peeking through, soft natural lighting, vivid stylized aesthetic, ultra detailed botanical illustration",
        placeholder_subjects=["massive cherry tree in full bloom", "Japanese garden in spring", "cherry blossom path with petals falling", "ancient cherry tree over a Shinto shrine"],
        settings=_p(ARTIST, qr_coverage=0.70),
        great_fit=True,
    ),
    Preset(
        slug="fantasy-temple",
        name="Fantasy temple cluster",
        category="🎨 Stylized / Artistic",
        icon="🛕",
        description="Gooey-style neon spires + ringed planets",
        prompt="{SUBJECT}, futuristic fantasy neon-lit landscape with dense cluster of ornate Indian temple domes and stupas stretching to the horizon, ringed planets and pastel moons in a vivid cosmic sky, rhythmic repeating bulbous dome shapes, hot amber lanterns burning from stone lattice towers, ultra detailed Midjourney-style atmospheric concept art",
        placeholder_subjects=["holy city of cosmos floating in space", "ringed-planet temple cluster", "neon-lit pagoda metropolis", "stupa-domed celestial city"],
        settings=_p(ARTIST, controlnet_scale=1.25),
        great_fit=True,
    ),
    Preset(
        slug="stained-glass",
        name="Stained glass window",
        category="🎨 Stylized / Artistic",
        icon="🪟",
        description="Vivid geometric cathedral glass",
        prompt="{SUBJECT}, intricate cathedral stained glass window, vivid jewel-tone colors, lead came outlines, sunlight refracting through each pane, dramatic chiaroscuro, ultra detailed gothic architectural illustration",
        placeholder_subjects=["a soaring angel", "rose window pattern", "scene of saints", "abstract geometric pattern"],
        settings=_p(ARTIST, controlnet_scale=1.25, qr_coverage=0.75),
    ),
    Preset(
        slug="watercolor-painterly",
        name="Watercolor painterly",
        category="🎨 Stylized / Artistic",
        icon="🎨",
        description="Soft flowing watercolor",
        prompt="{SUBJECT}, traditional watercolor painting, soft flowing washes of color bleeding into each other, paper texture visible, loose brushstrokes, atmospheric lighting, refined botanical illustration",
        placeholder_subjects=["botanical study of flowers", "misty mountain landscape", "old European village", "field of poppies"],
        settings=_p(ARTIST, model="openjourney", controlnet_scale=1.10, qr_coverage=0.75),
    ),
    Preset(
        slug="neon-synthwave",
        name="Neon synthwave",
        category="🎨 Stylized / Artistic",
        icon="🌅",
        description="80s gradient + grid + vaporwave",
        prompt="{SUBJECT}, 1980s synthwave retrofuture aesthetic, vivid magenta and cyan and orange neon gradient sky, grid horizon extending to infinity, palm tree silhouettes, chrome reflections, vaporwave atmosphere, ultra detailed retro illustration",
        placeholder_subjects=["chrome muscle car on neon highway", "sun setting behind a grid horizon", "Miami Vice beachfront", "outrun-style mountain pass"],
        settings=_p(ARTIST, qr_coverage=0.75),
    ),

    # ⚡ Quick & Special (3)
    Preset(
        slug="quick-iterate",
        name="Quick iterate (Fast mode)",
        category="⚡ Quick & Special",
        icon="⚡",
        description="LCM 6-step for fast prompt tuning",
        prompt="{SUBJECT}, hyperdetailed photography, cinematic lighting",
        placeholder_subjects=["a beautiful landscape", "an animal portrait", "an urban scene"],
        settings=_p(PHOTOREAL, fast_mode=True, candidates=1, steps=6, refine=False, hires_fix=False),
    ),
    Preset(
        slug="print-ready",
        name="Print-ready hi-res",
        category="⚡ Quick & Special",
        icon="🖨️",
        description="Flagship recipe + hi-res for print quality",
        prompt="{SUBJECT}, hyperdetailed photography, cinematic lighting, sharp focus, magazine quality",
        placeholder_subjects=["a stunning landscape", "wildlife portrait", "architectural detail"],
        settings=_p(PHOTOREAL, steps=40, refine_strength=0.40, hires_target=1280, hires_strength=0.18),
    ),
    Preset(
        slug="branded-logo-heavy",
        name="Branded logo (init dominant)",
        category="⚡ Quick & Special",
        icon="🏢",
        description="Your logo's structure heavily preserved",
        prompt="{SUBJECT}, sharp brand identity composition, intricate background texture, hyperdetailed product photography",
        placeholder_subjects=["logo on industrial backdrop", "logo with cinematic lighting"],
        settings=_p(PHOTOREAL, model="cyberrealistic", controlnet_scale=1.12, canny_scale=0.80, init_strength=0.15),
        requires_init=True,
    ),
]


# ── Tier 2: artist-style recipes (30) ────────────────────────────────────────

def _artist(name: str, vocab: str, subjects: list[str], **overrides) -> tuple:
    """Returns (prompt_template, negative_override). The shared artist
    vocabulary suffix + a negative that pushes away from photo aesthetic."""
    prompt = f"{{SUBJECT}} in the style of {name}, {vocab}"
    return prompt, "photograph, photorealistic, low quality, blurry, deformed, watermark, text, signature"


TIER2: list[Preset] = [
    # 🌻 Impressionist & Post-Impressionist (6)
    Preset(
        slug="van-gogh",
        name="Van Gogh",
        category="🌻 Impressionist",
        icon="🌻",
        description="Swirling impasto + vivid blues + yellows",
        prompt="{SUBJECT} in the style of Vincent Van Gogh, swirling impasto brushwork, thick textured oil paint, post-impressionist, vibrant cobalt blues and chrome yellows, expressive emotional brushstrokes, starry night atmosphere",
        placeholder_subjects=["a wheat field at sunset", "starry night village", "vase of sunflowers", "olive grove with cypress trees", "self-portrait"],
        negative_override="photograph, photorealistic, smooth, digital, vector, low detail, blurry, deformed, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="monet",
        name="Monet",
        category="🌻 Impressionist",
        icon="🪷",
        description="Soft impressionist water + reflected light",
        prompt="{SUBJECT} in the style of Claude Monet, soft impressionist brushwork, broken color, dappled light on water, hazy atmospheric haze, pastel pink and lavender and sage palette, plein air oil painting",
        placeholder_subjects=["water lilies on a pond", "Rouen cathedral facade", "haystacks at dawn", "Japanese footbridge", "poppy field"],
        negative_override="photograph, photorealistic, sharp edges, vector, digital, low detail, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="cezanne",
        name="Cézanne",
        category="🌻 Impressionist",
        icon="🍎",
        description="Geometric brushwork + planes of color",
        prompt="{SUBJECT} in the style of Paul Cézanne, structured geometric brushwork, planes of color, post-impressionist still life or landscape, muted ochre and sage and slate palette, visible canvas texture",
        placeholder_subjects=["a still life of apples and a vase", "Mont Sainte-Victoire", "card players in a tavern", "bathers by a river"],
        negative_override="photograph, photorealistic, smooth, digital, blurry, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="renoir",
        name="Renoir",
        category="🌻 Impressionist",
        icon="💐",
        description="Warm impressionist + leisure + soft figures",
        prompt="{SUBJECT} in the style of Pierre-Auguste Renoir, soft warm impressionist brushwork, rosy skin tones, dappled sunlight, lively crowd or intimate moment, French belle epoque palette, oil on canvas",
        placeholder_subjects=["a Parisian café luncheon", "girls dancing at a riverside ball", "a young woman with a parasol", "boating party on a sunlit terrace"],
        negative_override="photograph, photorealistic, harsh, gritty, dark, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="degas",
        name="Degas",
        category="🌻 Impressionist",
        icon="🩰",
        description="Pastel ballet dancers + behind-the-scenes",
        prompt="{SUBJECT} in the style of Edgar Degas, soft pastel impressionist technique, candid behind-the-scenes moment, off-balance composition, warm theater lighting, ballet pinks and stage golds, oil on canvas",
        placeholder_subjects=["ballet dancers backstage", "a single ballerina tying her shoe", "racehorses at the starting line", "café absinthe drinker"],
        negative_override="photograph, photorealistic, posed, vector, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="toulouse-lautrec",
        name="Toulouse-Lautrec",
        category="🌻 Impressionist",
        icon="🎭",
        description="Poster art + theatrical Belle Époque",
        prompt="{SUBJECT} in the style of Henri de Toulouse-Lautrec, bold poster art composition, flat planes of color, expressive line work, theatrical Belle Époque atmosphere, Moulin Rouge cabaret palette of crimson and amber",
        placeholder_subjects=["a Moulin Rouge cabaret dancer", "the Folies Bergère stage", "a singer at a piano", "a horse race at Longchamp"],
        negative_override="photograph, photorealistic, three-dimensional, smooth, watermark",
        settings=ARTIST,
    ),

    # 🟦 Modern & Abstract (6)
    Preset(
        slug="picasso-cubist",
        name="Picasso (cubist)",
        category="🟦 Modern & Abstract",
        icon="🟦",
        description="Fractured planes + multiple viewpoints",
        prompt="{SUBJECT} in the style of Pablo Picasso cubist period, fractured geometric planes, multiple viewpoints overlapping, muted earth tone palette of ochre and slate, analytical cubism, expressive distortion",
        placeholder_subjects=["a guitar still life", "a woman's portrait", "Les Demoiselles d'Avignon style figures", "a bull"],
        negative_override="photograph, photorealistic, three-dimensional, realistic, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="mondrian",
        name="Mondrian",
        category="🟦 Modern & Abstract",
        icon="🟥",
        description="Primary color grid + bold black lines",
        prompt="{SUBJECT} reimagined in the style of Piet Mondrian, geometric grid composition, bold black lines dividing rectangular fields of pure primary red yellow blue and white, neoplasticism, hard-edged abstraction",
        placeholder_subjects=["a city skyline grid", "Broadway Boogie Woogie composition", "a stained glass window", "a hard-edged abstract pattern"],
        negative_override="photograph, photorealistic, gradient, soft, organic, curved, watermark",
        settings=_p(ARTIST, controlnet_scale=1.25, qr_coverage=0.85),
        great_fit=True,
    ),
    Preset(
        slug="kandinsky",
        name="Kandinsky",
        category="🟦 Modern & Abstract",
        icon="🌀",
        description="Floating abstract shapes + vivid color",
        prompt="{SUBJECT} reimagined in the style of Wassily Kandinsky, floating abstract geometric shapes, vivid circles and triangles, expressive lines and color washes, spiritual abstraction, harmonious bold palette",
        placeholder_subjects=["a musical composition visualized", "concentric circles in squares", "abstract dance of forms"],
        negative_override="photograph, photorealistic, representational, recognizable subject, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="klee",
        name="Klee",
        category="🟦 Modern & Abstract",
        icon="🎨",
        description="Whimsical pictographic + childlike forms",
        prompt="{SUBJECT} reimagined in the style of Paul Klee, whimsical pictographic illustration, childlike simplified forms, muted earth tone and lavender palette, dreamy mosaic composition, painted on textured paper",
        placeholder_subjects=["a twittering machine", "a fish in moonlit water", "magic squares of color", "a small fantasy castle"],
        negative_override="photograph, photorealistic, complex, detailed realism, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="pollock",
        name="Pollock",
        category="🟦 Modern & Abstract",
        icon="💧",
        description="Action drip painting + chaotic energy",
        prompt="{SUBJECT} interpreted as action drip painting in the style of Jackson Pollock, dense layered splatters of black and white and ochre and cobalt, all-over composition, expressive abstract expressionism, rhythmic chaos",
        placeholder_subjects=["a chaotic dance of color", "Autumn Rhythm interpretation", "convergence of energetic drips"],
        negative_override="photograph, photorealistic, representational, recognizable, vector, watermark",
        settings=_p(ARTIST, controlnet_scale=1.30, qr_coverage=0.75),
    ),
    Preset(
        slug="rothko",
        name="Rothko (color field)",
        category="🟦 Modern & Abstract",
        icon="🟧",
        description="Large stacked color fields (hardest QR substrate)",
        prompt="{SUBJECT} interpreted in the style of Mark Rothko, soft-edged stacked rectangular color fields, luminous color, contemplative atmosphere, abstract color field painting, gentle gradients within each block",
        placeholder_subjects=["fields of orange and red", "deep blue and crimson harmony", "stacked sage and ochre"],
        negative_override="photograph, photorealistic, hard edges, geometric, representational, watermark",
        # Rothko needs a much higher scale to push QR through flat color blocks
        settings=_p(ARTIST, controlnet_scale=1.35, qr_coverage=0.85, control_start=0.20),
    ),

    # 🌀 Surrealist (4)
    Preset(
        slug="dali",
        name="Dalí",
        category="🌀 Surrealist",
        icon="🕰️",
        description="Melting forms + dreamscape + long shadows",
        prompt="{SUBJECT} in the style of Salvador Dalí, melting surrealist forms, infinite dreamlike landscape with low horizon, long dramatic shadows, soft Mediterranean light, hyperreal yet impossible, oil painting on canvas",
        placeholder_subjects=["melting clocks on a tree", "an elephant on stilt legs", "a barren dream desert", "a face composed of swans"],
        negative_override="photograph, photorealistic snapshot, low quality, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="magritte",
        name="Magritte",
        category="🌀 Surrealist",
        icon="🍎",
        description="Conceptual surrealism + clean-edged",
        prompt="{SUBJECT} in the style of René Magritte, conceptual surrealism, clean-edged Belgian sky-blue and bourgeois charcoal palette, ordinary objects in impossible juxtaposition, calm dreamlike clarity, oil on canvas",
        placeholder_subjects=["a man in bowler hat with apple face", "raining bowler-hatted men", "a window opening onto another sky", "pipe with 'this is not a pipe' inscription"],
        negative_override="photograph, photorealistic, chaotic, messy, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="bosch",
        name="Bosch",
        category="🌀 Surrealist",
        icon="👹",
        description="Medieval surreal + dense crowded figures",
        prompt="{SUBJECT} in the style of Hieronymus Bosch, medieval surreal triptych, densely crowded with fantastical creatures and grotesques, Northern Renaissance oil painting, garden of earthly delights aesthetic, intricate symbolic detail",
        placeholder_subjects=["a fantastical paradise garden", "creatures in a hellscape", "saints among demons", "a crowd of strange beings"],
        negative_override="photograph, photorealistic, modern, minimalist, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="escher",
        name="M.C. Escher",
        category="🌀 Surrealist",
        icon="♾️",
        description="Impossible geometry + tessellation",
        prompt="{SUBJECT} in the style of M.C. Escher, impossible geometric architecture, tessellating interlocking shapes, black-and-white lithograph crosshatching, mathematical precision, paradoxical perspective",
        placeholder_subjects=["an impossible staircase", "tessellating fish and birds", "a relativity-style endless stairway", "hands drawing each other"],
        negative_override="photograph, photorealistic, color, soft, watermark",
        settings=_p(ARTIST, qr_coverage=0.75),
        great_fit=True,
    ),

    # 👑 Classic Masters (5)
    Preset(
        slug="rembrandt",
        name="Rembrandt",
        category="👑 Classic Masters",
        icon="🕯️",
        description="Chiaroscuro + Dutch Golden Age",
        prompt="{SUBJECT} in the style of Rembrandt van Rijn, dramatic chiaroscuro lighting, Dutch Golden Age oil painting, deep umber and gold palette, weathered painterly texture, intense emotional gaze, single light source emerging from darkness",
        placeholder_subjects=["a self-portrait", "the night watch militia", "an old man's portrait", "a philosopher with book"],
        negative_override="photograph, photorealistic, bright, flat lighting, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="vermeer",
        name="Vermeer",
        category="👑 Classic Masters",
        icon="🪟",
        description="Soft window light + Dutch interior",
        prompt="{SUBJECT} in the style of Johannes Vermeer, soft northern window light, Dutch interior scene, luminous textures of fabric and pearl and glass, blue and ochre palette, contemplative quiet moment, oil on canvas",
        placeholder_subjects=["girl with a pearl earring", "milkmaid pouring milk", "lacemaker at work", "woman reading a letter by window"],
        negative_override="photograph, photorealistic, harsh, low quality, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="caravaggio",
        name="Caravaggio",
        category="👑 Classic Masters",
        icon="🗡️",
        description="Baroque chiaroscuro + dramatic moment",
        prompt="{SUBJECT} in the style of Caravaggio, intense Baroque chiaroscuro, dramatic moment captured, deep shadows with raking light, naturalistic figures, oil painting, dark theatrical atmosphere",
        placeholder_subjects=["the calling of Saint Matthew", "Judith beheading Holofernes", "the supper at Emmaus", "Bacchus with grapes"],
        negative_override="photograph, photorealistic, bright, flat, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="hokusai",
        name="Hokusai",
        category="👑 Classic Masters",
        icon="🌊",
        description="Japanese woodblock print + flat color",
        prompt="{SUBJECT} in the style of Katsushika Hokusai, traditional Japanese ukiyo-e woodblock print, bold black outlines, flat planes of indigo blue and ochre and white, hand-printed paper texture, vintage Edo period",
        placeholder_subjects=["the great wave with Mount Fuji", "Mount Fuji from various angles", "samurai under cherry blossoms", "snow-covered shrine", "geisha with parasol"],
        negative_override="photograph, photorealistic, three-dimensional, shaded, gradient, digital, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="klimt",
        name="Klimt",
        category="👑 Classic Masters",
        icon="✨",
        description="Gold leaf + ornate Byzantine pattern",
        prompt="{SUBJECT} in the style of Gustav Klimt, gold leaf and ornate Byzantine patterns, decorative spirals and eyes and geometric tiles, deep emerald and ruby and shimmering gold palette, art nouveau symbolism, oil and gold on canvas",
        placeholder_subjects=["the kiss embrace of lovers", "a woman in golden robes", "the tree of life", "Adele Bloch-Bauer portrait"],
        negative_override="photograph, photorealistic, plain, minimalist, modern, watermark",
        settings=_p(ARTIST, qr_coverage=0.75),
        great_fit=True,
    ),

    # 💥 Pop & Street (4)
    Preset(
        slug="warhol",
        name="Warhol",
        category="💥 Pop & Street",
        icon="🥫",
        description="Silk-screen pop art + repeated panels",
        prompt="{SUBJECT} in the style of Andy Warhol pop art, silk-screen technique with flat planes of vivid color, high-contrast portrait or commercial object, neon palette of magenta orange and electric green, 1960s pop sensibility",
        placeholder_subjects=["a Campbell's soup can", "Marilyn Monroe portrait quartet", "Elvis Presley silhouette", "pop celebrity grid"],
        negative_override="photograph, photorealistic, soft, painterly, gradients, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="lichtenstein",
        name="Lichtenstein",
        category="💥 Pop & Street",
        icon="💥",
        description="Ben-Day dots + comic panel pop",
        prompt="{SUBJECT} in the style of Roy Lichtenstein pop art, comic book panel composition, thick black outlines, primary red blue and yellow palette, Ben-Day dot halftone pattern, bold speech bubbles, retro 1960s comic illustration",
        placeholder_subjects=["a girl crying with a tear", "a fighter jet exploding", "a girl drowning", "a heroic kiss scene"],
        negative_override="photograph, photorealistic, gradient, soft, watermark",
        settings=ARTIST,
        great_fit=True,
    ),
    Preset(
        slug="banksy",
        name="Banksy",
        category="💥 Pop & Street",
        icon="🎨",
        description="Street stencil + spray paint on wall",
        prompt="{SUBJECT} in the style of Banksy, street art stencil with spray paint, weathered urban wall background with graffiti and texture, satirical political commentary, monochrome black silhouettes with single accent color",
        placeholder_subjects=["a girl releasing a balloon heart", "rats with picket signs", "a flower thrower", "kids playing with surveillance camera"],
        negative_override="photograph, photorealistic, painterly, soft, watermark",
        settings=_p(ARTIST, qr_coverage=0.78),
        requires_init=False,
    ),
    Preset(
        slug="basquiat",
        name="Basquiat",
        category="💥 Pop & Street",
        icon="👑",
        description="Neo-expressionist + scribbled symbols",
        prompt="{SUBJECT} in the style of Jean-Michel Basquiat, neo-expressionist raw painting, scribbled black outlines, scrawled text and crowns and skeletons, layered graffiti aesthetic, urgent emotional energy, oilstick on canvas",
        placeholder_subjects=["a figure with three-point crown", "a skeleton portrait", "a chaotic mixed-media composition", "Pez dispenser self-portrait"],
        negative_override="photograph, photorealistic, clean, smooth, vector, watermark",
        settings=ARTIST,
    ),

    # 🖋️ Stylized & Illustration (5)
    Preset(
        slug="mucha",
        name="Mucha (Art Nouveau)",
        category="🖋️ Stylized & Illustration",
        icon="🪷",
        description="Ornate flowing borders + flowing hair",
        prompt="{SUBJECT} in the style of Alphonse Mucha, Art Nouveau ornate flowing borders, sinuous decorative lines, elegant figure surrounded by flowing hair and botanical motifs, pastel cream and gold and sage palette, lithographic poster illustration",
        placeholder_subjects=["an Art Nouveau goddess", "a season personification", "a Slavic mythological figure", "an elegant woman with flowing hair"],
        negative_override="photograph, photorealistic, modern, minimalist, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="ghibli",
        name="Miyazaki / Ghibli",
        category="🖋️ Stylized & Illustration",
        icon="🌫️",
        description="Soft anime watercolor landscape",
        prompt="{SUBJECT}, Studio Ghibli anime aesthetic in the style of Hayao Miyazaki, soft watercolor and ink illustration, lush nature with floating elements, golden hour light, whimsical childhood wonder, Spirited Away atmospheric",
        placeholder_subjects=["a flying castle in clouds", "a girl in a sunflower field", "a forest spirit at a shrine", "an old steam train through countryside"],
        negative_override="photograph, photorealistic, dark, gritty, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="hopper",
        name="Edward Hopper",
        category="🖋️ Stylized & Illustration",
        icon="🍳",
        description="American realism + melancholic light",
        prompt="{SUBJECT} in the style of Edward Hopper, American realism painting, melancholic atmosphere, dramatic raking sunlight from windows, isolated figures, muted ochre and sage and crimson palette, mid-century American scene",
        placeholder_subjects=["a late-night diner with single patron", "a sunlit empty street at dawn", "a woman by a hotel window", "a lighthouse at high noon"],
        negative_override="photograph, photorealistic, busy, crowded, joyful, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="frazetta",
        name="Frazetta (heroic fantasy)",
        category="🖋️ Stylized & Illustration",
        icon="⚔️",
        description="Dramatic muscle + warrior fantasy",
        prompt="{SUBJECT} in the style of Frank Frazetta, heroic fantasy oil painting, dramatic muscular figures in dynamic action, ominous sky, rich earth tone palette, sword and sorcery atmosphere, painterly brushwork",
        placeholder_subjects=["a barbarian warrior with sword", "Conan on a throne of skulls", "a Death Dealer rider", "a fierce sorceress on dragon"],
        negative_override="photograph, photorealistic, modern, soft, watermark",
        settings=ARTIST,
    ),
    Preset(
        slug="beksinski",
        name="Beksiński",
        category="🖋️ Stylized & Illustration",
        icon="💀",
        description="Dystopian surreal + rusted dread",
        prompt="{SUBJECT} in the style of Zdzisław Beksiński, dystopian surreal landscape, towering rusted twisted architecture, foreboding atmosphere, sepia and rust and bone palette, oil painting on canvas, haunting otherworldly composition",
        placeholder_subjects=["towering rusted cathedral ruins", "a skeletal figure in a vast wasteland", "a colossal monolithic structure", "wandering souls in a dead landscape"],
        negative_override="photograph, photorealistic, bright, colorful, cheerful, watermark",
        settings=ARTIST,
    ),
]


PRESETS: list[Preset] = TIER1 + TIER2
PRESETS_BY_SLUG: dict[str, Preset] = {p.slug: p for p in PRESETS}
