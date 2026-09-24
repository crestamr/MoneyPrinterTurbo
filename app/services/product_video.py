"""Product-showcase scene prompts for image-conditioned video generation.

The prompt template here is modelled on the reference set in
``resource/renoise/prompts.json``. Those prompts are long on purpose: most of
their length is boilerplate that suppresses the failure modes which make AI
product video look fake - drifting product geometry, commercial colour grading,
teleprompter delivery, extra fingers. Shortening the template makes the output
visibly worse, so it is kept verbatim and only the per-scene parts vary.

Anatomy of one scene prompt:

1. reference declaration - which attached image is the product, which the person
2. style + camera - vertical UGC, phone texture, one continuous take
3. audio - no music, so the spoken line survives
4. action - how the product is physically handled on camera
5. dialogue - the spoken line, in quotes
6. timing beats - what happens when
7. authenticity constraints - not a commercial, no overacting
8. capture texture - real skin, phone exposure, no beauty filters
9. negative prompt - the long list of things not to do

Note the presenter **speaks on camera** and the model lip-syncs. That is what
sells the format, so a caller should keep the generated audio rather than
overlaying TTS, and can burn subtitles from ``Scene.dialogue`` which is known
up front.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from loguru import logger

from app.services import llm

# MiniMax caps the text content item at 7000 characters.
MAX_PROMPT_CHARS = 7000

# Marketplace titles run to 150+ characters of keywords; anything past this is
# search padding rather than a name a person would say out loud.
MAX_DISPLAY_NAME_CHARS = 48

DEFAULT_SCENE_COUNT = 4
DEFAULT_PRESENTER = "a young person"

STYLE_BLOCK = (
    "Short-form vertical UGC lifestyle video, authentic smartphone footage "
    "texture. Fixed camera angle, no editing throughout, no transitions, "
    "single continuous long take, medium shot."
)

AUDIO_BLOCK = "No music, keep only vocals and sound effects."

AUTHENTICITY_BLOCK = (
    "The overall performance looks like an authentic product sharing video "
    "casually shot by a lifestyle blogger, not like a traditional commercial. "
    "No deliberate selling, no overacting, no reading from a teleprompter. The "
    "speaking pace is natural and clear, with a conversational tone, and lip "
    "movements are accurately synced with the dialogue. The colour, material, "
    "size and structure of the product remain consistent throughout."
)

TEXTURE_BLOCK = (
    "Authentic skin texture, realistic smartphone camera exposure, slight video "
    "noise, natural ambient sound, no beauty filters, no skin smoothing, no "
    "HDR, no commercial studio look."
)

NEGATIVE_BLOCK = (
    "Negative prompt: Camera movement, editing, transitions, pan/tilt/zoom, "
    "slow motion, exaggerated acting, obviously reciting ad copy, excessive "
    "smiling, multiple people in frame, product deformation, product colour "
    "changes, product structure changes, product suddenly disappearing, hand "
    "deformities, extra fingers, fingers clipping, oily skin, plastic skin, "
    "excessive skin smoothing, commercial ad colour grading, subtitles, "
    "stickers, watermarks, logos, background music, lip sync mismatch, "
    "outfit changes, clothing changes, different person."
)

# Scenes are generated independently, so nothing carries over between them
# unless the prompt demands it. The first 4-scene run kept the face but put
# scene 1 in a different top. This sits in the head, which is never truncated.
CONTINUITY_BLOCK = (
    "The creator's face, hairstyle and clothing match the character "
    "reference and stay exactly the same in every scene."
)

_SCENE_SYSTEM_PROMPT = """You are a short-form UGC ad director for TikTok Shop.

Given a product, write {scene_count} scene(s) for a vertical product video that
looks like a real customer filmed it on their phone - not a commercial.

Rules:
- Each scene is ONE continuous take of a single person handling the product.
- "beats" are 2-4 short physical actions, in order, describing how the product
  is held, opened, used or demonstrated. Be concrete and physical.
- Write beats in the THIRD PERSON and present tense, describing the person from
  the outside: "lifts the lid", "presses their back into the pad". Never write
  "I" or "my" - the surrounding prompt describes the person in third person and
  mixing the two makes it ambiguous who is acting.
- "dialogue" is ONE sentence that person says to camera, casual and specific.
  Never say "buy now" or use ad language. Mention a real benefit.
- "setting" is a short phrase naming a believable everyday location.
- Do not mention brands other than the product itself.

Product: {name}
{details}

Return ONLY JSON in exactly this shape, no prose, no code fence:
{{"scenes": [{{"setting": "...", "beats": ["...", "..."], "dialogue": "..."}}]}}
"""


class ProductVideoError(RuntimeError):
    """Raised when a product cannot be turned into scenes."""


@dataclass
class ProductInfo:
    """The product being advertised. ``images`` feed the model as references."""

    name: str
    images: list[str]
    features: list[str] = field(default_factory=list)
    price: str = ""
    audience: str = ""
    source_url: str = ""

    def __post_init__(self):
        self.name = (self.name or "").strip()
        if not self.name:
            raise ValueError("product name is required")
        self.images = [str(i).strip() for i in (self.images or []) if str(i).strip()]
        if not self.images:
            raise ValueError("at least one product image is required")
        self.features = [str(f).strip() for f in (self.features or []) if str(f).strip()]

    @property
    def display_name(self) -> str:
        """A short name to say on camera.

        Marketplace titles are keyword-stuffed for search ("Dowinx ...
        165 degrees reclining chair high back grey LS-6679"), and the raw
        string appears three times in every scene prompt, crowding out the
        parts that shape the shot. Keep the leading words, which are reliably
        the brand and product type, and let --name override when it matters.
        """
        name = self.name
        if len(name) <= MAX_DISPLAY_NAME_CHARS:
            return name
        words = name.split()
        short = ""
        for word in words:
            candidate = f"{short} {word}".strip()
            if len(candidate) > MAX_DISPLAY_NAME_CHARS:
                break
            short = candidate
        return short or name[:MAX_DISPLAY_NAME_CHARS].rstrip()

    def details_block(self) -> str:
        lines = []
        if self.features:
            lines.append("Key features: " + "; ".join(self.features))
        if self.audience:
            lines.append(f"Target audience: {self.audience}")
        if self.price:
            lines.append(f"Price: {self.price}")
        return "\n".join(lines)


@dataclass
class Scene:
    """One continuous take."""

    setting: str
    beats: list[str]
    dialogue: str

    def __post_init__(self):
        self.setting = (self.setting or "").strip()
        self.beats = [str(b).strip() for b in (self.beats or []) if str(b).strip()]
        self.dialogue = (self.dialogue or "").strip()


def build_scene_prompt(
    product: ProductInfo,
    scene: Scene,
    *,
    presenter: str = DEFAULT_PRESENTER,
    outfit: str = "",
) -> str:
    """Render one scene into a MiniMax text prompt.

    Truncated to MAX_PROMPT_CHARS from the *action* side, so the style,
    authenticity and negative blocks - the parts that keep the output from
    looking like an advert - always survive.
    """
    beats = scene.beats or ["shows the product to the camera"]
    name = product.display_name
    timing = "; ".join(f"Beat {i}: {b}" for i, b in enumerate(beats, start=1))
    # "holding" only fits handheld goods - furniture and appliances are
    # demonstrated, not held - so the opening line stays neutral.
    action = (
        f"{presenter.capitalize()} is presenting the {name}, facing the camera "
        f"in {scene.setting or 'an everyday setting'}. "
        + _join_beats(beats)
    )
    dialogue_block = (
        "While showing the product, they look at the camera and speak in a "
        "relaxed, natural voice with a genuine user-experience vibe:\n\n"
        f"“{scene.dialogue}”"
    )

    continuity = CONTINUITY_BLOCK
    if outfit.strip():
        continuity = f"They wear {outfit.strip()}. {continuity}"
    head = (
        f"Character reference the on-camera creator, {name} reference "
        f"the product. {continuity} {STYLE_BLOCK}\n\n{AUDIO_BLOCK}\n\n"
    )
    tail = (
        f"\n\nTiming: {timing}.\n\n{AUTHENTICITY_BLOCK}\n\n{TEXTURE_BLOCK}\n\n"
        f"{NEGATIVE_BLOCK}"
    )

    budget = MAX_PROMPT_CHARS - len(head) - len(tail) - len(dialogue_block) - 8
    if budget < 0:
        # Nothing variable left to cut; drop beats detail rather than the
        # boilerplate, then hard-truncate as a last resort.
        return (head + dialogue_block + tail)[:MAX_PROMPT_CHARS]
    if len(action) > budget:
        action = action[: max(0, budget - 1)].rstrip() + "."

    return f"{head}{action}\n\n{dialogue_block}{tail}"[:MAX_PROMPT_CHARS]


def _join_beats(beats: "list[str]") -> str:
    """Join beats into one readable sentence.

    Naive joining produced "lifts it toward the desk Then Presses the lumbar
    pad", which reads as two broken fragments and invites the model to render
    the word rather than the action.
    """
    cleaned = []
    for beat in beats:
        beat = beat.strip().rstrip(".")
        if not beat:
            continue
        if cleaned and beat[:1].isupper() and not beat.split(" ")[0].isupper():
            beat = beat[0].lower() + beat[1:]
        cleaned.append(beat)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0] + "."
    return cleaned[0] + ", then " + ", then ".join(cleaned[1:]) + "."


def _strip_fence(text: str) -> str:
    match = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*)\n```\s*$", text or "", re.S)
    return match.group(1) if match else (text or "")


def parse_scenes(response: str) -> list[Scene]:
    """Parse an LLM response into scenes, tolerating the usual wrappers.

    Returns [] rather than raising: a malformed response is a quality problem
    for the caller to report, not a crash.
    """
    raw = _strip_fence(str(response or "")).strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        # Last resort: the first {...} or [...] block in the text.
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if not match:
            logger.warning("product scene response was not JSON")
            return []
        try:
            data = json.loads(match.group(1))
        except ValueError:
            logger.warning("product scene response was not JSON")
            return []

    items = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    scenes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scene = Scene(
            setting=item.get("setting", ""),
            beats=item.get("beats", []) or [],
            dialogue=item.get("dialogue", ""),
        )
        if not scene.dialogue:
            # A silent scene defeats the format; skip rather than ship it.
            continue
        scenes.append(scene)
    return scenes


def generate_scenes(
    product: ProductInfo, *, scene_count: int = DEFAULT_SCENE_COUNT
) -> list[Scene]:
    """Ask the configured LLM for scenes. Returns [] on any failure."""
    prompt = _SCENE_SYSTEM_PROMPT.format(
        scene_count=max(1, int(scene_count)),
        name=product.name,
        details=product.details_block(),
    )
    try:
        response = llm._generate_response(prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - provider-specific failures vary
        logger.error(
            f"product scene generation failed: {type(exc).__name__}: {exc}"
        )
        return []
    scenes = parse_scenes(response)
    if not scenes:
        logger.error("product scene generation returned no usable scenes")
    return scenes


def estimate_generations(*, scene_count: int) -> int:
    """How many paid video generations a run will cost.

    Each scene is one API call. Surfaced to the user before starting, because
    unlike the local qwen_image path this bills per clip.
    """
    return max(0, int(scene_count))
