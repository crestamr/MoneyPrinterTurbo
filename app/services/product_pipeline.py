"""End-to-end: a product plus its photos becomes a vertical UGC ad.

Deliberately separate from ``app.services.task``. That pipeline is built around
topic -> script -> search terms -> stock footage -> TTS -> subtitles, and this
one inverts most of it: the visuals come from the product images, and the
*model* speaks the dialogue on camera rather than a TTS track being laid over
silent footage. Forcing this through the existing task flow would mean
disabling half of it.

Every clip is a paid API generation, so the cost is reported up front and a
dry run renders the prompts without spending anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from loguru import logger

from app.models.schema import VideoAspect
from app.services import minimax_media, minimax_video, product_video, video
from app.services.product_video import ProductInfo, Scene
from app.utils import utils

DEFAULT_CLIP_SECONDS = 6
DEFAULT_THREADS = 2


@dataclass(frozen=True)
class ProductVideoRequest:
    """One queued product-video job.

    Frozen and self-contained: the WebUI hands this to a background worker and
    then keeps rerunning, so the job must not share mutable state with the page.
    """

    product: ProductInfo
    scene_count: int = product_video.DEFAULT_SCENE_COUNT
    clip_seconds: int = DEFAULT_CLIP_SECONDS
    character_image: str = ""
    outfit: str = ""
    output_path: str = ""
    aspect: VideoAspect = VideoAspect.portrait


@dataclass
class ProductVideoResult:
    """What a run produced, including what it failed to produce."""

    output_path: str = ""
    scenes: list[Scene] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    clip_paths: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    generations_spent: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.output_path)


def build_prompts(
    product: ProductInfo,
    scenes: "list[Scene]",
    *,
    presenter: str = product_video.DEFAULT_PRESENTER,
    outfit: str = "",
) -> list[str]:
    return [
        product_video.build_scene_prompt(
            product, scene, presenter=presenter, outfit=outfit
        )
        for scene in scenes
    ]


def create_product_video(
    product: ProductInfo,
    *,
    scene_count: int = product_video.DEFAULT_SCENE_COUNT,
    clip_seconds: int = DEFAULT_CLIP_SECONDS,
    output_path: str = "",
    presenter: str = product_video.DEFAULT_PRESENTER,
    aspect: VideoAspect = VideoAspect.portrait,
    scenes: "list[Scene] | None" = None,
    character_image: str = "",
    outfit: str = "",
    dry_run: bool = False,
) -> ProductVideoResult:
    """Generate one product ad end to end.

    ``scenes`` may be supplied to skip the LLM and re-render a known-good set -
    useful when only the visuals need another attempt, since regenerating
    scenes would otherwise change the script too.

    ``character_image`` pins the on-camera presenter across scenes. Without it
    the model invents a different person each time, which reads as several
    unrelated clips rather than one creator's video. It may be a photo or a
    video (a frame is taken from the body of the clip).

    ``dry_run`` builds and returns the prompts without calling the video API,
    so the wording can be reviewed before any credits are spent.
    """
    result = ProductVideoResult()

    result.scenes = list(scenes) if scenes else product_video.generate_scenes(
        product, scene_count=scene_count
    )
    if not result.scenes:
        result.failures.append("no scenes were produced; check the LLM provider")
        return result

    result.prompts = build_prompts(
        product, result.scenes, presenter=presenter, outfit=outfit
    )
    logger.info(
        f"product video: {len(result.scenes)} scene(s), "
        f"{product_video.estimate_generations(scene_count=len(result.scenes))} "
        f"paid generation(s)"
    )
    if dry_run:
        logger.info("dry run: prompts built, no video generated")
        return result

    # Character first, matching the order the prompt names them in
    # ("Character reference the on-camera creator, <product> reference the
    # product"). Up to 9 reference images are allowed in one request.
    references = ([character_image] if character_image else []) + list(product.images)

    # Upload every reference once for the whole run rather than once per
    # scene, and fail before any paid generation if one is rejected.
    try:
        references = minimax_video.resolve_references(references)
    except (minimax_video.MiniMaxVideoError, minimax_media.MiniMaxMediaError) as exc:
        result.failures.append(f"reference images could not be prepared: {exc}")
        return result

    for index, prompt in enumerate(result.prompts, start=1):
        logger.info(f"generating scene {index}/{len(result.prompts)}")
        materials = minimax_video.generate_product_clip(
            prompt, clip_seconds, references, aspect
        )
        result.generations_spent += 1
        if not materials:
            result.failures.append(f"scene {index} produced no clip")
            continue
        result.clip_paths.append(materials[0].url)

    if not result.clip_paths:
        result.failures.append("every scene failed; nothing to assemble")
        return result

    output_dir = utils.storage_dir("product_videos", create=True)
    if not output_path:
        output_path = os.path.join(
            output_dir, f"product-{utils.md5(product.name)[:8]}.mp4"
        )

    # ffmpeg will not create a missing parent directory, and a --out path
    # pointing somewhere that does not exist yet would otherwise throw away
    # clips that were already paid for.
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        video.concat_video_clips_with_ffmpeg(
            result.clip_paths,
            output_path,
            threads=DEFAULT_THREADS,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - surface, do not crash the caller
        logger.error(f"failed to assemble product video: {type(exc).__name__}: {exc}")
        result.failures.append(f"assembly failed: {exc}")
        return result

    result.output_path = output_path
    logger.info(f"product video written: {output_path}")
    return result


def product_from_tiktok(url: str, **overrides) -> ProductInfo:
    """Build a ProductInfo from a TikTok Shop link, with manual overrides.

    Whatever the listing yields is a draft: the caller should let a human edit
    the name and, ideally, replace the listing thumbnail with real product
    photography, which generates noticeably better video.
    """
    from app.services import tiktok_shop

    parsed = tiktok_shop.load_product(url)
    fields = {
        "name": parsed.title or "this product",
        "images": list(parsed.image_urls),
        "source_url": parsed.source_url or url,
    }
    fields.update({k: v for k, v in overrides.items() if v})
    return ProductInfo(**fields)
