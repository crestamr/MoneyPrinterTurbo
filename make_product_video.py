"""Make a vertical UGC product ad from a TikTok Shop link or manual details.

    # see the prompts without spending anything
    python make_product_video.py --url "https://shop.tiktok.com/jp/pdp/123" --dry-run

    # real run, own photos (better input than a listing thumbnail)
    python make_product_video.py --name "Portable Shaver" \
        --image photos/hero.png --image photos/in-hand.png \
        --features "net blade,low irritation,travel size" --scenes 4

Every scene is one paid MiniMax generation, so the cost is printed and
confirmed before anything is sent.
"""

from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from app.models.schema import VideoAspect
from app.services import minimax_media, product_pipeline, product_video
from app.utils import utils


def _download(url: str, index: int) -> str:
    """Pull a remote image local so it is validated and uploaded by us.

    MiniMax accepts a public URL, but a listing CDN may refuse an unknown
    fetcher or serve a resized thumbnail below the 256px floor. Fetching it
    here means a clear local error instead of an opaque remote rejection.
    """
    import requests

    cache_dir = utils.storage_dir("product_images", create=True)
    suffix = os.path.splitext(url.split("?")[0])[1].lower() or ".jpg"
    if suffix not in minimax_media.SUPPORTED_IMAGE_EXTENSIONS:
        suffix = ".jpg"
    path = os.path.join(cache_dir, f"{utils.md5(url)[:12]}-{index}{suffix}")
    if os.path.exists(path):
        return path
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://shop.tiktok.com/"},
    )
    if response.status_code != 200:
        raise SystemExit(
            f"could not download image (HTTP {response.status_code}): {url}\n"
            f"Save it manually and pass it with --image instead."
        )
    with open(path, "wb") as handle:
        handle.write(response.content)
    logger.info(f"downloaded product image -> {path}")
    return path


def _duration(path: str) -> float:
    return minimax_media._video_duration(path)


def _build_product(args) -> product_video.ProductInfo:
    images = list(args.image or [])
    features = [f.strip() for f in (args.features or "").split(",") if f.strip()]

    product = None
    if args.url:
        from app.services import tiktok_shop

        try:
            product = product_pipeline.product_from_tiktok(
                args.url,
                name=args.name,
                images=images,
                features=features,
                audience=args.audience,
            )
        except tiktok_shop.TikTokShopError as exc:
            # The listing is JavaScript-rendered and intermittently comes back
            # as an empty shell. That must not be fatal when the user already
            # gave us everything the listing would have: fall through to the
            # manual path instead of dying with a traceback.
            if not (args.name and images):
                raise SystemExit(
                    f"could not read the TikTok listing: {exc}\n"
                    f"Pass --name and --image to continue without it."
                )
            logger.warning(f"TikTok listing unavailable, using --name/--image: {exc}")
    if product is None:
        if not args.name or not images:
            raise SystemExit("without --url you must pass --name and at least one --image")
        product = product_video.ProductInfo(
            name=args.name, images=images, features=features, audience=args.audience
        )

    # Localise anything remote so validation happens before we pay.
    localised = []
    for index, source in enumerate(product.images):
        if source.startswith(("http://", "https://")):
            localised.append(_download(source, index))
        else:
            localised.append(source)
    product.images = localised

    for image in product.images:
        try:
            minimax_media.validate_image(image)
        except minimax_media.MiniMaxMediaError as exc:
            raise SystemExit(f"product image rejected: {exc}")
    return product


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="", help="TikTok Shop product URL")
    parser.add_argument("--name", default="", help="product name (overrides the listing)")
    parser.add_argument("--image", action="append", help="product image; repeatable")
    parser.add_argument("--features", default="", help="comma-separated key features")
    parser.add_argument("--audience", default="", help="who it is for")
    parser.add_argument("--presenter", default=product_video.DEFAULT_PRESENTER)
    parser.add_argument(
        "--character", default="",
        help="photo or video of the on-camera presenter, kept consistent across "
             "scenes; a frame is taken from a video")
    parser.add_argument(
        "--character-at", type=float, default=None,
        help="seconds into --character video to grab the frame from "
             "(default: 35%% in). Use this to pick a frame that does not show "
             "a different product.")
    parser.add_argument(
        "--outfit", default="",
        help='what the presenter wears in every scene, e.g. "a plain grey t-shirt"')
    parser.add_argument("--scenes", type=int, default=product_video.DEFAULT_SCENE_COUNT)
    parser.add_argument("--seconds", type=int, default=product_pipeline.DEFAULT_CLIP_SECONDS,
                        help="seconds per scene (MiniMax allows 4-15)")
    parser.add_argument("--out", default="", help="output mp4 path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompts, generate nothing, spend nothing")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)

    product = _build_product(args)
    logger.info(f"product: {product.name} ({len(product.images)} image(s))")

    character = ""
    if args.character:
        character = args.character
        if minimax_media.is_video(character):
            position = (
                args.character_at / max(0.001, _duration(character))
                if args.character_at is not None
                else minimax_media.FRAME_POSITION
            )
            character = minimax_media.extract_frame(
                args.character, position=min(max(position, 0.0), 0.99)
            )
            print(f"character frame: {character}")
            print(
                "  check this frame: anything else held in it (a different "
                "product, say) can bleed into the generated video. Use "
                "--character-at SECONDS to pick another."
            )
        try:
            minimax_media.validate_image(character)
        except minimax_media.MiniMaxMediaError as exc:
            raise SystemExit(f"character reference rejected: {exc}")

    if not args.dry_run and not args.yes:
        count = product_video.estimate_generations(scene_count=args.scenes)
        answer = input(
            f"This will run {count} paid MiniMax generation(s). Continue? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("cancelled")
            return 1

    result = product_pipeline.create_product_video(
        product,
        scene_count=args.scenes,
        clip_seconds=args.seconds,
        output_path=args.out,
        presenter=args.presenter,
        aspect=VideoAspect.portrait,
        character_image=character,
        outfit=args.outfit,
        dry_run=args.dry_run,
    )

    for index, (scene, prompt) in enumerate(zip(result.scenes, result.prompts), start=1):
        print(f"\n--- scene {index}: {scene.setting} ---")
        print(f'dialogue: "{scene.dialogue}"')
        if args.dry_run:
            print(prompt)

    if result.failures:
        print("\nproblems:")
        for failure in result.failures:
            print(f"  - {failure}")

    if result.ok:
        print(f"\nwrote {result.output_path}")
        return 0
    if args.dry_run:
        print(f"\ndry run: {len(result.prompts)} prompt(s) built, nothing spent")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
