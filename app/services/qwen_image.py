"""Local Qwen-Image-2.1 image generation via a running ComfyUI server.

This module deliberately stays independent of ``app.services.material``
(no import in either direction) so ``material.py`` can import this module
for its provider dispatch without a circular import — the same isolation
``minimax_video.py`` uses for the same reason.
"""

from __future__ import annotations

import os
import random
import time
import uuid

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import llm, video
from app.utils import utils

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_UNET_NAME = "qwen-image-2.1-Q8_0.gguf"
DEFAULT_CLIP_NAME = "qwen3vl_8b_bf16.safetensors"
DEFAULT_VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
DEFAULT_STEPS = 20
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_TIMEOUT_SECONDS = 180.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0


class QwenImageError(RuntimeError):
    """Base error for the local Qwen-Image-2.1 integration."""


class QwenImageAPIError(QwenImageError):
    """Raised when ComfyUI rejects a request or a generation errors."""


def _round_to_multiple_of_32(value: int) -> int:
    return max(32, round(value / 32) * 32)


def _get_base_url() -> str:
    return str(
        config.qwen_image.get("base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def is_comfyui_reachable(base_url: str | None = None) -> bool:
    """Return True if the configured ComfyUI server responds."""
    url = base_url or _get_base_url()
    try:
        response = requests.get(url, timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, 10.0))
        return response.status_code == 200
    except requests.RequestException:
        return False


def _build_plain_t2i_graph(
    prompt: str, width: int, height: int, steps: int, seed: int
) -> dict:
    unet_name = str(
        config.qwen_image.get("unet_name", DEFAULT_UNET_NAME) or DEFAULT_UNET_NAME
    )
    clip_name = str(
        config.qwen_image.get("clip_name", DEFAULT_CLIP_NAME) or DEFAULT_CLIP_NAME
    )
    vae_name = str(
        config.qwen_image.get("vae_name", DEFAULT_VAE_NAME) or DEFAULT_VAE_NAME
    )
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "resolution": max(width, height),
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["3", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "qwen_image_mpt"},
        },
    }


def _submit_and_wait(
    graph: dict, base_url: str, poll_interval: float, poll_timeout: float
) -> dict:
    response = requests.post(
        f"{base_url}/prompt",
        json={"prompt": graph},
        timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS),
    )
    if response.status_code != 200:
        raise QwenImageAPIError(
            f"ComfyUI rejected the prompt: HTTP {response.status_code}: "
            f"{str(getattr(response, 'text', ''))[:200]}"
        )
    response_json = response.json()
    prompt_id = (
        response_json.get("prompt_id") if isinstance(response_json, dict) else None
    )
    if not prompt_id:
        raise QwenImageAPIError(
            f"ComfyUI /prompt response missing prompt_id: {response_json}"
        )

    deadline = time.monotonic() + poll_timeout
    while time.monotonic() < deadline:
        history_response = requests.get(
            f"{base_url}/history/{prompt_id}",
            timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        )
        if history_response.status_code != 200:
            raise QwenImageAPIError(
                f"ComfyUI /history/{prompt_id} returned HTTP "
                f"{history_response.status_code}: "
                f"{str(getattr(history_response, 'text', ''))[:200]}"
            )
        history = history_response.json()
        entry = history.get(prompt_id) if isinstance(history, dict) else None
        if isinstance(entry, dict):
            status = entry.get("status", {})
            if not isinstance(status, dict):
                status = {}
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise QwenImageAPIError(
                    f"ComfyUI generation failed for prompt {prompt_id}"
                )
        time.sleep(poll_interval)

    raise QwenImageAPIError(
        f"timed out waiting for ComfyUI prompt {prompt_id} after {poll_timeout}s"
    )


def _fetch_output_image(history_entry: dict, base_url: str) -> bytes:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        raise QwenImageAPIError("ComfyUI history entry has no outputs")
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue
        for image_info in images:
            if not isinstance(image_info, dict):
                continue
            filename = image_info.get("filename")
            if not filename:
                continue
            params = {
                "filename": filename,
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
            response = requests.get(
                f"{base_url}/view",
                params=params,
                timeout=(
                    DEFAULT_CONNECT_TIMEOUT_SECONDS,
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                ),
            )
            if response.status_code == 200:
                return response.content
    raise QwenImageAPIError("ComfyUI history entry has no output image")


def generate_images_qwen(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Generate a single Qwen-Image-2.1 image (as a Ken-Burns video clip).

    This is the public entry point for the local Qwen-Image-2.1 provider,
    matching the calling convention of this codebase's other
    video-material-provider functions (search_videos_pexels,
    search_videos_pixabay, search_videos_coverr,
    minimax_video.generate_videos_minimax) so it can be plugged directly
    into material.py's provider dispatch.

    On any failure (ComfyUI unreachable, or a generation error) this logs
    the failure and returns an empty list rather than raising, matching
    how the other provider functions degrade gracefully instead of
    crashing the whole video generation task.
    """
    aspect = VideoAspect(video_aspect)
    base_url = _get_base_url()
    steps = int(config.qwen_image.get("steps", DEFAULT_STEPS) or DEFAULT_STEPS)
    poll_interval = float(
        config.qwen_image.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
    )
    poll_timeout = float(
        config.qwen_image.get("poll_timeout_seconds", DEFAULT_POLL_TIMEOUT_SECONDS)
    )

    width, height = aspect.to_resolution()
    width = _round_to_multiple_of_32(width)
    height = _round_to_multiple_of_32(height)

    expanded_prompt = llm.generate_image_prompt(search_term)
    seed = random.randint(0, 2**31 - 1)

    logger.info(f"generating image with Qwen-Image-2.1: term={search_term!r}")

    try:
        graph = _build_plain_t2i_graph(
            prompt=expanded_prompt, width=width, height=height, steps=steps, seed=seed
        )
        entry = _submit_and_wait(graph, base_url, poll_interval, poll_timeout)
        image_bytes = _fetch_output_image(entry, base_url)
    except QwenImageError as exc:
        logger.error(
            f"Qwen-Image generation failed: term={search_term!r}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return []
    except requests.RequestException as exc:
        logger.error(
            f"could not reach ComfyUI at {base_url}: term={search_term!r}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return []

    cache_dir = utils.storage_dir("qwen_image_generated", create=True)
    image_path = os.path.join(cache_dir, f"qwen-{uuid.uuid4().hex}.png")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    try:
        video_path = video.render_image_as_zoom_clip(
            image_path, duration=minimum_duration
        )
    except Exception as exc:
        logger.error(
            f"failed to render Qwen-Image output as a video clip: "
            f"term={search_term!r}, error={type(exc).__name__}, detail={exc}"
        )
        return []

    item = MaterialInfo()
    item.provider = "qwen_image"
    item.url = video_path
    item.duration = minimum_duration
    item.source_info = {
        "provider": "qwen_image",
        "search_term": search_term,
        "asset_id": os.path.basename(image_path),
        "rendition": {"id": "qwen-image-2.1", "width": width, "height": height},
    }
    return [item]
