"""MiniMax H3 hosted video generation (https://platform.minimax.io).

This module deliberately stays independent of ``app.services.material`` (no
import in either direction) so ``material.py`` can import this module for
its provider dispatch without a circular import.
"""

from __future__ import annotations

import os
import tempfile
import time
from urllib.parse import urlparse

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import minimax_media
from app.utils import utils

MINIMAX_VIDEO_GLOBAL_BASE_URL = "https://api.minimax.io"
MINIMAX_VIDEO_CN_BASE_URL = "https://api.minimaxi.com"

DEFAULT_MODEL = "MiniMax-H3"
DEFAULT_RESOLUTION = "768P"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 300.0
# Reference-conditioned clips are much slower: with 9 references one run
# took 3:45-5:00 per scene, and the 300 s default abandoned a clip that
# MiniMax was still rendering (and billing). Waiting costs nothing.
REFERENCE_POLL_TIMEOUT_SECONDS = 900.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0

# Bound how much of a non-200 response body we fold into the error message,
# mirroring elevenlabs_music.py's _safe_response_error truncation.
MAX_ERROR_BODY_CHARS = 200

# (min_duration, max_duration) seconds, per the platform.minimax.io API docs.
MODEL_DURATION_RANGES = {
    "MiniMax-H3": (4, 15),
    "MiniMax-H3-Max": (5, 15),
}


class MiniMaxVideoError(RuntimeError):
    """Base error for the MiniMax H3 video integration."""


class MiniMaxVideoConfigurationError(MiniMaxVideoError):
    """Raised when the integration is used without a resolvable API key."""


class MiniMaxVideoAPIError(MiniMaxVideoError):
    """Raised when the MiniMax API rejects a request or returns invalid JSON."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _resolve_minimax_video_base_url(configured_url: str) -> str:
    configured_url = (configured_url or "").strip().rstrip("/")
    if not configured_url:
        return MINIMAX_VIDEO_GLOBAL_BASE_URL
    return configured_url


def _infer_minimax_video_base_url(base_url: str) -> str:
    """根据 MiniMax LLM 地址推断同区域的视频生成地址，无法识别时返回空值。"""
    normalized_url = str(base_url or "").strip()
    if not normalized_url:
        return ""

    parse_target = normalized_url if "://" in normalized_url else f"//{normalized_url}"
    host = (urlparse(parse_target).hostname or "").lower()
    if host == "minimaxi.com" or host.endswith(".minimaxi.com"):
        return MINIMAX_VIDEO_CN_BASE_URL
    if host == "minimax.io" or host.endswith(".minimax.io"):
        return MINIMAX_VIDEO_GLOBAL_BASE_URL
    return ""


def get_minimax_video_api_key() -> str:
    """返回 MiniMax 视频生成的有效密钥，专用配置优先于 LLM 共享配置。"""
    return str(
        config.minimax_video.get("api_key", "")
        or config.app.get("minimax_api_key", "")
        or os.getenv("MINIMAX_API_KEY", "")
        or ""
    ).strip()


def get_minimax_video_base_url() -> str:
    """
    返回与当前有效密钥匹配的 MiniMax 视频生成地址。

    独立配置视频 Key 时尊重用户选择的视频地址；复用 MiniMax LLM Key 时，
    优先跟随 LLM Base URL 的区域，避免中国站 Key 被发送到国际站而返回 401。
    """
    dedicated_key = str(config.minimax_video.get("api_key", "") or "").strip()
    if not dedicated_key:
        inferred_url = _infer_minimax_video_base_url(
            config.app.get("minimax_base_url", "")
        )
        if inferred_url:
            return inferred_url
    return _resolve_minimax_video_base_url(config.minimax_video.get("base_url", ""))


def _build_prompt(search_term: str) -> str:
    return (
        f"Cinematic stock-footage-style video clip: {search_term}. "
        "No text, no captions, no subtitles, no watermarks, no logos."
    )


def _aspect_to_ratio(video_aspect) -> str:
    return str(video_aspect.value)


def _clamp_duration(minimum_duration: int, *, model: str) -> int:
    low, high = MODEL_DURATION_RANGES.get(model, MODEL_DURATION_RANGES[DEFAULT_MODEL])
    return max(low, min(high, int(minimum_duration)))


def _raise_for_status(
    response: requests.Response, *, action: str, log_context: str = ""
) -> None:
    """Raise MiniMaxVideoAPIError for a non-200 response.

    Shared by _create_task, _poll_task, and _download_to_cache, which all
    need the same truncate-body -> log -> raise sequence and previously
    duplicated it verbatim. ``action`` names the operation in messages
    ("create", "query", "download"); ``log_context`` optionally prefixes the
    log line with extra identifying context (e.g. ``task_id=...``).
    """
    if response.status_code == 200:
        return
    detail = ""
    try:
        detail = str(response.text or "")[:MAX_ERROR_BODY_CHARS]
    except Exception:
        detail = ""
    message = f"MiniMax video {action} returned HTTP {response.status_code}"
    if detail:
        message = f"{message}: {detail}"
    context_prefix = f"{log_context}, " if log_context else ""
    logger.error(
        f"MiniMax video {action} failed: "
        f"{context_prefix}status={response.status_code}, detail={detail or 'unavailable'}"
    )
    raise MiniMaxVideoAPIError(message, status_code=response.status_code)


def _parse_json_object(response: requests.Response, *, action: str) -> dict:
    """Parse a response body as a JSON object, raising MiniMaxVideoAPIError otherwise.

    Shared by _create_task and _poll_task, which both need the same
    parse-then-type-check sequence. ``action`` names the operation in
    messages ("create", "query").
    """
    try:
        body = response.json()
    except ValueError as exc:
        logger.error(f"MiniMax video {action} returned invalid JSON")
        raise MiniMaxVideoAPIError(
            f"MiniMax video {action} returned invalid JSON"
        ) from exc

    if not isinstance(body, dict):
        logger.error(
            f"MiniMax video {action} response is not a JSON object: "
            f"type={type(body).__name__}"
        )
        raise MiniMaxVideoAPIError(
            f"MiniMax video {action} response is not a JSON object"
        )
    return body


def _remove_temp_file(file_path: str) -> None:
    """Best-effort cleanup of a partial MiniMax download temp file.

    Never overrides the caller's original exception, matching the analogous
    _remove_file helper in elevenlabs_music.py.
    """
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            f"failed to remove MiniMax temporary download file: "
            f"path={file_path}, error={exc}"
        )


def _create_task(
    *,
    prompt: str,
    resolution: str,
    duration: int,
    ratio: str,
    model: str,
    api_key: str,
    base_url: str,
    reference_images: "list[str] | tuple[str, ...]" = (),
    first_frame: str | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Submit a MiniMax H3 video generation job and return its task_id.

    Raises MiniMaxVideoAPIError on any request failure, non-200 response,
    non-JSON body, non-dict JSON body, or a response missing task_id.
    """
    url = f"{base_url}/v2/video_generation"
    # minimax_media owns the payload rules: role sits beside image_url, the two
    # image modes are mutually exclusive, and any attached image forces
    # ratio="adaptive" because the output follows the image, not the request.
    content = minimax_media.build_content(
        prompt, first_frame=first_frame, reference_images=reference_images
    )
    payload = {
        "model": model,
        "content": content,
        "resolution": resolution,
        "duration": duration,
        "ratio": minimax_media.ratio_for(content, ratio),
    }
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, request_timeout),
        )
    except requests.RequestException as exc:
        logger.error(
            f"MiniMax video create request failed: error={type(exc).__name__}"
        )
        raise MiniMaxVideoAPIError(
            f"MiniMax video create request failed: {type(exc).__name__}"
        ) from exc

    _raise_for_status(response, action="create")
    body = _parse_json_object(response, action="create")

    raw_task_id = body.get("task_id")
    task_id = str(raw_task_id).strip() if raw_task_id else ""
    if not task_id:
        logger.error("MiniMax video create response is missing task_id")
        raise MiniMaxVideoAPIError("MiniMax video create response is missing task_id")
    return task_id


def _poll_task(
    *,
    task_id: str,
    api_key: str,
    base_url: str,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Poll a MiniMax video generation task until it succeeds, fails, or times out.

    Returns the presigned content URL on success. Raises MiniMaxVideoAPIError on
    any request failure, non-200 response, non-JSON body, non-dict JSON body, a
    response missing task status, or a failed/cancelled task. Raises
    MiniMaxVideoError if the task does not complete within poll_timeout_seconds.
    """
    url = f"{base_url}/v2/query/video_generation/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.monotonic() + poll_timeout_seconds

    while True:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, request_timeout),
            )
        except requests.RequestException as exc:
            # The generation is already paid for and still running server-side,
            # so a flaky poll must not abandon it. Keep polling until the
            # overall deadline; only give up if the network stays down.
            if time.monotonic() < deadline:
                logger.warning(
                    f"MiniMax video query request failed, retrying: "
                    f"task_id={task_id}, error={type(exc).__name__}"
                )
                time.sleep(poll_interval_seconds)
                continue
            logger.error(
                f"MiniMax video query request failed: error={type(exc).__name__}"
            )
            raise MiniMaxVideoAPIError(
                f"MiniMax video query request failed: {type(exc).__name__}"
            ) from exc

        _raise_for_status(response, action="query")
        body = _parse_json_object(response, action="query")

        task = body.get("task")
        if not isinstance(task, dict):
            logger.error("MiniMax video query response is missing task")
            raise MiniMaxVideoAPIError("MiniMax video query response is missing task")

        raw_status = task.get("status")
        if not raw_status:
            logger.error(
                f"MiniMax video query response is missing task status: task_id={task_id}"
            )
            raise MiniMaxVideoAPIError(
                "MiniMax video query response is missing task status"
            )
        status = str(raw_status).strip().lower()
        if status == "succeeded":
            content = task.get("content")
            content_url = (
                str(content.get("url", "")).strip()
                if isinstance(content, dict)
                else ""
            )
            if not content_url:
                logger.error(
                    "MiniMax video task succeeded but returned no content url: "
                    f"task_id={task_id}"
                )
                raise MiniMaxVideoAPIError(
                    "MiniMax video task succeeded but returned no content url"
                )
            return content_url

        if status in {"failed", "cancelled"}:
            error = task.get("error")
            detail = (
                str(error.get("message", "")).strip()
                if isinstance(error, dict)
                else ""
            ) or status
            logger.error(
                f"MiniMax video task {status}: task_id={task_id}, detail={detail}"
            )
            raise MiniMaxVideoAPIError(f"MiniMax video task {status}: {detail}")

        if time.monotonic() >= deadline:
            logger.error(
                "MiniMax video task did not complete within timeout: "
                f"task_id={task_id}, timeout={poll_timeout_seconds:g}"
            )
            raise MiniMaxVideoError(
                f"MiniMax video task {task_id} did not complete within "
                f"{poll_timeout_seconds:g} seconds"
            )

        time.sleep(poll_interval_seconds)


def _download_to_cache(content_url: str, task_id: str) -> str:
    """Download a MiniMax presigned content URL to a durable local cache file.

    MiniMax's ``content.url`` is a presigned link that expires in roughly 9
    hours. This app's material search cache persists entries for up to 24
    hours, so caching the presigned URL directly would let the cache outlive
    the URL. Downloading immediately to a stable local path sidesteps this:
    the local path -- not the presigned URL -- is what gets cached.
    """
    cache_dir = utils.storage_dir("cache_videos", create=True)
    destination = os.path.join(cache_dir, f"minimax-{task_id}.mp4")
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return destination

    try:
        response = requests.get(content_url, timeout=(10.0, 240.0))
    except requests.RequestException as exc:
        logger.error(
            f"MiniMax video download failed: task_id={task_id}, "
            f"error={type(exc).__name__}"
        )
        raise MiniMaxVideoAPIError(
            f"MiniMax video download failed: {type(exc).__name__}"
        ) from exc

    _raise_for_status(response, action="download", log_context=f"task_id={task_id}")

    # Write to a temp file in the same directory, then atomically publish via
    # os.replace. This avoids leaving a partial/truncated file at
    # `destination` if the process is killed mid-write, or if two concurrent
    # callers race on the same task_id -- either way, `destination` only
    # ever observes a complete file, matching the pattern used by
    # elevenlabs_music.py, bgm.py, and sonilo.py for downloaded/generated
    # media.
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".minimax-download-",
        suffix=".mp4",
        dir=cache_dir,
    )
    try:
        with os.fdopen(descriptor, "wb") as f:
            f.write(response.content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, destination)
        temp_path = ""
    finally:
        _remove_temp_file(temp_path)
    return destination


def _probe_dimensions(video_path: str) -> "tuple[int, int] | None":
    """Read the real frame size of a downloaded clip.

    With a reference image attached the API returns ``ratio="adaptive"`` and the
    output follows the *image*, so the requested VideoAspect is not necessarily
    what came back. Recording the request's aspect would quietly mislabel the
    material, so measure the file instead and fall back only if that fails.
    """
    try:
        from moviepy import VideoFileClip

        with VideoFileClip(video_path) as clip:
            width, height = clip.size
        return int(width), int(height)
    except Exception as exc:  # noqa: BLE001 - probing is best-effort
        logger.warning(
            f"could not probe MiniMax clip dimensions, falling back to the "
            f"requested aspect: path={video_path}, error={type(exc).__name__}: {exc}"
        )
        return None


def _generate_clip(
    *,
    prompt: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
    reference_images: "list[str] | tuple[str, ...]" = (),
    search_term: str = "",
) -> list[MaterialInfo]:
    """Create one MiniMax clip and return it as a MaterialInfo, or [] on failure.

    Shared by the stock-footage and product entry points. Like every other
    provider in material.py this degrades to an empty list rather than raising,
    so one bad clip cannot abort a whole video task.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_minimax_video_api_key()
    if not api_key:
        logger.error(
            "MiniMax video generation is not configured: set [minimax_video].api_key "
            "or app.minimax_api_key in config.toml"
        )
        return []

    base_url = get_minimax_video_base_url()
    model = str(config.minimax_video.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)
    resolution = str(
        config.minimax_video.get("resolution", DEFAULT_RESOLUTION)
        or DEFAULT_RESOLUTION
    )
    poll_interval_seconds = float(
        config.minimax_video.get(
            "poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS
        )
    )
    poll_timeout_seconds = float(
        config.minimax_video.get("poll_timeout_seconds", DEFAULT_POLL_TIMEOUT_SECONDS)
    )
    duration = _clamp_duration(minimum_duration, model=model)
    ratio = _aspect_to_ratio(aspect)
    reference_images = list(reference_images or [])
    if reference_images:
        poll_timeout_seconds = max(poll_timeout_seconds, REFERENCE_POLL_TIMEOUT_SECONDS)

    if reference_images:
        logger.info(
            f"generating MiniMax {model} clip with {len(reference_images)} "
            f"reference image(s)"
        )
    else:
        logger.info(f"generating video with MiniMax {model}: term={search_term!r}")

    try:
        task_id = _create_task(
            prompt=prompt,
            resolution=resolution,
            duration=duration,
            ratio=ratio,
            model=model,
            api_key=api_key,
            base_url=base_url,
            reference_images=reference_images,
        )
        content_url = _poll_task(
            task_id=task_id,
            api_key=api_key,
            base_url=base_url,
            poll_interval_seconds=poll_interval_seconds,
            poll_timeout_seconds=poll_timeout_seconds,
        )
        local_path = _download_to_cache(content_url, task_id)
    except (MiniMaxVideoError, minimax_media.MiniMaxMediaError) as exc:
        logger.error(
            f"MiniMax video generation failed: term={search_term!r}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return []

    width, height = aspect.to_resolution()
    if reference_images:
        probed = _probe_dimensions(local_path)
        if probed:
            width, height = probed

    item = MaterialInfo()
    item.provider = "minimax"
    item.url = local_path
    item.duration = duration
    item.source_info = {
        "provider": "minimax",
        "search_term": search_term,
        "asset_id": task_id,
        "rendition": {"id": model, "width": width, "height": height},
    }
    if reference_images:
        item.source_info["reference_image_count"] = len(reference_images)
    return [item]


def generate_videos_minimax(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Generate a single MiniMax H3 video clip for search_term.

    This is the public entry point for the MiniMax H3 video-generation
    provider, matching the calling convention of this codebase's other
    video-material-provider functions (search_videos_pexels,
    search_videos_pixabay, search_videos_coverr in app/services/material.py)
    so it can be plugged directly into material.py's provider dispatch.

    On any failure (missing API key, or any MiniMaxVideoError raised while
    creating/polling/downloading the task) this logs the failure and returns
    an empty list rather than raising, matching how the other provider
    functions degrade gracefully instead of crashing the whole video
    generation task.
    """
    return _generate_clip(
        prompt=_build_prompt(search_term),
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
        search_term=search_term,
    )


def resolve_references(sources: "list[str] | tuple[str, ...]") -> list[str]:
    """Upload local reference images once and return their API handles.

    ``generate_product_clip`` resolves references itself, which is right for a
    one-off call but uploads everything again for every scene of a run. Calling
    this first yields ``mm_file://`` handles (valid 7 days) that later calls
    pass straight through. Raises instead of returning [] so the caller can
    stop *before* spending on generations with a broken reference set.
    """
    api_key = get_minimax_video_api_key()
    if not api_key:
        raise MiniMaxVideoConfigurationError(
            "MiniMax video generation is not configured: set [minimax_video].api_key "
            "or app.minimax_api_key in config.toml"
        )
    base_url = get_minimax_video_base_url()
    return [
        minimax_media.resolve_image_ref(str(source), api_key=api_key, base_url=base_url)
        for source in (sources or [])
    ]


def generate_product_clip(
    prompt: str,
    minimum_duration: int,
    reference_images: "list[str] | tuple[str, ...]",
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Generate one product clip conditioned on reference images.

    ``reference_images`` may be local paths, public URLs, or ``mm_file://``
    handles; local files are validated and uploaded first. Up to 9 are allowed,
    which is what lets a product shot and a presenter shot go in together.

    ``prompt`` is passed through verbatim - unlike the stock-footage path it is
    not wrapped in a generic template, because product prompts are authored
    per-scene and describe how the product should be handled on camera.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_minimax_video_api_key()
    if not api_key:
        logger.error(
            "MiniMax video generation is not configured: set [minimax_video].api_key "
            "or app.minimax_api_key in config.toml"
        )
        return []
    base_url = get_minimax_video_base_url()

    try:
        resolved = [
            minimax_media.resolve_image_ref(
                str(source), api_key=api_key, base_url=base_url
            )
            for source in (reference_images or [])
        ]
    except minimax_media.MiniMaxMediaError as exc:
        logger.error(f"MiniMax product clip reference image rejected: {exc}")
        return []

    if not resolved:
        logger.error("generate_product_clip requires at least one reference image")
        return []

    return _generate_clip(
        prompt=prompt,
        minimum_duration=minimum_duration,
        video_aspect=aspect,
        reference_images=resolved,
        search_term=prompt[:60],
    )
