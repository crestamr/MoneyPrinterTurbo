"""MiniMax H3 hosted video generation (https://platform.minimax.io).

This module deliberately stays independent of ``app.services.material`` (no
import in either direction) so ``material.py`` can import this module for
its provider dispatch without a circular import.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

from app.config import config

MINIMAX_VIDEO_GLOBAL_BASE_URL = "https://api.minimax.io"
MINIMAX_VIDEO_CN_BASE_URL = "https://api.minimaxi.com"

DEFAULT_MODEL = "MiniMax-H3"
DEFAULT_RESOLUTION = "768P"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

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


def _create_task(
    *,
    prompt: str,
    resolution: str,
    duration: int,
    ratio: str,
    model: str,
    api_key: str,
    base_url: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> str:
    url = f"{base_url}/v2/video_generation"
    payload = {
        "model": model,
        "content": [{"type": "text", "text": prompt}],
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
    }
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(5.0, request_timeout),
        )
    except requests.RequestException as exc:
        raise MiniMaxVideoAPIError(
            f"MiniMax video create request failed: {type(exc).__name__}"
        ) from exc

    if response.status_code != 200:
        raise MiniMaxVideoAPIError(
            f"MiniMax video create returned HTTP {response.status_code}",
            status_code=response.status_code,
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise MiniMaxVideoAPIError(
            "MiniMax video create returned invalid JSON"
        ) from exc

    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        raise MiniMaxVideoAPIError("MiniMax video create response is missing task_id")
    return task_id
