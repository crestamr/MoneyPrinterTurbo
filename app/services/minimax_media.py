"""Image references for MiniMax v2 video generation.

Split out of ``minimax_video`` so the payload rules live in one small,
directly testable place. Everything here follows the v2 contract published at
platform.minimax.io; the parts that matter and are easy to get wrong:

* ``role`` is a **sibling** of ``image_url``, not a key inside it::

      {"type": "image_url", "image_url": {"url": "..."}, "role": "reference_image"}

* ``first_frame``/``last_frame`` and ``reference_image`` are **mutually
  exclusive** within one request - sending both is a 400.
* ``ratio`` must be ``adaptive`` whenever any image is attached (the output
  follows the image), and must *never* be ``adaptive`` for text-only requests.
* A request needs exactly one non-empty ``text`` item.

Images are validated locally before upload on purpose: the docs publish the
size/dimension/aspect limits but no error code for breaching them, so a remote
rejection would only be distinguishable by prose. Refusing early also avoids
paying for a round trip with a file that was never going to be accepted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from loguru import logger

# Verified limits (platform.minimax.io, v2 video generation).
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
)
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MIN_IMAGE_DIMENSION = 256
MAX_IMAGE_DIMENSION = 5760
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5
MAX_REFERENCE_IMAGES = 9

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
# Where to grab a still from, as a fraction of duration. The opening frames are
# often black or mid-fade and the closing ones often hold an end card, so take
# something from the body of the clip.
FRAME_POSITION = 0.35

ROLE_FIRST_FRAME = "first_frame"
ROLE_LAST_FRAME = "last_frame"
ROLE_REFERENCE_IMAGE = "reference_image"

FILES_UPLOAD_PATH = "/v1/files/upload"
UPLOAD_PURPOSE = "video_generation_input"
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 120.0

# Text-to-video rejects "adaptive", so a caller asking for it needs a concrete
# fallback. Portrait matches this project's short-form output.
DEFAULT_TEXT_RATIO = "9:16"
ADAPTIVE_RATIO = "adaptive"

MAX_ERROR_BODY_CHARS = 300


class MiniMaxMediaError(RuntimeError):
    """Raised when an image cannot be validated, uploaded, or attached."""


@dataclass(frozen=True)
class ImageSpec:
    """What local validation learned about an image."""

    path: str
    width: int
    height: int
    size_bytes: int


def validate_image(path: str) -> ImageSpec:
    """Check one local image against the documented limits.

    Raises MiniMaxMediaError with a message naming the specific limit, so the
    caller can show the user something actionable rather than "invalid image".
    """
    if not path or not os.path.isfile(path):
        raise MiniMaxMediaError(f"image not found: {path!r}")

    extension = os.path.splitext(path)[1].lower()
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_IMAGE_EXTENSIONS))
        raise MiniMaxMediaError(
            f"unsupported image format {extension or '(none)'!r} for {path!r}; "
            f"supported formats: {supported}"
        )

    size_bytes = os.path.getsize(path)
    if size_bytes > MAX_IMAGE_BYTES:
        raise MiniMaxMediaError(
            f"image is {size_bytes / 1024 / 1024:.1f} MB, above the 30 MB limit: {path!r}"
        )

    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except MiniMaxMediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - unreadable file, whatever the cause
        raise MiniMaxMediaError(f"could not read image {path!r}: {exc}") from exc

    smallest = min(width, height)
    largest = max(width, height)
    if smallest < MIN_IMAGE_DIMENSION:
        raise MiniMaxMediaError(
            f"image is {width}x{height}; both sides must be at least "
            f"{MIN_IMAGE_DIMENSION}px: {path!r}"
        )
    if largest > MAX_IMAGE_DIMENSION:
        raise MiniMaxMediaError(
            f"image is {width}x{height}; neither side may exceed "
            f"{MAX_IMAGE_DIMENSION}px: {path!r}"
        )

    aspect = width / height if height else 0.0
    if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
        raise MiniMaxMediaError(
            f"image aspect ratio {aspect:.2f} ({width}x{height}) is outside the "
            f"supported {MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO} range: {path!r}"
        )

    return ImageSpec(path=path, width=width, height=height, size_bytes=size_bytes)


def is_video(path: str) -> bool:
    return os.path.splitext(path or "")[1].lower() in SUPPORTED_VIDEO_EXTENSIONS


def extract_frame(video_path: str, *, out_path: str = "", position: float = FRAME_POSITION) -> str:
    """Grab one still from a video for use as a reference image.

    A character reference is often a previous clip rather than a photo, and
    reference_video has tighter limits (15s per clip) than most footage
    satisfies - while reference_image is the shape the API documents for
    combining a person with a product. Pulling a frame keeps us on that path.
    """
    if not video_path or not os.path.isfile(video_path):
        raise MiniMaxMediaError(f"video not found: {video_path!r}")

    from app.utils import utils

    duration = _video_duration(video_path)
    timestamp = max(0.0, duration * position) if duration else 0.0

    if not out_path:
        cache_dir = utils.storage_dir("product_images", create=True)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # Key on the position too, so two --character-at values can coexist
        # instead of silently overwriting one another.
        key = utils.md5(f"{video_path}@{position:.4f}")[:10]
        out_path = os.path.join(cache_dir, f"frame-{key}-{stem}.png")

    command = [
        utils.get_ffmpeg_binary(),
        "-y", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        out_path,
    ]
    import subprocess

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not os.path.isfile(out_path):
        raise MiniMaxMediaError(
            f"could not extract a frame from {video_path!r}: "
            f"{(completed.stderr or '').strip()[:200]}"
        )
    logger.info(f"extracted character reference frame at {timestamp:.1f}s -> {out_path}")
    return out_path


def _video_duration(video_path: str) -> float:
    from app.utils import utils
    import subprocess

    try:
        completed = subprocess.run(
            [
                utils.get_ffmpeg_binary().replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return float((completed.stdout or "0").strip() or 0.0)
    except (ValueError, OSError):
        return 0.0


def _error_message(response: requests.Response) -> str:
    """Pull the provider's own message out of the nested v2 error envelope.

    No error code is documented for a rejected image, so the remote prose is
    the only trustworthy signal and must survive to the caller intact.
    """
    try:
        body = response.json()
    except ValueError:
        return str(getattr(response, "text", "") or "")[:MAX_ERROR_BODY_CHARS]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:MAX_ERROR_BODY_CHARS]
        base = body.get("base_resp")
        if isinstance(base, dict) and base.get("status_msg"):
            return str(base["status_msg"])[:MAX_ERROR_BODY_CHARS]
    return str(getattr(response, "text", "") or "")[:MAX_ERROR_BODY_CHARS]


def upload_image(
    path: str,
    *,
    api_key: str,
    base_url: str,
    timeout: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
) -> str:
    """Upload one image to the Files API and return its ``mm_file://`` reference.

    Preferred over inlining base64: the request body caps at 64 MB and base64
    inflates by roughly a third, so several inline images exhaust the body
    budget long before any single image reaches its own 30 MB limit.
    Uploads remain valid for 7 days.
    """
    url = f"{base_url.rstrip('/')}{FILES_UPLOAD_PATH}"
    try:
        with open(path, "rb") as handle:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"purpose": UPLOAD_PURPOSE},
                files={"file": (os.path.basename(path), handle)},
                timeout=timeout,
            )
    except OSError as exc:
        raise MiniMaxMediaError(f"could not open image {path!r}: {exc}") from exc
    except requests.RequestException as exc:
        raise MiniMaxMediaError(
            f"MiniMax file upload request failed: {type(exc).__name__}"
        ) from exc

    if getattr(response, "status_code", None) != 200:
        detail = _error_message(response)
        logger.error(
            f"MiniMax file upload failed: status={response.status_code}, "
            f"detail={detail or 'unavailable'}"
        )
        raise MiniMaxMediaError(
            f"MiniMax file upload returned HTTP {response.status_code}: {detail}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise MiniMaxMediaError("MiniMax file upload returned invalid JSON") from exc

    file_id = None
    if isinstance(body, dict) and isinstance(body.get("file"), dict):
        file_id = body["file"].get("file_id")
    if file_id in (None, ""):
        raise MiniMaxMediaError(
            f"MiniMax file upload response had no file_id: {_error_message(response)}"
        )

    logger.info(f"uploaded reference image to MiniMax: file_id={file_id}")
    return f"mm_file://{file_id}"


def resolve_image_ref(source: str, *, api_key: str, base_url: str) -> str:
    """Turn a user-supplied image into something the content array accepts.

    Public URLs and existing ``mm_file://`` handles pass straight through; a
    local path is validated and uploaded.
    """
    source = (source or "").strip()
    if not source:
        raise MiniMaxMediaError("empty image reference")
    if source.startswith(("http://", "https://", "mm_file://", "data:")):
        return source
    if is_video(source):
        source = extract_frame(source)
    validate_image(source)
    return upload_image(source, api_key=api_key, base_url=base_url)


def build_content(
    prompt: str,
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: "list[str] | tuple[str, ...]" = (),
) -> list[dict]:
    """Assemble the v2 ``content`` array, enforcing the documented rules."""
    text = (prompt or "").strip()
    if not text:
        raise MiniMaxMediaError("a non-empty text prompt is required")

    reference_images = list(reference_images or [])
    frames = [f for f in (first_frame, last_frame) if f]
    if frames and reference_images:
        raise MiniMaxMediaError(
            "first_frame/last_frame and reference_image are mutually exclusive "
            "in a single MiniMax request; send one mode or the other"
        )
    if len(reference_images) > MAX_REFERENCE_IMAGES:
        raise MiniMaxMediaError(
            f"at most {MAX_REFERENCE_IMAGES} reference images are allowed, "
            f"got {len(reference_images)}"
        )

    content: list[dict] = [{"type": "text", "text": text}]
    if first_frame:
        content.append(_image_item(first_frame, ROLE_FIRST_FRAME))
    if last_frame:
        content.append(_image_item(last_frame, ROLE_LAST_FRAME))
    for reference in reference_images:
        content.append(_image_item(reference, ROLE_REFERENCE_IMAGE))
    return content


def _image_item(url: str, role: str) -> dict:
    # role sits beside image_url, never inside it.
    return {"type": "image_url", "image_url": {"url": url}, "role": role}


def has_image(content: "list[dict]") -> bool:
    return any(item.get("type") == "image_url" for item in content or [])


def ratio_for(content: "list[dict]", requested_ratio: str) -> str:
    """Pick the ratio the API will actually accept for this content array."""
    if has_image(content):
        return ADAPTIVE_RATIO
    requested = (requested_ratio or "").strip()
    if not requested or requested == ADAPTIVE_RATIO:
        return DEFAULT_TEXT_RATIO
    return requested
