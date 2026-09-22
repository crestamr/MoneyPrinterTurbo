"""Entry point: ``python -m services.omnivoice_server``.

Settings resolve in this order, highest priority first:

1. an explicit command-line flag,
2. the ``OMNIVOICE_API_KEY`` environment variable (``--api-key`` only),
3. the ``[omnivoice]`` table of the repository's ``config.toml``,
4. the built-in default shown by ``--help``.

Only the keys that describe *this service* are taken from ``config.toml``:
``voices_dir``, ``num_step``, ``idle_unload_seconds`` and ``api_key``. The
``base_url`` and ``model_id`` that live in the same table belong to the client —
``model_id`` there is the name the client puts in the request payload
(``omnivoice``), not the Hugging Face repo this service loads weights from.

``config.toml`` is read directly with :mod:`tomllib`. This service deliberately
imports nothing from ``app/`` so it can keep living in its own environment.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.omnivoice_server.app import Settings, create_app
from services.omnivoice_server.engine import Engine

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8890
DEFAULT_VOICES_DIR = os.path.join("storage", "voices")
DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_STEP = 32
DEFAULT_IDLE_UNLOAD_SECONDS = 600.0

SWEEP_INTERVAL_SECONDS = 30
CONFIG_SECTION = "omnivoice"
API_KEY_ENV_VAR = "OMNIVOICE_API_KEY"

# The package sits at <repo>/services/omnivoice_server/, so the root is two
# levels above it.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"


@dataclass(frozen=True)
class Launch:
    """Everything ``main`` needs, resolved but not yet acted on."""

    host: str
    port: int
    settings: Settings


def _load_config_table(config_path: Path | str) -> dict[str, Any]:
    """Return the ``[omnivoice]`` table, or an empty one if it is unusable.

    A missing, unreadable or malformed ``config.toml`` must never stop the
    service from starting — it just means the built-in defaults apply.
    """

    try:
        raw = Path(config_path).read_bytes()
        # Decode explicitly as UTF-8 (tolerating a BOM) rather than letting the
        # platform's default codec — cp1252 on Windows — mangle the file.
        text = raw.decode("utf-8-sig")
        data = tomllib.loads(text)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        if not isinstance(error, FileNotFoundError):
            print(f"ignoring {config_path}: {type(error).__name__}: {error}")
        return {}

    table = data.get(CONFIG_SECTION)
    return table if isinstance(table, dict) else {}


def _as_str(value: Any) -> str | None:
    """A usable string, or ``None``. Empty means "not set", per config.toml."""

    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first(*candidates: Any) -> Any:
    """The first candidate that was actually set."""

    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _build_parser() -> argparse.ArgumentParser:
    # Every default is ``None`` so an explicit flag is distinguishable from an
    # omitted one; the real defaults are applied in ``resolve_settings``.
    parser = argparse.ArgumentParser(
        prog="python -m services.omnivoice_server",
        description="Local OmniVoice TTS service",
        epilog=(
            "voices_dir, num_step, idle_unload_seconds and api_key also read "
            "their defaults from the [omnivoice] table of config.toml; flags "
            "given here override it."
        ),
    )
    parser.add_argument(
        "--host", default=None, help=f"bind address (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=None, help=f"bind port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--voices-dir",
        default=None,
        help=(
            "directory of reference clips; each file becomes a voice "
            f"(default: {DEFAULT_VOICES_DIR})"
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"Hugging Face repo to load weights from (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--device", default=None, help=f"torch device (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help=f"diffusion steps (default: {DEFAULT_NUM_STEP})",
    )
    parser.add_argument(
        "--idle-unload-seconds",
        type=float,
        default=None,
        help=(
            "unload the model after this long idle, 0 disables "
            f"(default: {DEFAULT_IDLE_UNLOAD_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"bearer token; empty disables auth (default: ${API_KEY_ENV_VAR} or empty)",
    )
    return parser


def resolve_settings(
    argv: Sequence[str] | None = None,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> Launch:
    """Resolve CLI flags, ``config.toml`` and built-in defaults into a `Launch`.

    Pure apart from reading ``config_path``: it starts nothing and loads no
    model, so it is safe to call from tests.
    """

    args = _build_parser().parse_args(argv)
    config = _load_config_table(config_path)
    environ = os.environ if env is None else env

    settings = Settings(
        voices_dir=_first(
            args.voices_dir, _as_str(config.get("voices_dir")), DEFAULT_VOICES_DIR
        ),
        # Intentionally not from config.toml: [omnivoice] model_id is the
        # client's payload model name, not a weights repo.
        model_id=_first(args.model_id, DEFAULT_MODEL_ID),
        device=_first(args.device, DEFAULT_DEVICE),
        num_step=_first(args.num_step, _as_int(config.get("num_step")), DEFAULT_NUM_STEP),
        idle_unload_seconds=_first(
            args.idle_unload_seconds,
            _as_float(config.get("idle_unload_seconds")),
            DEFAULT_IDLE_UNLOAD_SECONDS,
        ),
        api_key=_first(
            args.api_key,
            _as_str(environ.get(API_KEY_ENV_VAR)),
            _as_str(config.get("api_key")),
            "",
        ),
    )
    return Launch(
        host=_first(args.host, DEFAULT_HOST),
        port=_first(args.port, DEFAULT_PORT),
        settings=settings,
    )


def _start_idle_sweeper(engine: Engine, interval: float = SWEEP_INTERVAL_SECONDS):
    """Periodically unload the idle model so the GPU returns to other work.

    Runs as a daemon thread started only by this entry point, so importing or
    testing ``create_app`` never spawns background threads.
    """

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                if engine.sweep_idle():
                    print("OmniVoice model unloaded after idle timeout")
            except Exception as error:
                print(f"idle sweep failed: {type(error).__name__}: {error}")

    thread = threading.Thread(target=loop, name="omnivoice-idle-sweeper", daemon=True)
    thread.start()
    return thread


def main() -> None:
    launch = resolve_settings()

    # Imported after argparse so ``--help`` works without the server extras.
    import uvicorn

    settings = replace(
        launch.settings, voices_dir=os.path.abspath(launch.settings.voices_dir)
    )
    os.makedirs(settings.voices_dir, exist_ok=True)
    engine = Engine(
        model_id=settings.model_id,
        device=settings.device,
        idle_unload_seconds=settings.idle_unload_seconds,
    )
    if settings.idle_unload_seconds > 0:
        _start_idle_sweeper(engine)

    print(f"OmniVoice service on http://{launch.host}:{launch.port}")
    print(f"reference clips: {settings.voices_dir}")
    uvicorn.run(
        create_app(settings=settings, engine=engine),
        host=launch.host,
        port=launch.port,
    )


if __name__ == "__main__":
    main()
