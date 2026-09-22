"""Entry point: ``python -m services.omnivoice_server``."""

from __future__ import annotations

import argparse
import os
import threading
import time

from services.omnivoice_server.app import Settings, create_app
from services.omnivoice_server.engine import Engine

DEFAULT_PORT = 8890
SWEEP_INTERVAL_SECONDS = 30


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
    parser = argparse.ArgumentParser(description="Local OmniVoice TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--voices-dir",
        default=os.path.join("storage", "voices"),
        help="directory of reference clips; each file becomes a voice",
    )
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--idle-unload-seconds", type=float, default=600)
    parser.add_argument("--api-key", default=os.environ.get("OMNIVOICE_API_KEY", ""))
    args = parser.parse_args()

    # Imported after argparse so ``--help`` works without the server extras.
    import uvicorn

    os.makedirs(args.voices_dir, exist_ok=True)
    settings = Settings(
        voices_dir=os.path.abspath(args.voices_dir),
        model_id=args.model_id,
        device=args.device,
        num_step=args.num_step,
        idle_unload_seconds=args.idle_unload_seconds,
        api_key=args.api_key,
    )
    engine = Engine(
        model_id=settings.model_id,
        device=settings.device,
        idle_unload_seconds=settings.idle_unload_seconds,
    )
    if settings.idle_unload_seconds > 0:
        _start_idle_sweeper(engine)

    print(f"OmniVoice service on http://{args.host}:{args.port}")
    print(f"reference clips: {settings.voices_dir}")
    uvicorn.run(create_app(settings=settings, engine=engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
