"""OpenAI-compatible HTTP surface for local OmniVoice synthesis.

MoneyPrinterTurbo already speaks the OpenAI ``/v1/audio/speech`` contract for
its self-hosted providers (Kokoro, Chatterbox), so matching that contract means
the client side is a thin config wrapper rather than a new transport.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from services.omnivoice_server.engine import Engine
from services.omnivoice_server.synthesis import SynthesisError, synthesize
from services.omnivoice_server.voices import discover_voices


@dataclass(frozen=True)
class Settings:
    voices_dir: str
    model_id: str
    device: str
    num_step: int
    idle_unload_seconds: float
    api_key: str


class SpeechRequest(BaseModel):
    """The subset of the OpenAI speech payload this service honours."""

    model: str = "omnivoice"
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


def create_app(*, settings: Settings, engine: Engine) -> FastAPI:
    app = FastAPI(title="OmniVoice TTS", docs_url="/docs")

    def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        header = request.headers.get("authorization", "")
        presented = header.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(presented, settings.api_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "voices": len(discover_voices(settings.voices_dir)),
            "model_loaded": bool(engine.is_loaded),
        }

    @app.get("/v1/audio/voices", dependencies=[Depends(require_api_key)])
    def list_voices() -> dict:
        return {"voices": sorted(discover_voices(settings.voices_dir))}

    @app.post("/unload", dependencies=[Depends(require_api_key)])
    def unload() -> dict:
        return {"unloaded": bool(engine.unload())}

    @app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
    def speech(request: SpeechRequest) -> Response:
        text = (request.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="input is empty")

        # The client sends the WebUI's prefixed ID; the prefix is not part of
        # the filename, so strip it before looking the voice up.
        voice_id = request.voice.removeprefix("omnivoice:").strip()
        available = discover_voices(settings.voices_dir)
        voice = available.get(voice_id)
        if voice is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown voice {voice_id!r}; available: "
                    f"{', '.join(sorted(available)) or 'none'} "
                    f"(add reference clips to {settings.voices_dir})"
                ),
            )

        try:
            audio = synthesize(
                model=engine.model,
                voice=voice,
                text=text,
                speed=request.speed,
                num_step=settings.num_step,
            )
        except SynthesisError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except Exception as error:
            # Loading the model can fail for reasons synthesis never sees —
            # most likely CUDA OOM when another local workload (ComfyUI) already
            # holds the GPU, or missing weights. Without this the client gets a
            # bare "Internal Server Error" and the user has nothing to act on.
            raise HTTPException(
                status_code=500,
                detail=f"OmniVoice model unavailable: {type(error).__name__}: {error}",
            ) from error
        finally:
            # Stamp the model as freshly used: a synthesis longer than the idle
            # timeout must not be swept away while its own request is running.
            engine.touch()

        return Response(content=audio, media_type="audio/mpeg")

    return app
