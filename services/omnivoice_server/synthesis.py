"""Turning text plus a reference clip into MP3 bytes.

Building a clone prompt is the expensive part of zero-shot cloning (it may run
Whisper over the reference clip), so prompts are cached on disk next to their
reference audio and keyed by the reference's size and mtime.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import tempfile
from typing import Any

import numpy as np

from services.omnivoice_server.voices import Voice

SAMPLE_RATE = 24000

# OmniVoice's text normalizer (spelling out numbers, currency, abbreviations)
# lives in WeTextProcessing, which is built on pynini. pynini ships no Windows
# wheel and its source build passes GCC-only flags that MSVC rejects, so on
# Windows these are simply absent.
TEXT_NORMALIZATION_MODULES = ("pynini", "tn")


def _find_spec(name: str) -> Any:
    """Locate a module without importing it. Separate so tests can patch it."""
    return importlib.util.find_spec(name)


@functools.lru_cache(maxsize=1)
def text_normalization_available() -> bool:
    """Whether the optional text-normalization dependencies are importable.

    Asking the model to normalize without them makes ``generate`` raise
    ImportError, failing every request; detecting up front degrades to
    un-normalized text instead, which is the only option on Windows.
    """
    for name in TEXT_NORMALIZATION_MODULES:
        try:
            if _find_spec(name) is None:
                return False
        except Exception:
            # A half-installed package makes find_spec raise rather than
            # return None. Either way it cannot be used.
            return False
    return True


class SynthesisError(RuntimeError):
    """Raised when the model cannot produce usable audio."""


def _load_prompt_file(path: str) -> Any:
    """Load a cached clone prompt. Separate so tests can patch it."""
    from omnivoice import VoiceClonePrompt

    return VoiceClonePrompt.load(path)


def load_or_build_prompt(model: Any, voice: Voice) -> Any:
    """Return a clone prompt for ``voice``, building and caching it if needed."""
    path = voice.prompt_path
    if os.path.isfile(path):
        try:
            return _load_prompt_file(path)
        except Exception:
            # A stale or truncated cache must never be fatal: rebuild instead.
            pass

    prompt = model.create_voice_clone_prompt(
        ref_audio=voice.ref_audio, ref_text=voice.ref_text
    )
    try:
        prompt.save(path)
    except Exception:
        # Caching is an optimisation; failing to persist it is not an error.
        pass
    return prompt


def _encode_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float samples to MP3 via ffmpeg, which this project already needs."""
    import soundfile as sf

    wav_path = ""
    mp3_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        sf.write(wav_path, samples, sample_rate)
        mp3_path = f"{wav_path}.mp3"
        result = subprocess.run(
            [
                os.environ.get("FFMPEG_BINARY", "ffmpeg"),
                "-hide_banner", "-loglevel", "error", "-y",
                "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k",
                mp3_path,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not os.path.exists(mp3_path):
            raise SynthesisError(
                f"ffmpeg failed to encode audio: {result.stderr[:200]!r}"
            )
        with open(mp3_path, "rb") as handle:
            return handle.read()
    finally:
        for path in (wav_path, mp3_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def synthesize(
    *,
    model: Any,
    voice: Voice,
    text: str,
    speed: float,
    num_step: int,
    normalize_text: bool | None = None,
) -> bytes:
    """Synthesize ``text`` in ``voice`` and return MP3 bytes.

    ``normalize_text`` defaults to whatever the environment can actually
    support; pass a bool to force the choice.
    """
    if normalize_text is None:
        normalize_text = text_normalization_available()

    prompt = load_or_build_prompt(model, voice)
    chunks = model.generate(
        text=text,
        voice_clone_prompt=prompt,
        num_step=int(num_step),
        speed=float(speed),
        normalize_text=bool(normalize_text),
    )
    if not chunks:
        raise SynthesisError("OmniVoice returned no audio")

    samples = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks])
    if samples.size == 0:
        raise SynthesisError("OmniVoice returned empty audio")
    return _encode_mp3(samples, SAMPLE_RATE)
