"""Discovery of OmniVoice reference clips.

Voices are plain audio files in a directory; the filename stem is the voice ID.
An optional sidecar ``<stem>.txt`` supplies the reference transcript, which lets
OmniVoice skip Whisper auto-transcription when building a clone prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


@dataclass(frozen=True)
class Voice:
    """One reference clip and everything derived from it."""

    voice_id: str
    ref_audio: str
    ref_text: str | None
    cache_key: str

    @property
    def prompt_path(self) -> str:
        """Where the cached clone prompt for this voice lives."""
        return f"{os.path.splitext(self.ref_audio)[0]}.{self.cache_key}.pt"


def _cache_key(path: str) -> str:
    """Identify a reference file by size and mtime so edits invalidate caches."""
    stat = os.stat(path)
    return f"{stat.st_size}-{int(stat.st_mtime)}"


def _read_sidecar_text(audio_path: str) -> str | None:
    sidecar = f"{os.path.splitext(audio_path)[0]}.txt"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    return text or None


def discover_voices(voices_dir: str) -> dict[str, Voice]:
    """Return every usable reference clip in ``voices_dir``, keyed by voice ID."""
    if not voices_dir or not os.path.isdir(voices_dir):
        return {}

    discovered: dict[str, Voice] = {}
    for entry in sorted(os.listdir(voices_dir)):
        path = os.path.join(voices_dir, entry)
        if not os.path.isfile(path):
            continue
        stem, extension = os.path.splitext(entry)
        if extension.lower() not in SUPPORTED_EXTENSIONS or not stem:
            continue
        discovered[stem] = Voice(
            voice_id=stem,
            ref_audio=path,
            ref_text=_read_sidecar_text(path),
            cache_key=_cache_key(path),
        )
    return discovered
