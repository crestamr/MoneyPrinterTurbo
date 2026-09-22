"""Model lifecycle for the OmniVoice wrapper service.

The model is loaded lazily on first use and kept warm, because loading costs
tens of seconds. It is unloaded on demand or after an idle timeout so the GPU
can be handed back to other local workloads (this machine also runs ComfyUI).
``torch`` and ``omnivoice`` are imported inside ``_load_model`` so the module —
and its tests — stay importable without a GPU or model weights.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


def _load_model(model_id: str, device: str) -> Any:
    """Load the OmniVoice model. Imports live here so tests can patch this out."""
    import torch
    from omnivoice import OmniVoice

    return OmniVoice.from_pretrained(
        model_id, device_map=device, dtype=torch.float16
    )


def _release_vram() -> None:
    """Best-effort VRAM release; never fails the caller."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Engine:
    """Owns the model instance and decides when it lives and dies."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        idle_unload_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._model_id = model_id
        self._device = device
        self._idle_unload_seconds = float(idle_unload_seconds or 0)
        self._clock = clock
        self._model: Any = None
        self._last_used = clock()
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Any:
        """Return the loaded model, loading it on first access."""
        with self._lock:
            if self._model is None:
                self._model = _load_model(self._model_id, self._device)
            self._last_used = self._clock()
            return self._model

    def touch(self) -> None:
        """Mark the model as used now, so a long synthesis is not swept away.

        ``model`` stamps the timestamp when the model is *acquired*; a synthesis
        that runs longer than the idle timeout would otherwise be swept out from
        under itself. Callers touch again once the work is done.
        """
        with self._lock:
            self._last_used = self._clock()

    def unload(self) -> bool:
        """Drop the model. Returns whether anything was actually unloaded."""
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            _release_vram()
            return True

    def sweep_idle(self) -> bool:
        """Unload if idle longer than the configured timeout (0 disables)."""
        with self._lock:
            if self._model is None or self._idle_unload_seconds <= 0:
                return False
            if self._clock() - self._last_used < self._idle_unload_seconds:
                return False
            return self.unload()
