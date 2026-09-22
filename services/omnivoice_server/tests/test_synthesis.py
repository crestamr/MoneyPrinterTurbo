import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.omnivoice_server import synthesis, voices


def _make_voice(tmp: str, stem: str = "narrator", ref_text=None) -> voices.Voice:
    audio_path = os.path.join(tmp, f"{stem}.wav")
    with open(audio_path, "wb") as handle:
        handle.write(b"reference-audio")
    return voices.Voice(
        voice_id=stem,
        ref_audio=audio_path,
        ref_text=ref_text,
        cache_key="1-2",
    )


class TestClonePrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.voice = _make_voice(self.tmp)

    def test_builds_and_saves_a_prompt_when_no_cache_exists(self):
        model = MagicMock()
        prompt = MagicMock()
        model.create_voice_clone_prompt.return_value = prompt

        synthesis.load_or_build_prompt(model, self.voice)

        model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=self.voice.ref_audio, ref_text=None
        )
        prompt.save.assert_called_once_with(self.voice.prompt_path)

    def test_passes_sidecar_reference_text_when_available(self):
        voice = _make_voice(self.tmp, stem="host", ref_text="A transcript.")
        model = MagicMock()

        synthesis.load_or_build_prompt(model, voice)

        model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=voice.ref_audio, ref_text="A transcript."
        )

    def test_reuses_the_cached_prompt_without_rebuilding(self):
        with open(self.voice.prompt_path, "wb") as handle:
            handle.write(b"cached")
        model = MagicMock()
        cached = MagicMock()

        with patch.object(
            synthesis, "_load_prompt_file", return_value=cached
        ) as load_mock:
            result = synthesis.load_or_build_prompt(model, self.voice)

        load_mock.assert_called_once_with(self.voice.prompt_path)
        model.create_voice_clone_prompt.assert_not_called()
        self.assertIs(result, cached)

    def test_rebuilds_when_the_cached_prompt_is_unreadable(self):
        with open(self.voice.prompt_path, "wb") as handle:
            handle.write(b"corrupt")
        model = MagicMock()

        with patch.object(
            synthesis, "_load_prompt_file", side_effect=ValueError("bad")
        ):
            synthesis.load_or_build_prompt(model, self.voice)

        model.create_voice_clone_prompt.assert_called_once()


class TestSynthesize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.voice = _make_voice(self.tmp)

    def test_returns_encoded_audio_bytes(self):
        model = MagicMock()
        model.generate.return_value = [np.zeros(2400, dtype=np.float32)]

        with patch.object(
            synthesis, "load_or_build_prompt", return_value=MagicMock()
        ), patch.object(synthesis, "_encode_mp3", return_value=b"ID3-mp3") as encode:
            data = synthesis.synthesize(
                model=model, voice=self.voice, text="Hello.", speed=1.0, num_step=32
            )

        self.assertEqual(data, b"ID3-mp3")
        self.assertEqual(encode.call_args.args[1], synthesis.SAMPLE_RATE)

    def test_passes_speed_and_num_step_through_to_the_model(self):
        model = MagicMock()
        model.generate.return_value = [np.zeros(10, dtype=np.float32)]

        with patch.object(
            synthesis, "load_or_build_prompt", return_value=MagicMock()
        ), patch.object(synthesis, "_encode_mp3", return_value=b"x"):
            synthesis.synthesize(
                model=model, voice=self.voice, text="Hi.", speed=1.5, num_step=16
            )

        self.assertEqual(model.generate.call_args.kwargs["speed"], 1.5)
        self.assertEqual(model.generate.call_args.kwargs["num_step"], 16)

    def test_concatenates_multiple_returned_chunks(self):
        model = MagicMock()
        model.generate.return_value = [
            np.ones(100, dtype=np.float32),
            np.ones(50, dtype=np.float32),
        ]
        captured = {}

        def fake_encode(samples, rate):
            captured["length"] = len(samples)
            return b"x"

        with patch.object(
            synthesis, "load_or_build_prompt", return_value=MagicMock()
        ), patch.object(synthesis, "_encode_mp3", side_effect=fake_encode):
            synthesis.synthesize(
                model=model, voice=self.voice, text="Hi.", speed=1.0, num_step=32
            )

        self.assertEqual(captured["length"], 150)

    def test_raises_when_the_model_returns_no_audio(self):
        model = MagicMock()
        model.generate.return_value = []

        with patch.object(
            synthesis, "load_or_build_prompt", return_value=MagicMock()
        ):
            with self.assertRaises(synthesis.SynthesisError):
                synthesis.synthesize(
                    model=model, voice=self.voice, text="Hi.", speed=1.0, num_step=32
                )


if __name__ == "__main__":
    unittest.main()
