import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.omnivoice_server import voices


class TestVoiceDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, name: str, data: bytes = b"x"):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_discovers_supported_audio_files_by_stem(self):
        self._write("narrator-deep.wav")
        self._write("calm-female.mp3")
        self._write("host.flac")

        found = voices.discover_voices(self.tmp)

        self.assertEqual(
            sorted(found.keys()), ["calm-female", "host", "narrator-deep"]
        )

    def test_ignores_unsupported_and_sidecar_files(self):
        self._write("narrator.wav")
        self._write("narrator.txt")
        self._write("narrator.pt")
        self._write("notes.md")

        found = voices.discover_voices(self.tmp)

        self.assertEqual(list(found.keys()), ["narrator"])

    def test_returns_empty_for_missing_directory(self):
        self.assertEqual(voices.discover_voices(os.path.join(self.tmp, "nope")), {})

    def test_reads_sidecar_reference_text_when_present(self):
        self._write("narrator.wav")
        with open(os.path.join(self.tmp, "narrator.txt"), "w", encoding="utf-8") as f:
            f.write("  This is the reference transcript.  ")

        found = voices.discover_voices(self.tmp)

        self.assertEqual(
            found["narrator"].ref_text, "This is the reference transcript."
        )

    def test_reference_text_is_none_without_sidecar(self):
        self._write("narrator.wav")
        self.assertIsNone(voices.discover_voices(self.tmp)["narrator"].ref_text)

    def test_cache_key_changes_when_source_file_changes(self):
        path = self._write("narrator.wav", b"aaa")
        first = voices.discover_voices(self.tmp)["narrator"].cache_key
        with open(path, "wb") as f:
            f.write(b"bbbbbbbb")
        second = voices.discover_voices(self.tmp)["narrator"].cache_key
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
