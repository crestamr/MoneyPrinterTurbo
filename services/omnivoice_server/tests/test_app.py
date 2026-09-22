import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.omnivoice_server import app as app_module


class TestApp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "narrator.wav"), "wb") as handle:
            handle.write(b"ref")
        self.settings = app_module.Settings(
            voices_dir=self.tmp,
            model_id="k2-fsa/OmniVoice",
            device="cuda:0",
            num_step=32,
            idle_unload_seconds=600,
            api_key="",
        )
        self.engine = MagicMock()
        self.engine.is_loaded = False
        self.client = TestClient(
            app_module.create_app(settings=self.settings, engine=self.engine)
        )

    def test_health_reports_ready_and_voice_count(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["voices"], 1)
        self.assertFalse(body["model_loaded"])

    def test_voices_endpoint_lists_discovered_voice_ids(self):
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"voices": ["narrator"]})

    def test_speech_returns_audio_bytes(self):
        with patch.object(
            app_module, "synthesize", return_value=b"ID3-audio"
        ) as synth:
            response = self.client.post(
                "/v1/audio/speech",
                json={
                    "model": "omnivoice",
                    "input": "Hello there.",
                    "voice": "narrator",
                    "response_format": "mp3",
                    "speed": 1.25,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"ID3-audio")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(synth.call_args.kwargs["text"], "Hello there.")
        self.assertEqual(synth.call_args.kwargs["speed"], 1.25)
        self.assertEqual(synth.call_args.kwargs["num_step"], 32)

    def test_speech_accepts_the_omnivoice_prefixed_voice_id(self):
        with patch.object(app_module, "synthesize", return_value=b"a") as synth:
            response = self.client.post(
                "/v1/audio/speech",
                json={"model": "omnivoice", "input": "Hi.", "voice": "omnivoice:narrator"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(synth.call_args.kwargs["voice"].voice_id, "narrator")

    def test_speech_rejects_an_unknown_voice_with_the_available_list(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={"model": "omnivoice", "input": "Hi.", "voice": "missing"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("narrator", response.json()["detail"])

    def test_speech_touches_the_engine_even_when_synthesis_fails(self):
        with patch.object(
            app_module, "synthesize", side_effect=app_module.SynthesisError("boom")
        ):
            response = self.client.post(
                "/v1/audio/speech",
                json={"model": "omnivoice", "input": "Hi.", "voice": "narrator"},
            )
        self.assertEqual(response.status_code, 500)
        # The idle sweeper must not reclaim the model right after a long call.
        self.engine.touch.assert_called_once()

    def test_speech_reports_a_model_load_failure_with_a_usable_detail(self):
        engine = MagicMock()
        type(engine).model = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
        )
        client = TestClient(
            app_module.create_app(settings=self.settings, engine=engine)
        )

        response = client.post(
            "/v1/audio/speech",
            json={"model": "omnivoice", "input": "Hi.", "voice": "narrator"},
        )

        self.assertEqual(response.status_code, 500)
        # A bare "Internal Server Error" leaves the user nothing to act on; an
        # OOM here usually means ComfyUI is holding the GPU.
        self.assertIn("CUDA out of memory", response.json()["detail"])

    def test_speech_rejects_empty_input(self):
        response = self.client.post(
            "/v1/audio/speech",
            json={"model": "omnivoice", "input": "   ", "voice": "narrator"},
        )
        self.assertEqual(response.status_code, 400)

    def test_unload_reports_whether_a_model_was_released(self):
        self.engine.unload.return_value = True
        response = self.client.post("/unload")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["unloaded"])

    def test_requests_require_the_api_key_when_one_is_configured(self):
        settings = app_module.Settings(
            voices_dir=self.tmp, model_id="m", device="cpu",
            num_step=32, idle_unload_seconds=0, api_key="secret",
        )
        client = TestClient(
            app_module.create_app(settings=settings, engine=MagicMock())
        )

        denied = client.get("/v1/audio/voices")
        allowed = client.get(
            "/v1/audio/voices", headers={"Authorization": "Bearer secret"}
        )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
