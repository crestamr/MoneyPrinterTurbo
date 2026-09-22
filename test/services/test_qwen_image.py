import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoAspect
from app.services import qwen_image


class TestQwenImageSettings(unittest.TestCase):
    def setUp(self):
        self.original_qwen_image_config = dict(config.qwen_image)

    def tearDown(self):
        config.qwen_image.clear()
        config.qwen_image.update(self.original_qwen_image_config)

    def test_round_to_multiple_of_32(self):
        self.assertEqual(qwen_image._round_to_multiple_of_32(1080), 1088)
        self.assertEqual(qwen_image._round_to_multiple_of_32(1920), 1920)
        self.assertEqual(qwen_image._round_to_multiple_of_32(1), 32)

    def test_is_comfyui_reachable_true_on_200(self):
        with patch(
            "app.services.qwen_image.requests.get",
            return_value=SimpleNamespace(status_code=200),
        ):
            self.assertTrue(qwen_image.is_comfyui_reachable("http://127.0.0.1:8188"))

    def test_is_comfyui_reachable_false_on_connection_error(self):
        with patch(
            "app.services.qwen_image.requests.get",
            side_effect=qwen_image.requests.RequestException("refused"),
        ):
            self.assertFalse(qwen_image.is_comfyui_reachable("http://127.0.0.1:8188"))


class TestGenerateImagesQwen(unittest.TestCase):
    def setUp(self):
        self.original_qwen_image_config = dict(config.qwen_image)
        config.qwen_image["use_consistent_host"] = False

    def tearDown(self):
        config.qwen_image.clear()
        config.qwen_image.update(self.original_qwen_image_config)

    def test_generate_images_qwen_happy_path(self):
        config.qwen_image["poll_interval_seconds"] = 0.01
        config.qwen_image["poll_timeout_seconds"] = 5

        queue_response = SimpleNamespace(
            status_code=200, json=lambda: {"prompt_id": "prompt-abc"}
        )
        history_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prompt-abc": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "8": {
                            "images": [
                                {
                                    "filename": "qwen_image_mpt_00001_.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            },
        )
        view_response = SimpleNamespace(status_code=200, content=b"fake-png-bytes")

        with tempfile.TemporaryDirectory() as fake_storage, patch(
            "app.services.qwen_image.requests.post",
            return_value=queue_response,
        ), patch(
            "app.services.qwen_image.requests.get",
            side_effect=[history_response, view_response],
        ), patch(
            "app.services.qwen_image.llm.generate_image_prompt",
            return_value="a golden retriever running on a sunlit beach",
        ), patch(
            "app.services.qwen_image.utils.storage_dir",
            side_effect=lambda sub_dir, create=False: os.path.join(
                fake_storage, sub_dir
            ),
        ), patch(
            "app.services.qwen_image.video.render_image_as_zoom_clip",
            side_effect=lambda image_path, duration: f"{image_path}.mp4",
        ):
            results = qwen_image.generate_images_qwen(
                search_term="golden retriever beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "qwen_image")
        self.assertTrue(item.url.endswith(".png.mp4"))
        self.assertEqual(item.duration, 5)
        self.assertEqual(item.source_info["search_term"], "golden retriever beach")
        self.assertEqual(item.source_info["rendition"]["width"], 1088)
        self.assertEqual(item.source_info["rendition"]["height"], 1920)

    def test_generate_images_qwen_returns_empty_list_when_comfyui_unreachable(self):
        with patch(
            "app.services.qwen_image.requests.post",
            side_effect=qwen_image.requests.RequestException("connection refused"),
        ):
            results = qwen_image.generate_images_qwen(
                search_term="golden retriever beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, [])

    def test_generate_images_qwen_returns_empty_list_on_generation_error(self):
        queue_response = SimpleNamespace(
            status_code=200, json=lambda: {"prompt_id": "prompt-abc"}
        )
        error_history_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prompt-abc": {
                    "status": {"completed": False, "status_str": "error"},
                    "outputs": {},
                }
            },
        )

        with patch(
            "app.services.qwen_image.requests.post",
            return_value=queue_response,
        ), patch(
            "app.services.qwen_image.requests.get",
            return_value=error_history_response,
        ), patch(
            "app.services.qwen_image.llm.generate_image_prompt",
            return_value="a golden retriever running on a sunlit beach",
        ):
            results = qwen_image.generate_images_qwen(
                search_term="golden retriever beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
