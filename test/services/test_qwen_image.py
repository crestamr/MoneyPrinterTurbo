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


def _fake_storage_dir(root):
    """Stand-in for utils.storage_dir that honors its create=True contract."""

    def storage_dir(sub_dir, create=False):
        path = os.path.join(root, sub_dir)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    return storage_dir


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
            side_effect=_fake_storage_dir(fake_storage),
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

    def test_generate_images_qwen_returns_empty_list_on_malformed_prompt_response(self):
        malformed_response = SimpleNamespace(status_code=200, json=lambda: {})

        with patch(
            "app.services.qwen_image.requests.post",
            return_value=malformed_response,
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

    def test_generate_images_qwen_returns_empty_list_on_malformed_outputs(self):
        queue_response = SimpleNamespace(
            status_code=200, json=lambda: {"prompt_id": "prompt-abc"}
        )
        history_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "prompt-abc": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"8": "done", "9": {"images": ["not-a-dict"]}},
                }
            },
        )

        with patch(
            "app.services.qwen_image.requests.post",
            return_value=queue_response,
        ), patch(
            "app.services.qwen_image.requests.get",
            return_value=history_response,
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


def _make_history_response(prompt_id, filename="qwen_image_mpt_00001_.png"):
    return SimpleNamespace(
        status_code=200,
        json=lambda: {
            prompt_id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "8": {
                        "images": [
                            {
                                "filename": filename,
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
        },
    )


class TestGenerateImagesQwenWithHost(unittest.TestCase):
    def setUp(self):
        self.original_qwen_image_config = dict(config.qwen_image)
        config.qwen_image["use_consistent_host"] = True
        config.qwen_image["poll_interval_seconds"] = 0.01
        config.qwen_image["poll_timeout_seconds"] = 5

    def tearDown(self):
        config.qwen_image.clear()
        config.qwen_image.update(self.original_qwen_image_config)

    def test_uses_uploaded_reference_photo_when_configured(self):
        with tempfile.TemporaryDirectory() as fake_storage:
            host_photo_path = os.path.join(fake_storage, "host_photo.png")
            with open(host_photo_path, "wb") as f:
                f.write(b"fake-host-photo-bytes")

            config.qwen_image["host_reference_image"] = host_photo_path

            upload_response = SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "name": "uploaded_host.png",
                    "subfolder": "",
                    "type": "input",
                },
            )
            scene_queue_response = SimpleNamespace(
                status_code=200, json=lambda: {"prompt_id": "scene-prompt"}
            )
            scene_history_response = _make_history_response("scene-prompt")
            scene_view_response = SimpleNamespace(
                status_code=200, content=b"fake-scene-bytes"
            )

            with patch(
                "app.services.qwen_image.requests.post",
                side_effect=[upload_response, scene_queue_response],
            ) as mock_post, patch(
                "app.services.qwen_image.requests.get",
                side_effect=[scene_history_response, scene_view_response],
            ), patch(
                "app.services.qwen_image.llm.generate_image_prompt",
                return_value="a golden retriever running on a sunlit beach",
            ), patch(
                "app.services.qwen_image.utils.storage_dir",
                side_effect=_fake_storage_dir(fake_storage),
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
            first_post_call = mock_post.call_args_list[0]
            self.assertIn("/upload/image", first_post_call.args[0])

    def test_auto_generates_and_caches_host_portrait_when_no_reference_set(self):
        config.qwen_image["host_reference_image"] = ""
        config.qwen_image["host_description"] = "a friendly presenter"

        portrait_queue_response = SimpleNamespace(
            status_code=200, json=lambda: {"prompt_id": "portrait-prompt"}
        )
        portrait_history_response = _make_history_response("portrait-prompt")
        portrait_view_response = SimpleNamespace(
            status_code=200, content=b"fake-portrait-bytes"
        )
        upload_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "name": "uploaded_host.png",
                "subfolder": "",
                "type": "input",
            },
        )
        scene_queue_response = SimpleNamespace(
            status_code=200, json=lambda: {"prompt_id": "scene-prompt"}
        )
        scene_history_response = _make_history_response("scene-prompt")
        scene_view_response = SimpleNamespace(
            status_code=200, content=b"fake-scene-bytes"
        )

        with tempfile.TemporaryDirectory() as fake_storage:
            with patch(
                "app.services.qwen_image.requests.post",
                side_effect=[
                    portrait_queue_response,
                    upload_response,
                    scene_queue_response,
                ],
            ) as mock_post, patch(
                "app.services.qwen_image.requests.get",
                side_effect=[
                    portrait_history_response,
                    portrait_view_response,
                    scene_history_response,
                    scene_view_response,
                ],
            ), patch(
                "app.services.qwen_image.llm.generate_image_prompt",
                return_value="a golden retriever running on a sunlit beach",
            ), patch(
                "app.services.qwen_image.utils.storage_dir",
                side_effect=_fake_storage_dir(fake_storage),
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
            self.assertEqual(mock_post.call_count, 3)
            self.assertIn("/prompt", mock_post.call_args_list[0].args[0])
            self.assertIn("/upload/image", mock_post.call_args_list[1].args[0])
            self.assertIn("/prompt", mock_post.call_args_list[2].args[0])

            cache_dir = os.path.join(fake_storage, "qwen_image_host_cache")
            cached_files = os.listdir(cache_dir)
            self.assertEqual(len(cached_files), 1)
            with open(os.path.join(cache_dir, cached_files[0]), "rb") as f:
                self.assertEqual(f.read(), b"fake-portrait-bytes")

    def test_reuses_cached_host_portrait_without_regenerating(self):
        config.qwen_image["host_reference_image"] = ""
        config.qwen_image["host_description"] = "a friendly presenter"

        with tempfile.TemporaryDirectory() as fake_storage:
            cache_dir = os.path.join(fake_storage, "qwen_image_host_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir, f"{qwen_image.utils.md5('a friendly presenter')}.png"
            )
            with open(cache_path, "wb") as f:
                f.write(b"already-cached-portrait-bytes")

            upload_response = SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "name": "uploaded_host.png",
                    "subfolder": "",
                    "type": "input",
                },
            )
            scene_queue_response = SimpleNamespace(
                status_code=200, json=lambda: {"prompt_id": "scene-prompt"}
            )
            scene_history_response = _make_history_response("scene-prompt")
            scene_view_response = SimpleNamespace(
                status_code=200, content=b"fake-scene-bytes"
            )

            with patch(
                "app.services.qwen_image.requests.post",
                side_effect=[upload_response, scene_queue_response],
            ) as mock_post, patch(
                "app.services.qwen_image.requests.get",
                side_effect=[scene_history_response, scene_view_response],
            ), patch(
                "app.services.qwen_image.llm.generate_image_prompt",
                return_value="a golden retriever running on a sunlit beach",
            ), patch(
                "app.services.qwen_image.utils.storage_dir",
                side_effect=_fake_storage_dir(fake_storage),
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
            self.assertEqual(mock_post.call_count, 2)
            self.assertIn("/upload/image", mock_post.call_args_list[0].args[0])
            self.assertIn("/prompt", mock_post.call_args_list[1].args[0])

    def test_returns_empty_list_when_reference_photo_missing(self):
        config.qwen_image["host_reference_image"] = os.path.join(
            tempfile.gettempdir(), "this-host-photo-does-not-exist.png"
        )

        with patch(
            "app.services.qwen_image.requests.post"
        ) as mock_post, patch(
            "app.services.qwen_image.llm.generate_image_prompt",
            return_value="a golden retriever running on a sunlit beach",
        ):
            results = qwen_image.generate_images_qwen(
                search_term="golden retriever beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, [])
        mock_post.assert_not_called()

    def test_returns_empty_list_when_host_portrait_cache_write_fails(self):
        with patch(
            "app.services.qwen_image._get_host_reference_filename",
            side_effect=PermissionError("cache directory is read-only"),
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
