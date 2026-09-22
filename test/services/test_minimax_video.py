import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoAspect
from app.services import minimax_video


class TestMiniMaxVideoSettings(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_minimax_video_config = dict(config.minimax_video)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.minimax_video.clear()
        config.minimax_video.update(self.original_minimax_video_config)

    def test_falls_back_to_llm_key_when_dedicated_key_is_empty(self):
        config.minimax_video["api_key"] = ""
        config.app["minimax_api_key"] = "llm-shared-key"

        self.assertEqual(
            minimax_video.get_minimax_video_api_key(), "llm-shared-key"
        )

    def test_dedicated_key_takes_priority_over_llm_key(self):
        config.minimax_video["api_key"] = "dedicated-video-key"
        config.app["minimax_api_key"] = "llm-shared-key"

        self.assertEqual(
            minimax_video.get_minimax_video_api_key(), "dedicated-video-key"
        )

    def test_base_url_defaults_to_global_when_unset(self):
        config.minimax_video["base_url"] = ""
        config.minimax_video["api_key"] = ""
        config.app["minimax_base_url"] = ""

        self.assertEqual(
            minimax_video.get_minimax_video_base_url(),
            "https://api.minimax.io",
        )

    def test_base_url_follows_llm_region_when_using_shared_key(self):
        config.minimax_video["base_url"] = ""
        config.minimax_video["api_key"] = ""
        config.app["minimax_base_url"] = "https://api.minimaxi.com/v1"

        self.assertEqual(
            minimax_video.get_minimax_video_base_url(),
            "https://api.minimaxi.com",
        )

    def test_dedicated_base_url_is_respected_when_key_is_dedicated(self):
        config.minimax_video["base_url"] = "https://api.minimaxi.com"
        config.minimax_video["api_key"] = "dedicated-video-key"
        config.app["minimax_base_url"] = "https://api.minimax.io/v1"

        self.assertEqual(
            minimax_video.get_minimax_video_base_url(),
            "https://api.minimaxi.com",
        )


class TestMiniMaxVideoMapping(unittest.TestCase):
    def test_build_prompt_excludes_text_and_captions(self):
        prompt = minimax_video._build_prompt("golden retriever running on beach")

        self.assertIn("golden retriever running on beach", prompt)
        self.assertIn("no text", prompt.lower())
        self.assertIn("no captions", prompt.lower())

    def test_aspect_ratio_matches_minimax_enum_values(self):
        self.assertEqual(
            minimax_video._aspect_to_ratio(VideoAspect.portrait), "9:16"
        )
        self.assertEqual(
            minimax_video._aspect_to_ratio(VideoAspect.landscape), "16:9"
        )
        self.assertEqual(minimax_video._aspect_to_ratio(VideoAspect.square), "1:1")

    def test_clamp_duration_within_model_range(self):
        self.assertEqual(
            minimax_video._clamp_duration(5, model="MiniMax-H3"), 5
        )

    def test_clamp_duration_below_minimum_is_raised_to_minimum(self):
        self.assertEqual(
            minimax_video._clamp_duration(2, model="MiniMax-H3"), 4
        )

    def test_clamp_duration_above_maximum_is_lowered_to_maximum(self):
        self.assertEqual(
            minimax_video._clamp_duration(30, model="MiniMax-H3"), 15
        )

    def test_clamp_duration_uses_h3_max_range(self):
        self.assertEqual(
            minimax_video._clamp_duration(2, model="MiniMax-H3-Max"), 5
        )


class TestMiniMaxCreateTask(unittest.TestCase):
    def test_create_task_returns_task_id_on_success(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task_id": "task-abc123"},
        )
        with patch(
            "app.services.minimax_video.requests.post",
            return_value=fake_response,
        ) as post:
            task_id = minimax_video._create_task(
                prompt="a cat playing piano",
                resolution="768P",
                duration=5,
                ratio="9:16",
                model="MiniMax-H3",
                api_key="test-key",
                base_url="https://api.minimax.io",
            )

        self.assertEqual(task_id, "task-abc123")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["content"],
            [{"type": "text", "text": "a cat playing piano"}],
        )

    def test_create_task_raises_on_non_200_status(self):
        fake_response = SimpleNamespace(
            status_code=402,
            json=lambda: {"error": "insufficient balance"},
            text="insufficient balance",
        )
        with patch(
            "app.services.minimax_video.requests.post",
            return_value=fake_response,
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._create_task(
                    prompt="a cat playing piano",
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                    model="MiniMax-H3",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                )
        self.assertEqual(ctx.exception.status_code, 402)

    def test_create_task_raises_when_task_id_is_missing(self):
        fake_response = SimpleNamespace(status_code=200, json=lambda: {})
        with patch(
            "app.services.minimax_video.requests.post",
            return_value=fake_response,
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._create_task(
                    prompt="a cat playing piano",
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                    model="MiniMax-H3",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                )

    def test_create_task_raises_when_task_id_is_null(self):
        fake_response = SimpleNamespace(
            status_code=200, json=lambda: {"task_id": None}
        )
        with patch(
            "app.services.minimax_video.requests.post",
            return_value=fake_response,
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._create_task(
                    prompt="a cat playing piano",
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                    model="MiniMax-H3",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                )

    def test_create_task_raises_when_response_body_is_not_a_dict(self):
        fake_response = SimpleNamespace(
            status_code=200, json=lambda: ["unexpected", "list"]
        )
        with patch(
            "app.services.minimax_video.requests.post",
            return_value=fake_response,
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._create_task(
                    prompt="a cat playing piano",
                    resolution="768P",
                    duration=5,
                    ratio="9:16",
                    model="MiniMax-H3",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                )


if __name__ == "__main__":
    unittest.main()
