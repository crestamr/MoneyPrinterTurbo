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


class TestMiniMaxPollTask(unittest.TestCase):
    def test_poll_task_returns_content_url_on_success(self):
        responses = [
            SimpleNamespace(
                status_code=200,
                json=lambda: {"task": {"status": "running"}},
            ),
            SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "task": {
                        "status": "succeeded",
                        "content": {"url": "https://cdn.example.com/video.mp4"},
                    }
                },
            ),
        ]
        with patch(
            "app.services.minimax_video.requests.get", side_effect=responses
        ), patch("app.services.minimax_video.time.sleep") as sleep_mock:
            content_url = minimax_video._poll_task(
                task_id="task-abc123",
                api_key="test-key",
                base_url="https://api.minimax.io",
                poll_interval_seconds=1.0,
                poll_timeout_seconds=10.0,
            )

        self.assertEqual(content_url, "https://cdn.example.com/video.mp4")
        sleep_mock.assert_called_once()

    def test_poll_task_raises_with_message_on_failure(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "task": {
                    "status": "failed",
                    "error": {"code": "2013", "message": "invalid parameters"},
                }
            },
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )
        self.assertIn("invalid parameters", str(ctx.exception))

    def test_poll_task_raises_on_timeout(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task": {"status": "running"}},
        )
        clock = iter([0.0, 0.0, 11.0])
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ), patch("app.services.minimax_video.time.sleep"), patch(
            "app.services.minimax_video.time.monotonic", side_effect=lambda: next(clock)
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )

    def test_poll_task_raises_immediately_when_task_status_is_missing(self):
        # Regression test: a response missing "status" entirely must fail
        # fast rather than being treated as "still pending" and polled
        # until poll_timeout_seconds elapses. requests.get is only stubbed
        # to return once (return_value, not an iterator/side_effect list),
        # and time.sleep is patched to explode if called -- so if the
        # missing-status check regresses back to silently polling, this
        # test fails loudly instead of hanging for the full timeout.
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task": {}},
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ), patch(
            "app.services.minimax_video.time.sleep",
            side_effect=AssertionError(
                "polling should not continue when task status is missing"
            ),
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=300.0,
                )
        self.assertIn("missing task status", str(ctx.exception))

    def test_poll_task_raises_when_task_status_is_empty_string(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task": {"status": ""}},
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ), patch(
            "app.services.minimax_video.time.sleep",
            side_effect=AssertionError(
                "polling should not continue when task status is empty"
            ),
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=300.0,
                )
        self.assertIn("missing task status", str(ctx.exception))

    def test_poll_task_raises_on_non_200_status(self):
        fake_response = SimpleNamespace(
            status_code=500,
            json=lambda: {"error": "internal error"},
            text="internal error",
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )
        self.assertEqual(ctx.exception.status_code, 500)

    def test_poll_task_raises_when_succeeded_but_content_url_is_missing(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task": {"status": "succeeded", "content": {}}},
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError) as ctx:
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )
        self.assertIn("no content url", str(ctx.exception))

    def test_poll_task_raises_when_succeeded_but_content_is_missing(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: {"task": {"status": "succeeded"}},
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )

    def test_poll_task_raises_on_request_exception(self):
        import requests as requests_module

        with patch(
            "app.services.minimax_video.requests.get",
            side_effect=requests_module.ConnectionError("boom"),
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )

    def test_poll_task_raises_when_response_body_is_not_a_dict(self):
        fake_response = SimpleNamespace(
            status_code=200,
            json=lambda: ["unexpected", "list"],
        )
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )

    def test_poll_task_raises_when_task_key_is_missing(self):
        fake_response = SimpleNamespace(status_code=200, json=lambda: {})
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )

    def test_poll_task_raises_on_invalid_json(self):
        def _raise_value_error():
            raise ValueError("not json")

        fake_response = SimpleNamespace(status_code=200, json=_raise_value_error)
        with patch(
            "app.services.minimax_video.requests.get", return_value=fake_response
        ):
            with self.assertRaises(minimax_video.MiniMaxVideoAPIError):
                minimax_video._poll_task(
                    task_id="task-abc123",
                    api_key="test-key",
                    base_url="https://api.minimax.io",
                    poll_interval_seconds=1.0,
                    poll_timeout_seconds=10.0,
                )


class TestMiniMaxGenerateVideos(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_minimax_video_config = dict(config.minimax_video)
        self.temp_dir = tempfile.mkdtemp()
        self.patched_storage_dir = patch(
            "app.services.minimax_video.utils.storage_dir",
            return_value=self.temp_dir,
        )
        self.patched_storage_dir.start()

    def tearDown(self):
        self.patched_storage_dir.stop()
        config.app.clear()
        config.app.update(self.original_app_config)
        config.minimax_video.clear()
        config.minimax_video.update(self.original_minimax_video_config)

    def test_generate_videos_minimax_happy_path(self):
        config.minimax_video["api_key"] = "test-key"
        config.minimax_video["model"] = "MiniMax-H3"
        config.minimax_video["resolution"] = "768P"
        config.minimax_video["poll_interval_seconds"] = 0.01
        config.minimax_video["poll_timeout_seconds"] = 5

        create_response = SimpleNamespace(
            status_code=200, json=lambda: {"task_id": "task-xyz"}
        )
        poll_response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "task": {
                    "status": "succeeded",
                    "content": {"url": "https://cdn.example.com/video.mp4"},
                }
            },
        )
        download_response = SimpleNamespace(content=b"fake-video-bytes")

        with patch(
            "app.services.minimax_video.requests.post",
            return_value=create_response,
        ), patch(
            "app.services.minimax_video.requests.get",
            side_effect=[poll_response, download_response],
        ):
            results = minimax_video.generate_videos_minimax(
                search_term="golden retriever running on beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "minimax")
        self.assertTrue(os.path.exists(item.url))
        with open(item.url, "rb") as f:
            self.assertEqual(f.read(), b"fake-video-bytes")
        self.assertEqual(item.duration, 5)
        self.assertEqual(item.source_info["asset_id"], "task-xyz")
        self.assertEqual(
            item.source_info["search_term"], "golden retriever running on beach"
        )

    def test_generate_videos_minimax_returns_empty_list_without_api_key(self):
        config.minimax_video["api_key"] = ""
        config.app["minimax_api_key"] = ""

        results = minimax_video.generate_videos_minimax(
            search_term="golden retriever running on beach",
            minimum_duration=5,
            video_aspect=VideoAspect.portrait,
        )

        self.assertEqual(results, [])

    def test_generate_videos_minimax_returns_empty_list_on_api_error(self):
        config.minimax_video["api_key"] = "test-key"

        with patch(
            "app.services.minimax_video.requests.post",
            side_effect=minimax_video.requests.RequestException("boom"),
        ):
            results = minimax_video.generate_videos_minimax(
                search_term="golden retriever running on beach",
                minimum_duration=5,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
