import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
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


if __name__ == "__main__":
    unittest.main()
