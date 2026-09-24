import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import product_pipeline
from app.services.product_video import ProductInfo, Scene


_resolve_patch = patch.object(
    product_pipeline.minimax_video,
    "resolve_references",
    side_effect=lambda refs: list(refs),
)


def setUpModule():
    _resolve_patch.start()


def tearDownModule():
    _resolve_patch.stop()


def _product():
    return ProductInfo(
        name="Portable Shaver",
        images=["https://img/shaver.png"],
        features=["net blade"],
    )


def _scene(dialogue="It just works."):
    return Scene(setting="a bathroom", beats=["holds it up"], dialogue=dialogue)


def _material(path="/tmp/clip.mp4"):
    return SimpleNamespace(url=path)


class TestDryRun(unittest.TestCase):
    def test_builds_prompts_without_calling_the_video_api(self):
        with patch.object(product_pipeline.minimax_video, "generate_product_clip") as gen:
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene(), _scene("Second line.")], dry_run=True
            )
        gen.assert_not_called()
        self.assertEqual(len(result.prompts), 2)
        self.assertEqual(result.generations_spent, 0)
        self.assertFalse(result.ok)

    def test_dry_run_prompts_contain_the_dialogue(self):
        result = product_pipeline.create_product_video(
            _product(), scenes=[_scene("Closest shave ever.")], dry_run=True
        )
        self.assertIn("Closest shave ever.", result.prompts[0])


class TestSceneSourcing(unittest.TestCase):
    def test_supplied_scenes_skip_the_llm(self):
        with patch.object(product_pipeline.product_video, "generate_scenes") as gen:
            product_pipeline.create_product_video(
                _product(), scenes=[_scene()], dry_run=True
            )
        gen.assert_not_called()

    def test_falls_back_to_generating_scenes(self):
        with patch.object(
            product_pipeline.product_video, "generate_scenes", return_value=[_scene()]
        ) as gen:
            result = product_pipeline.create_product_video(_product(), dry_run=True)
        gen.assert_called_once()
        self.assertEqual(len(result.scenes), 1)

    def test_reports_failure_when_no_scenes_are_produced(self):
        with patch.object(
            product_pipeline.product_video, "generate_scenes", return_value=[]
        ):
            result = product_pipeline.create_product_video(_product())
        self.assertFalse(result.ok)
        self.assertTrue(any("no scenes" in f for f in result.failures))


class TestGeneration(unittest.TestCase):
    def test_assembles_clips_and_returns_the_output_path(self):
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ), patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg") as concat:
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene(), _scene("b")], output_path="/tmp/out.mp4"
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.output_path, "/tmp/out.mp4")
        self.assertEqual(len(result.clip_paths), 2)
        concat.assert_called_once()

    def test_one_failed_scene_does_not_abort_the_run(self):
        # Partial output beats no output: the rest of the ad is still usable.
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            side_effect=[[], [_material()]],
        ), patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene(), _scene("b")], output_path="/tmp/o.mp4"
            )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.clip_paths), 1)
        self.assertTrue(any("scene 1" in f for f in result.failures))

    def test_counts_every_attempt_as_a_paid_generation(self):
        # Failed attempts can still bill, so the count must not skip them.
        with patch.object(
            product_pipeline.minimax_video, "generate_product_clip", return_value=[]
        ):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene(), _scene("b")]
            )
        self.assertEqual(result.generations_spent, 2)

    def test_reports_failure_when_every_scene_fails(self):
        with patch.object(
            product_pipeline.minimax_video, "generate_product_clip", return_value=[]
        ):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene()]
            )
        self.assertFalse(result.ok)
        self.assertTrue(any("every scene failed" in f for f in result.failures))

    def test_assembly_failure_is_reported_not_raised(self):
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ), patch.object(
            product_pipeline.video,
            "concat_video_clips_with_ffmpeg",
            side_effect=RuntimeError("ffmpeg exploded"),
        ):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path="/tmp/o.mp4"
            )
        self.assertFalse(result.ok)
        self.assertTrue(any("assembly failed" in f for f in result.failures))


class TestProductFromTikTok(unittest.TestCase):
    def test_uses_the_parsed_listing(self):
        parsed = SimpleNamespace(
            title="Shaver X", image_urls=["https://i/a.jpg"], source_url="https://s/u"
        )
        with patch("app.services.tiktok_shop.load_product", return_value=parsed):
            product = product_pipeline.product_from_tiktok("https://s/u")
        self.assertEqual(product.name, "Shaver X")
        self.assertEqual(product.images, ["https://i/a.jpg"])

    def test_overrides_win_over_the_listing(self):
        # A seller's own name and photos beat a scraped thumbnail.
        parsed = SimpleNamespace(
            title="Scraped", image_urls=["https://i/thumb.jpg"], source_url="u"
        )
        with patch("app.services.tiktok_shop.load_product", return_value=parsed):
            product = product_pipeline.product_from_tiktok(
                "u", name="Real Name", images=["C:/photos/hero.png"]
            )
        self.assertEqual(product.name, "Real Name")
        self.assertEqual(product.images, ["C:/photos/hero.png"])

class TestOutputDirectory(unittest.TestCase):
    def test_creates_a_missing_output_directory(self):
        # ffmpeg will not mkdir, and failing here throws away paid clips.
        import tempfile, os
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = os.path.join(tmp.name, "nested", "deeper", "out.mp4")
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ), patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path=target
            )
        self.assertTrue(os.path.isdir(os.path.dirname(target)))
        self.assertTrue(result.ok)

class TestCharacterReference(unittest.TestCase):
    """A fixed presenter is what makes several scenes read as one creator."""

    def test_character_is_sent_first_then_the_product(self):
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ) as gen, patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path="/tmp/o.mp4",
                character_image="C:/frames/creator.png",
            )
        references = gen.call_args.args[2]
        self.assertEqual(references[0], "C:/frames/creator.png")
        self.assertEqual(references[1:], ["https://img/shaver.png"])

    def test_without_a_character_only_product_images_are_sent(self):
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ) as gen, patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path="/tmp/o.mp4"
            )
        self.assertEqual(gen.call_args.args[2], ["https://img/shaver.png"])


class TestOutfitPlumbing(unittest.TestCase):
    def test_outfit_reaches_every_prompt(self):
        result = product_pipeline.create_product_video(
            _product(), scenes=[_scene(), _scene("b")],
            outfit="a plain grey t-shirt", dry_run=True,
        )
        self.assertTrue(all("a plain grey t-shirt" in p for p in result.prompts))

    def test_request_carries_outfit(self):
        request = product_pipeline.ProductVideoRequest(
            product=_product(), outfit="a hoodie"
        )
        self.assertEqual(request.outfit, "a hoodie")


class TestReferencesUploadOncePerRun(unittest.TestCase):
    """A 4-scene run with 9 references used to upload 36 times."""

    def test_each_reference_is_resolved_once_across_all_scenes(self):
        _resolve_patch.stop()
        self.addCleanup(_resolve_patch.start)
        uploads = []

        def _resolve(source, **kwargs):
            uploads.append(source)
            return f"mm_file://{len(uploads)}"

        with patch.object(
            product_pipeline.minimax_video.minimax_media, "resolve_image_ref", _resolve
        ), patch.object(
            product_pipeline.minimax_video, "get_minimax_video_api_key", return_value="k"
        ), patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ) as gen, patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            product_pipeline.create_product_video(
                _product(),
                scenes=[_scene(), _scene("b"), _scene("c"), _scene("d")],
                character_image="C:/frames/creator.png",
                output_path="/tmp/o.mp4",
            )
        # 2 references (character + product), resolved once, not 4 x 2.
        self.assertEqual(len(uploads), 2)
        # Every scene received the already-resolved handles.
        for call in gen.call_args_list:
            self.assertEqual(call.args[2], ["mm_file://1", "mm_file://2"])

    def test_a_rejected_reference_stops_before_any_generation(self):
        with patch.object(
            product_pipeline.minimax_video,
            "resolve_references",
            side_effect=product_pipeline.minimax_media.MiniMaxMediaError("too small"),
        ), patch.object(
            product_pipeline.minimax_video, "generate_product_clip"
        ) as gen:
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path="/tmp/o.mp4"
            )
        gen.assert_not_called()
        self.assertEqual(result.generations_spent, 0)
        self.assertTrue(any("could not be prepared" in f for f in result.failures))


class TestProgressReporting(unittest.TestCase):
    """The WebUI sat at 0% for a 15-20 minute run."""

    def test_reports_writing_each_scene_and_assembly_in_order(self):
        events = []
        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ), patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            product_pipeline.create_product_video(
                _product(), scenes=[_scene(), _scene("b")], output_path="/tmp/o.mp4",
                on_progress=lambda *event: events.append(event),
            )
        stages = [(stage, current) for _, stage, current, _ in events]
        self.assertEqual(
            stages, [("writing", 0), ("scene", 1), ("scene", 2), ("assembling", 2)]
        )
        percents = [percent for percent, *_ in events]
        self.assertEqual(percents, sorted(percents))
        self.assertTrue(all(0 < p < 100 for p in percents))

    def test_a_failing_callback_does_not_stop_the_run(self):
        def boom(*_):
            raise RuntimeError("ui went away")

        with patch.object(
            product_pipeline.minimax_video,
            "generate_product_clip",
            return_value=[_material()],
        ), patch.object(product_pipeline.video, "concat_video_clips_with_ffmpeg"):
            result = product_pipeline.create_product_video(
                _product(), scenes=[_scene()], output_path="/tmp/o.mp4", on_progress=boom,
            )
        self.assertTrue(result.ok)

if __name__ == "__main__":
    unittest.main()
