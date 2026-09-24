import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import minimax_media


def _write_png(path: str, width: int = 512, height: int = 512) -> str:
    from PIL import Image

    Image.new("RGB", (width, height), (120, 140, 160)).save(path, format="PNG")
    return path


class TestValidateImage(unittest.TestCase):
    """Local validation, so a bad image never costs an API round trip.

    The docs publish the limits but no error code for breaching them, so we
    refuse locally rather than pattern-matching a guessed remote code.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_accepts_a_normal_png(self):
        spec = minimax_media.validate_image(_write_png(self._path("a.png")))
        self.assertEqual(spec.width, 512)
        self.assertEqual(spec.height, 512)

    def test_rejects_missing_file(self):
        with self.assertRaises(minimax_media.MiniMaxMediaError):
            minimax_media.validate_image(self._path("nope.png"))

    def test_rejects_unsupported_extension(self):
        p = self._path("a.gif")
        _write_png(self._path("a.png"))
        os.replace(self._path("a.png"), p)
        with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
            minimax_media.validate_image(p)
        self.assertIn("format", str(ctx.exception).lower())

    def test_rejects_image_below_minimum_dimension(self):
        p = _write_png(self._path("small.png"), width=100, height=100)
        with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
            minimax_media.validate_image(p)
        self.assertIn("256", str(ctx.exception))

    def test_rejects_aspect_ratio_outside_supported_range(self):
        # 3000x512 -> 5.86, well beyond the documented 2.5 ceiling.
        p = _write_png(self._path("wide.png"), width=3000, height=512)
        with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
            minimax_media.validate_image(p)
        self.assertIn("aspect", str(ctx.exception).lower())

    def test_rejects_oversized_file(self):
        p = _write_png(self._path("big.png"))
        with patch.object(minimax_media.os.path, "getsize", return_value=31 * 1024 * 1024):
            with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
                minimax_media.validate_image(p)
        self.assertIn("30", str(ctx.exception))


class TestBuildContent(unittest.TestCase):
    """The payload shape verified against platform.minimax.io v2 docs."""

    def test_text_only_content_has_a_single_item(self):
        content = minimax_media.build_content("hello")
        self.assertEqual(content, [{"type": "text", "text": "hello"}])

    def test_role_is_a_sibling_of_image_url_not_nested(self):
        content = minimax_media.build_content(
            "p", reference_images=["mm_file://1"]
        )
        image_item = content[1]
        self.assertEqual(image_item["type"], "image_url")
        self.assertEqual(image_item["image_url"], {"url": "mm_file://1"})
        self.assertEqual(image_item["role"], "reference_image")
        self.assertNotIn("role", image_item["image_url"])

    def test_multiple_reference_images_are_preserved_in_order(self):
        refs = ["mm_file://1", "mm_file://2"]
        content = minimax_media.build_content("p", reference_images=refs)
        self.assertEqual([c["image_url"]["url"] for c in content[1:]], refs)
        self.assertTrue(all(c["role"] == "reference_image" for c in content[1:]))

    def test_first_frame_uses_the_first_frame_role(self):
        content = minimax_media.build_content("p", first_frame="https://x/a.png")
        self.assertEqual(content[1]["role"], "first_frame")

    def test_first_frame_and_reference_images_are_mutually_exclusive(self):
        # Sending both is a documented 400; refuse before spending a request.
        with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
            minimax_media.build_content(
                "p", first_frame="https://x/a.png", reference_images=["mm_file://1"]
            )
        self.assertIn("mutually exclusive", str(ctx.exception).lower())

    def test_rejects_more_than_nine_reference_images(self):
        with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
            minimax_media.build_content(
                "p", reference_images=[f"mm_file://{i}" for i in range(10)]
            )
        self.assertIn("9", str(ctx.exception))

    def test_rejects_empty_prompt(self):
        # Exactly one non-empty text item is required by the API.
        with self.assertRaises(minimax_media.MiniMaxMediaError):
            minimax_media.build_content("   ")


class TestRatioFor(unittest.TestCase):
    """Any image forces 'adaptive'; text-to-video must never send it."""

    def test_text_only_keeps_the_requested_ratio(self):
        content = minimax_media.build_content("p")
        self.assertEqual(minimax_media.ratio_for(content, "9:16"), "9:16")

    def test_any_image_forces_adaptive(self):
        content = minimax_media.build_content("p", reference_images=["mm_file://1"])
        self.assertEqual(minimax_media.ratio_for(content, "9:16"), "adaptive")

    def test_text_only_never_returns_adaptive(self):
        content = minimax_media.build_content("p")
        self.assertNotEqual(minimax_media.ratio_for(content, "adaptive"), "adaptive")


class TestResolveImageRef(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_https_url_is_passed_through_untouched(self):
        url = "https://example.com/a.png"
        self.assertEqual(
            minimax_media.resolve_image_ref(url, api_key="k", base_url="https://api"),
            url,
        )

    def test_mm_file_reference_is_passed_through_untouched(self):
        ref = "mm_file://12345"
        self.assertEqual(
            minimax_media.resolve_image_ref(ref, api_key="k", base_url="https://api"),
            ref,
        )

    def test_local_file_is_uploaded_and_returned_as_mm_file(self):
        path = _write_png(os.path.join(self.tmp.name, "p.png"))
        with patch.object(minimax_media, "upload_image", return_value="mm_file://77") as up:
            out = minimax_media.resolve_image_ref(
                path, api_key="k", base_url="https://api"
            )
        self.assertEqual(out, "mm_file://77")
        up.assert_called_once()


class TestUploadImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = _write_png(os.path.join(self.tmp.name, "p.png"))

    def test_posts_multipart_with_the_documented_purpose(self):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"file": {"file_id": 999}, "base_resp": {"status_code": 0}}

        with patch.object(minimax_media.requests, "post", return_value=_Resp()) as post:
            ref = minimax_media.upload_image(
                self.path, api_key="key", base_url="https://api.minimax.io"
            )
        self.assertEqual(ref, "mm_file://999")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["data"], {"purpose": "video_generation_input"})
        self.assertIn("file", kwargs["files"])
        self.assertEqual(post.call_args.args[0], "https://api.minimax.io/v1/files/upload")

    def test_raises_when_the_response_has_no_file_id(self):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"base_resp": {"status_code": 0}}

        with patch.object(minimax_media.requests, "post", return_value=_Resp()):
            with self.assertRaises(minimax_media.MiniMaxMediaError):
                minimax_media.upload_image(
                    self.path, api_key="key", base_url="https://api.minimax.io"
                )

    def test_surfaces_the_api_error_message_verbatim(self):
        # No documented code for a rejected image, so the remote message is the
        # only trustworthy signal - it must reach the caller intact.
        class _Resp:
            status_code = 400
            text = '{"error": {"message": "invalid params (2013)"}}'

            @staticmethod
            def json():
                return {"error": {"message": "invalid params (2013)"}}

        with patch.object(minimax_media.requests, "post", return_value=_Resp()):
            with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
                minimax_media.upload_image(
                    self.path, api_key="key", base_url="https://api.minimax.io"
                )
        self.assertIn("invalid params (2013)", str(ctx.exception))

class TestVideoDetection(unittest.TestCase):
    def test_recognises_common_video_extensions(self):
        for name in ("a.mp4", "b.MOV", "c.mkv", "d.webm"):
            self.assertTrue(minimax_media.is_video(name), name)

    def test_images_are_not_videos(self):
        for name in ("a.png", "b.jpg", "c.webp"):
            self.assertFalse(minimax_media.is_video(name), name)

    def test_empty_path_is_not_a_video(self):
        self.assertFalse(minimax_media.is_video(""))


class TestExtractFrame(unittest.TestCase):
    def test_missing_video_raises(self):
        with self.assertRaises(minimax_media.MiniMaxMediaError):
            minimax_media.extract_frame("nope.mp4")

    def test_failed_ffmpeg_raises_with_its_stderr(self):
        import tempfile, os as _os
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = _os.path.join(tmp.name, "v.mp4")
        open(fake, "wb").write(b"not really a video")

        class _Completed:
            returncode = 1
            stdout = ""
            stderr = "moov atom not found"

        with patch("subprocess.run", return_value=_Completed()):
            with self.assertRaises(minimax_media.MiniMaxMediaError) as ctx:
                minimax_media.extract_frame(fake)
        self.assertIn("moov atom not found", str(ctx.exception))


class TestResolveVideoRef(unittest.TestCase):
    def test_a_video_path_is_turned_into_a_frame_then_uploaded(self):
        with patch.object(
            minimax_media, "extract_frame", return_value="C:/frames/a.png"
        ) as extract, patch.object(
            minimax_media, "validate_image"
        ), patch.object(
            minimax_media, "upload_image", return_value="mm_file://5"
        ):
            out = minimax_media.resolve_image_ref(
                "C:/clips/creator.mp4", api_key="k", base_url="https://api"
            )
        extract.assert_called_once()
        self.assertEqual(out, "mm_file://5")


if __name__ == "__main__":
    unittest.main()


class TestFrameCacheKey(unittest.TestCase):
    def test_different_positions_produce_different_files(self):
        import tempfile, os as _os
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = _os.path.join(tmp.name, "v.mp4")
        open(fake, "wb").write(b"x")

        seen = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd, **kwargs):
            seen.append(cmd[-1])
            open(cmd[-1], "wb").write(b"png")
            return _Completed()

        # Redirect the cache dir instead of passing out_path: the point of
        # this test is the generated filename, and an earlier version left
        # stub PNGs behind in the real storage/product_images.
        from app.utils import utils as _utils

        with (
            patch("subprocess.run", _run),
            patch.object(_utils, "storage_dir", return_value=tmp.name),
            patch.object(minimax_media, "_video_duration", return_value=20.0),
        ):
            minimax_media.extract_frame(fake, position=0.1)
            minimax_media.extract_frame(fake, position=0.9)
        self.assertNotEqual(seen[0], seen[1])
        for path in seen:
            self.assertTrue(path.startswith(tmp.name), path)
