import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import tiktok_shop

SHARE_URL_TEMPLATE = (
    "https://shop.tiktok.com/jp/pdp/1735290027543529248"
    "?_d=f0k9g2d8jm6jhf&og_info={og}&share_region=JP"
)
PLAIN_URL = (
    "https://shop.tiktok.com/jp/pdp/1731596597173191895"
    "?source=ecommerce_mall&enter_method=feed_list_recommended_for_you"
)


class TestProductId(unittest.TestCase):
    def test_extracts_id_from_a_pdp_url(self):
        self.assertEqual(
            tiktok_shop.parse_product_id(PLAIN_URL), "1731596597173191895"
        )

    def test_extracts_id_regardless_of_region_segment(self):
        self.assertEqual(
            tiktok_shop.parse_product_id("https://shop.tiktok.com/pdp/123456"),
            "123456",
        )

    def test_returns_none_for_a_non_product_url(self):
        self.assertIsNone(
            tiktok_shop.parse_product_id("https://shop.tiktok.com/jp/search?q=x")
        )

    def test_returns_none_for_junk(self):
        self.assertIsNone(tiktok_shop.parse_product_id("not a url"))


class TestShareParams(unittest.TestCase):
    """Shared links carry title+image in og_info, so no network is needed."""

    def _share_url(self, title="Enchenal Lab Shaver", image="https://img/x.webp"):
        og = quote(json.dumps({"title": title, "image": image}))
        return SHARE_URL_TEMPLATE.format(og=og)

    def test_reads_title_and_image_without_network(self):
        product = tiktok_shop.from_share_url(self._share_url())
        self.assertIsNotNone(product)
        self.assertEqual(product.title, "Enchenal Lab Shaver")
        self.assertEqual(product.image_urls, ["https://img/x.webp"])
        self.assertEqual(product.product_id, "1735290027543529248")

    def test_plus_signs_in_the_title_become_spaces(self):
        product = tiktok_shop.from_share_url(self._share_url(title="Fast Dry Towel"))
        self.assertEqual(product.title, "Fast Dry Towel")

    def test_returns_none_when_og_info_is_absent(self):
        self.assertIsNone(tiktok_shop.from_share_url(PLAIN_URL))

    def test_returns_none_when_og_info_is_not_json(self):
        self.assertIsNone(
            tiktok_shop.from_share_url(SHARE_URL_TEMPLATE.format(og="%7Bnope"))
        )

    def test_rewrites_a_thumbnail_to_the_origin_asset(self):
        # The CDN rejects the bare path and the signed query alike; only a
        # ~tplv template directive is served, so swap the resize for origin.
        thumb = (
            "https://p16-oec-sg.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/"
            "deadbeef~tplv-aphluv4xwc-resize-webp:260:260.webp?dr=1&t=2"
        )
        product = tiktok_shop.from_share_url(self._share_url(image=thumb))
        self.assertEqual(
            product.image_urls[0],
            "https://p16-oec-sg.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/"
            "deadbeef~tplv-aphluv4xwc-origin-image.webp",
        )

    def test_adds_a_template_directive_when_the_url_has_none(self):
        bare = (
            "https://p16-oec-sg.ibyteimg.com/tos-alisg-i-aphluv4xwc-sg/"
            "abc123?dr=15582&from=337"
        )
        product = tiktok_shop.from_share_url(self._share_url(image=bare))
        self.assertTrue(product.image_urls[0].endswith(
            "~tplv-aphluv4xwc-origin-image.webp"))
        self.assertNotIn("?", product.image_urls[0])

    def test_non_bytedance_urls_are_left_alone(self):
        other = "https://example.com/photo.jpg?size=1"
        product = tiktok_shop.from_share_url(self._share_url(image=other))
        self.assertEqual(product.image_urls[0], other)


class TestFetchProduct(unittest.TestCase):
    """Network fallback for links that carry no og_info."""

    def _response(self, html, status=200):
        class _R:
            status_code = status
            text = html

        return _R()

    def test_reads_open_graph_tags(self):
        html = (
            '<html><head>'
            '<meta property="og:title" content="Portable Shaver X1"/>'
            '<meta property="og:image" content="https://img/a.jpg"/>'
            "</head></html>"
        )
        with patch.object(tiktok_shop.requests, "get", return_value=self._response(html)):
            product = tiktok_shop.fetch_product(PLAIN_URL)
        self.assertEqual(product.title, "Portable Shaver X1")
        self.assertEqual(product.image_urls, ["https://img/a.jpg"])

    def test_raises_when_the_page_has_no_recognisable_product(self):
        with patch.object(
            tiktok_shop.requests, "get", return_value=self._response("<html></html>")
        ):
            with self.assertRaises(tiktok_shop.TikTokShopError):
                tiktok_shop.fetch_product(PLAIN_URL)

    def test_raises_on_non_200(self):
        with patch.object(
            tiktok_shop.requests, "get", return_value=self._response("", status=403)
        ):
            with self.assertRaises(tiktok_shop.TikTokShopError) as ctx:
                tiktok_shop.fetch_product(PLAIN_URL)
        self.assertIn("403", str(ctx.exception))

    def test_network_failure_is_wrapped(self):
        with patch.object(
            tiktok_shop.requests,
            "get",
            side_effect=tiktok_shop.requests.RequestException("boom"),
        ):
            with self.assertRaises(tiktok_shop.TikTokShopError):
                tiktok_shop.fetch_product(PLAIN_URL)


class TestLoadProduct(unittest.TestCase):
    """The public helper: share params first, network only if needed."""

    def test_prefers_share_params_and_never_touches_the_network(self):
        og = quote(json.dumps({"title": "T", "image": "https://img/x.jpg"}))
        with patch.object(tiktok_shop.requests, "get") as get:
            product = tiktok_shop.load_product(SHARE_URL_TEMPLATE.format(og=og))
        get.assert_not_called()
        self.assertEqual(product.title, "T")

    def test_falls_back_to_fetching_when_share_params_are_absent(self):
        html = '<meta property="og:title" content="Fetched"/><meta property="og:image" content="https://i/a.jpg"/>'

        class _R:
            status_code = 200
            text = html

        with patch.object(tiktok_shop.requests, "get", return_value=_R()) as get:
            product = tiktok_shop.load_product(PLAIN_URL)
        get.assert_called_once()
        self.assertEqual(product.title, "Fetched")



class TestHtmlEntityUnescaping(unittest.TestCase):
    """Attribute values are HTML-escaped; a literal &amp; makes the CDN 400."""

    def test_image_url_query_separators_are_unescaped(self):
        html = (
            '<meta property="og:title" content="Shaver &amp; Trimmer"/>'
            '<meta property="og:image" content="https://p16.img/a?dr=1&amp;t=2&amp;ps=3"/>'
        )

        class _R:
            status_code = 200
            text = html

        with patch.object(tiktok_shop.requests, "get", return_value=_R()):
            product = tiktok_shop.fetch_product(PLAIN_URL)
        self.assertEqual(product.image_urls[0], "https://p16.img/a?dr=1&t=2&ps=3")
        self.assertNotIn("&amp;", product.image_urls[0])

    def test_title_entities_are_unescaped_too(self):
        html = '<meta property="og:title" content="Shaver &amp; Trimmer"/>'

        class _R:
            status_code = 200
            text = html

        with patch.object(tiktok_shop.requests, "get", return_value=_R()):
            product = tiktok_shop.fetch_product(PLAIN_URL)
        self.assertEqual(product.title, "Shaver & Trimmer")

if __name__ == "__main__":
    unittest.main()
