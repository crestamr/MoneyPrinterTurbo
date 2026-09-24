"""Best-effort product details from a TikTok Shop product (PDP) link.

Two paths, cheapest first:

1. **Share links carry their own metadata.** A link copied from the app has an
   ``og_info`` query parameter holding the product title and hero image, so the
   common case needs no network request and cannot be broken by markup changes.
2. **Plain links are fetched** and scraped for Open Graph tags.

Path 2 is deliberately thin. TikTok's PDP is JavaScript-rendered and its markup
changes without notice, so this reads only the two tags that have stayed stable
and raises a clear error otherwise. Anything parsed here is a *starting point* a
human is expected to correct - the caller must let the user edit the title and
supply better images, because a seller's own product shots consistently beat a
listing thumbnail as video-generation input.
"""

from __future__ import annotations

import html as html_module
import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote_plus, urlparse

import requests
from loguru import logger

DEFAULT_TIMEOUT_SECONDS = 20.0

# A desktop UA: the PDP returns a near-empty shell to unknown clients.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)

_PRODUCT_ID_PATTERN = re.compile(r"/pdp/(\d+)")
_OG_TAG_PATTERN = (
    '<meta[^>]+property=["\']og:{name}["\'][^>]+content=["\']([^"\']+)["\']'
)
_OG_TAG_PATTERN_REVERSED = (
    '<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:{name}["\']'
)

# ByteDance CDN URLs *require* a "~tplv-<token>-..." template directive: the
# bare object path returns HTTP 400, and so does the signed query string a
# listing hands out. The token is the bucket id embedded in the path
# ("tos-alisg-i-<token>-sg"), so an origin-sized asset can be requested
# directly - verified 2000x2000 against a 260x260 listing thumbnail.
_CDN_TEMPLATE_SUFFIX = re.compile(r"~tplv[^/?]*")
_CDN_BUCKET_TOKEN = re.compile(r"/tos-[a-z0-9]+-i-([a-z0-9]+)(?:-[a-z0-9]+)?/")
_CDN_HOST_HINT = "ibyteimg.com"


class TikTokShopError(RuntimeError):
    """Raised when a product cannot be read from a TikTok Shop link."""


@dataclass
class TikTokProduct:
    """What could be recovered from a listing. Treat every field as a draft."""

    product_id: str
    title: str
    image_urls: list[str] = field(default_factory=list)
    source_url: str = ""


def parse_product_id(url: str) -> "str | None":
    """Return the numeric product id in a PDP URL, or None if there isn't one."""
    if not url or not isinstance(url, str):
        return None
    match = _PRODUCT_ID_PATTERN.search(url)
    return match.group(1) if match else None


def _clean_image_url(url: str) -> str:
    """Rewrite a listing image URL to the full-size original.

    A listing hands out a 260x260 thumbnail carrying signed query parameters
    that are rejected outside the page. Swapping the template directive for
    the origin one drops the query and yields the full asset, which matters
    because the model needs a clear look at the product.

    Non-ByteDance URLs are returned untouched.
    """
    url = (url or "").strip()
    if not url or _CDN_HOST_HINT not in url:
        return url

    token_match = _CDN_BUCKET_TOKEN.search(url)
    if not token_match:
        return url

    base = url.split("?", 1)[0]
    base = _CDN_TEMPLATE_SUFFIX.sub("", base)
    return f"{base}~tplv-{token_match.group(1)}-origin-image.webp"


def from_share_url(url: str) -> "TikTokProduct | None":
    """Read title/image straight out of a share link's ``og_info`` parameter.

    Returns None (never raises) when the parameter is missing or unparseable,
    so the caller can fall through to fetching the page.
    """
    product_id = parse_product_id(url)
    if not product_id:
        return None
    try:
        query = parse_qs(urlparse(url).query)
    except ValueError:
        return None

    raw = (query.get("og_info") or [""])[0]
    if not raw:
        return None
    try:
        info = json.loads(unquote_plus(raw))
    except (ValueError, TypeError):
        logger.debug("TikTok share link had an unparseable og_info parameter")
        return None
    if not isinstance(info, dict):
        return None

    title = unquote_plus(str(info.get("title") or "")).strip()
    image = _clean_image_url(unquote_plus(str(info.get("image") or "")))
    if not title and not image:
        return None

    return TikTokProduct(
        product_id=product_id,
        title=title,
        image_urls=[image] if image else [],
        source_url=url,
    )


def _find_og(html: str, name: str) -> str:
    """Read one og: meta tag, unescaping HTML entities in the value.

    Attribute values are HTML-escaped, so a CDN image URL arrives with its
    query separators as "&amp;". Passing that through verbatim makes the CDN
    return HTTP 400 - which looks like a blocked download rather than a
    parsing bug, so it is unescaped here at the source.
    """
    for pattern in (_OG_TAG_PATTERN, _OG_TAG_PATTERN_REVERSED):
        match = re.search(pattern.format(name=name), html, re.IGNORECASE)
        if match:
            return html_module.unescape(match.group(1).strip())
    return ""


def fetch_product(
    url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> TikTokProduct:
    """Fetch a PDP and scrape its Open Graph tags.

    Raises TikTokShopError on any failure, including a page that loaded but
    carried nothing recognisable - which is the expected outcome if TikTok
    changes its markup, and is why the caller must offer manual entry.
    """
    product_id = parse_product_id(url) or ""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "en,ja;q=0.8"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TikTokShopError(
            f"could not reach TikTok Shop: {type(exc).__name__}"
        ) from exc

    if getattr(response, "status_code", None) != 200:
        raise TikTokShopError(
            f"TikTok Shop returned HTTP {response.status_code} for {url!r}"
        )

    html = str(getattr(response, "text", "") or "")
    title = _find_og(html, "title")
    image = _clean_image_url(_find_og(html, "image"))
    if not title and not image:
        raise TikTokShopError(
            "no product details found on the page - TikTok renders its listings "
            "with JavaScript and may have changed its markup; enter the product "
            "name and images manually"
        )

    return TikTokProduct(
        product_id=product_id,
        title=title,
        image_urls=[image] if image else [],
        source_url=url,
    )


def load_product(
    url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> TikTokProduct:
    """Share parameters when present, otherwise fetch the page."""
    shared = from_share_url(url)
    if shared is not None:
        logger.info(f"read TikTok product from share link: id={shared.product_id}")
        return shared
    return fetch_product(url, timeout=timeout)
