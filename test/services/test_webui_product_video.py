"""WebUI integration for the TikTok Shop product-video source.

The product pipeline is separate from the normal task flow, so these tests
pin the two things that separation makes easy to get wrong: the form only
appears for this source, and the generation branch runs *before* the script,
voice and subtitle validation that product videos have no use for.
"""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _widget_by_key(elements, key):
    return next(
        item
        for item in elements
        if str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
    )


def _has_key(elements, key):
    return any(
        str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
        for item in elements
    )


def _app(**config_overrides):
    test_config = dict(
        config.app,
        llm_provider="openai",
        video_source="pexels",
        minimax_api_key="test-key",
        **config_overrides,
    )
    return test_config


def test_product_form_appears_only_for_the_product_source():
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()

        # Not shown for a stock source.
        assert not _has_key(app.text_input, "product_video_name")

        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        assert _has_key(app.text_input, "product_video_name")
        assert _has_key(app.text_input, "product_video_features")
        assert _has_key(app.number_input, "product_video_scenes")
        assert [str(item.value) for item in app.exception] == []


def test_cost_warning_names_the_number_of_paid_generations():
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        _widget_by_key(app.number_input, "product_video_scenes").set_value(3).run()
        warnings = " ".join(str(item.value) for item in app.warning)
        assert "3" in warnings and "paid" in warnings.lower()


def test_submitting_without_a_product_name_is_rejected():
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_product_generation") as submit,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        _widget_by_key(app.button, "generate_video_button").click().run()
        assert submit.call_count == 0
        assert any("product name" in str(item.value).lower() for item in app.error)


def test_submitting_without_an_image_is_rejected():
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_product_generation") as submit,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        _widget_by_key(app.text_input, "product_video_name").set_value("Shaver").run()
        _widget_by_key(app.button, "generate_video_button").click().run()
        assert submit.call_count == 0
        assert any("image" in str(item.value).lower() for item in app.error)


def test_a_complete_form_queues_a_product_request():
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_product_generation") as submit,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        _widget_by_key(app.text_input, "product_video_name").set_value(
            "Dowinx gaming chair"
        ).run()
        _widget_by_key(app.text_input, "product_video_features").set_value(
            "lumbar support, 165 degree recline"
        ).run()
        # A listing image stands in for an upload; remote URLs skip local
        # validation because the provider fetches them itself.
        app.session_state["product_video_listing_images"] = ["https://img/a.jpg"]
        _widget_by_key(app.number_input, "product_video_scenes").set_value(2).run()

        _widget_by_key(app.button, "generate_video_button").click().run()

        assert submit.call_count == 1
        request = submit.call_args.kwargs["request"]
        assert request.product.name == "Dowinx gaming chair"
        assert request.product.features == ["lumbar support", "165 degree recline"]
        assert request.product.images == ["https://img/a.jpg"]
        assert request.scene_count == 2
        assert [str(item.value) for item in app.exception] == []


def test_product_source_skips_the_script_requirement():
    """A product video has no script, so the usual emptiness check must not fire."""
    with (
        patch.object(config, "app", _app()),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_product_generation") as submit,
        patch("app.services.webui_task.submit_generation") as submit_normal,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.session_state["video_source_select_en"] = "minimax_product"
        app.run()

        _widget_by_key(app.text_input, "product_video_name").set_value("Thing").run()
        app.session_state["product_video_listing_images"] = ["https://img/a.jpg"]
        _widget_by_key(app.button, "generate_video_button").click().run()

        # Subject and script are both empty, yet it submits.
        assert submit.call_count == 1
        assert submit_normal.call_count == 0
        errors = " ".join(str(item.value).lower() for item in app.error)
        assert "cannot both be empty" not in errors


def test_normal_sources_still_use_the_ordinary_pipeline():
    """The new branch must not divert anything else."""
    with (
        patch.object(config, "app", _app(pexels_api_keys="k")),
        patch.object(config, "try_save_config", return_value=True),
        patch("app.services.webui_task.submit_generation") as submit_normal,
        patch("app.services.webui_task.submit_product_generation") as submit_product,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()

        _widget_by_key(app.text_area, "video_subject").set_value("Space fleet").run()
        _widget_by_key(app.button, "generate_video_button").click().run()

        assert submit_normal.call_count == 1
        assert submit_product.call_count == 0
