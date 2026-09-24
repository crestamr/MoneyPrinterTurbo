import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import product_video


def _product(**kw):
    defaults = dict(
        name="Enchenal Lab Portable Shaver",
        images=["C:/tmp/shaver.png"],
        features=["net blade", "low irritation", "travel size"],
        audience="men who travel",
    )
    defaults.update(kw)
    return product_video.ProductInfo(**defaults)


class TestProductInfo(unittest.TestCase):
    def test_requires_a_name(self):
        with self.assertRaises(ValueError):
            product_video.ProductInfo(name="  ", images=["a.png"])

    def test_requires_at_least_one_image(self):
        with self.assertRaises(ValueError):
            product_video.ProductInfo(name="Thing", images=[])

    def test_normalises_blank_features_away(self):
        product = _product(features=["real", "  ", ""])
        self.assertEqual(product.features, ["real"])


class TestBuildScenePrompt(unittest.TestCase):
    """The template is what buys authenticity, so assert its parts are present."""

    def setUp(self):
        self.scene = product_video.Scene(
            setting="a bright bathroom",
            beats=["holds it up and turns it", "runs it along the jaw"],
            dialogue="I take this everywhere now.",
        )

    def test_declares_the_reference_roles(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("reference the product", prompt.lower())
        self.assertIn("character reference", prompt.lower())

    def test_pins_single_take_and_no_transitions(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        lowered = prompt.lower()
        self.assertIn("single continuous long take", lowered)
        self.assertIn("no transitions", lowered)

    def test_keeps_music_out_so_the_spoken_line_survives(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("No music", prompt)

    def test_includes_the_dialogue_verbatim_in_quotes(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("I take this everywhere now.", prompt)

    def test_numbers_the_beats(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("Beat 1:", prompt)
        self.assertIn("Beat 2:", prompt)

    def test_carries_a_negative_prompt_covering_product_deformation(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        lowered = prompt.lower()
        self.assertIn("negative prompt:", lowered)
        self.assertIn("deformation", lowered)
        self.assertIn("extra fingers", lowered)

    def test_names_the_product_so_the_model_knows_what_it_is_holding(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("Enchenal Lab Portable Shaver", prompt)

    def test_mentions_the_setting(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("a bright bathroom", prompt)

    def test_stays_within_the_api_text_limit(self):
        # MiniMax caps the text item at 7000 characters.
        long_scene = product_video.Scene(
            setting="x" * 500,
            beats=["y" * 300] * 8,
            dialogue="z" * 500,
        )
        prompt = product_video.build_scene_prompt(_product(), long_scene)
        self.assertLessEqual(len(prompt), product_video.MAX_PROMPT_CHARS)


class TestParseScenes(unittest.TestCase):
    def test_parses_a_well_formed_llm_response(self):
        payload = json.dumps(
            {
                "scenes": [
                    {
                        "setting": "a bathroom",
                        "beats": ["holds it up", "shaves"],
                        "dialogue": "Honestly the closest shave I've had.",
                    }
                ]
            }
        )
        scenes = product_video.parse_scenes(payload)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].setting, "a bathroom")
        self.assertEqual(len(scenes[0].beats), 2)

    def test_tolerates_a_fenced_json_block(self):
        payload = "```json\n" + json.dumps({"scenes": [
            {"setting": "s", "beats": ["b"], "dialogue": "d"}]}) + "\n```"
        self.assertEqual(len(product_video.parse_scenes(payload)), 1)

    def test_tolerates_a_bare_list(self):
        payload = json.dumps([{"setting": "s", "beats": ["b"], "dialogue": "d"}])
        self.assertEqual(len(product_video.parse_scenes(payload)), 1)

    def test_skips_scenes_without_dialogue(self):
        payload = json.dumps({"scenes": [
            {"setting": "s", "beats": ["b"], "dialogue": ""},
            {"setting": "s2", "beats": ["b"], "dialogue": "kept"},
        ]})
        scenes = product_video.parse_scenes(payload)
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].dialogue, "kept")

    def test_returns_empty_for_unparseable_text(self):
        self.assertEqual(product_video.parse_scenes("not json at all"), [])


class TestGenerateScenes(unittest.TestCase):
    def test_asks_the_llm_and_returns_parsed_scenes(self):
        response = json.dumps({"scenes": [
            {"setting": "s", "beats": ["b1", "b2"], "dialogue": "d"}]})
        with patch.object(product_video.llm, "_generate_response", return_value=response) as gen:
            scenes = product_video.generate_scenes(_product(), scene_count=1)
        self.assertEqual(len(scenes), 1)
        prompt = gen.call_args.kwargs.get("prompt") or gen.call_args.args[0]
        self.assertIn("Enchenal Lab Portable Shaver", prompt)

    def test_returns_empty_when_the_llm_fails(self):
        with patch.object(
            product_video.llm, "_generate_response", side_effect=RuntimeError("down")
        ):
            self.assertEqual(product_video.generate_scenes(_product()), [])


class TestEstimateCost(unittest.TestCase):
    def test_scales_with_scene_count(self):
        one = product_video.estimate_generations(scene_count=1)
        four = product_video.estimate_generations(scene_count=4)
        self.assertEqual(one, 1)
        self.assertEqual(four, 4)



class TestDisplayName(unittest.TestCase):
    """Marketplace titles are keyword soup; the spoken name must be short."""

    LONG = (
        "Dowinx ゲーミング座椅子 ゲーミングチェア 腰が痛くならない 回転座椅子 おしゃれ "
        "ゲーム パソコンチェア 連動アームレスト ランバーサポート 165°リクライニング "
        "チェア ハイバック グレー LS-6679"
    )

    def test_short_names_are_untouched(self):
        self.assertEqual(_product(name="Portable Shaver").display_name, "Portable Shaver")

    def test_long_marketplace_title_is_trimmed(self):
        product = _product(name=self.LONG)
        self.assertLessEqual(
            len(product.display_name), product_video.MAX_DISPLAY_NAME_CHARS
        )
        self.assertTrue(product.display_name.startswith("Dowinx"))

    def test_trimming_happens_on_a_word_boundary(self):
        product = _product(name="Alpha Beta Gamma " + "x" * 80)
        self.assertFalse(product.display_name.endswith("x"))

    def test_full_name_is_preserved_on_the_model(self):
        # Only the spoken/display form is shortened; details keep the original.
        product = _product(name=self.LONG)
        self.assertEqual(product.name, self.LONG)

    def test_prompt_uses_the_short_name(self):
        product = _product(name=self.LONG)
        scene = product_video.Scene(setting="s", beats=["b"], dialogue="d")
        prompt = product_video.build_scene_prompt(product, scene)
        self.assertNotIn("LS-6679", prompt)


class TestBeatJoining(unittest.TestCase):
    def test_beats_are_joined_readably(self):
        joined = product_video._join_beats(
            ["Lifts it toward the desk", "Presses the lumbar pad"]
        )
        self.assertEqual(
            joined, "Lifts it toward the desk, then presses the lumbar pad."
        )

    def test_single_beat_gets_a_full_stop(self):
        self.assertEqual(product_video._join_beats(["Holds it up"]), "Holds it up.")

    def test_acronyms_keep_their_capitals(self):
        joined = product_video._join_beats(["Opens it", "USB cable is plugged in"])
        self.assertIn("USB cable", joined)

    def test_empty_beats_are_dropped(self):
        self.assertEqual(product_video._join_beats(["  ", "Holds it"]), "Holds it.")


class TestNeutralPhrasing(unittest.TestCase):
    def test_prompt_does_not_assume_the_product_is_handheld(self):
        # "holding" is wrong for furniture and appliances.
        product = _product(name="Gaming Chair")
        scene = product_video.Scene(setting="a desk", beats=["rolls it"], dialogue="d")
        prompt = product_video.build_scene_prompt(product, scene)
        self.assertNotIn("is holding the", prompt)
        self.assertIn("is presenting the", prompt)

class TestSceneInstructions(unittest.TestCase):
    """The LLM instruction has to pin person, or beats clash with the template."""

    def test_requires_third_person_beats(self):
        prompt = product_video._SCENE_SYSTEM_PROMPT
        self.assertIn("THIRD PERSON", prompt)
        self.assertIn('Never write', prompt)

    def test_forbids_ad_language(self):
        self.assertIn("buy now", product_video._SCENE_SYSTEM_PROMPT)


class TestOutfitContinuity(unittest.TestCase):
    """Scenes are generated independently, so clothing must be pinned in text."""

    def setUp(self):
        self.scene = product_video.Scene(setting="s", beats=["b"], dialogue="d")

    def test_continuity_clause_is_always_present(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("stay exactly the same in every scene", prompt)

    def test_explicit_outfit_is_pinned(self):
        prompt = product_video.build_scene_prompt(
            _product(), self.scene, outfit="a plain grey t-shirt"
        )
        self.assertIn("They wear a plain grey t-shirt.", prompt)

    def test_blank_outfit_adds_nothing(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene, outfit="  ")
        self.assertNotIn("They wear", prompt)

    def test_negative_prompt_bans_clothing_changes(self):
        prompt = product_video.build_scene_prompt(_product(), self.scene)
        self.assertIn("clothing changes", prompt)

    def test_continuity_survives_truncation(self):
        # It lives in the head, which is never cut to fit the 7000-char limit.
        huge = product_video.Scene(setting="x" * 3000, beats=["y" * 3000], dialogue="d")
        prompt = product_video.build_scene_prompt(
            _product(), huge, outfit="a denim jacket"
        )
        self.assertIn("They wear a denim jacket.", prompt)
        self.assertLessEqual(len(prompt), product_video.MAX_PROMPT_CHARS)

if __name__ == "__main__":
    unittest.main()
