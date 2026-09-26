"""scripts/setup_anime_scene.py runs on every start-all, so it must be additive."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import setup_anime_scene as setup  # noqa: E402

PROMPT_MD = """**Girl Character Sheet Prompt:**



girl sheet



**Background Scene Prompt:**

store at night

**Skateboard Prompt:**

skateboard

**Car prompt:**
white car



**Minimax H3 Reference to video Prompt:**



Image 1 is the girl.



Keep the camera locked.
"""


def _node(node_id, node_type, **named):
    return {"id": node_id, "type": node_type,
            "widgets_values": list(named.values()), "widgets_values_named": named}


def _make_pack(root):
    pack = root / "pack"
    (pack / "Workflows").mkdir(parents=True)
    (pack / "Prompt.md").write_text(PROMPT_MD, encoding="utf-8")
    for _, name in setup.REFERENCES:
        (pack / name).write_bytes(b"png")
    nodes = [_node(i, "LoadImage", image="x.png", upload="image") for i, _ in setup.REFERENCES]
    nodes.append(_node(setup.PROMPT_NODE, "PrimitiveStringMultiline", value=""))
    nodes.append(_node(setup.SAVE_VIDEO_NODE, "SaveVideo", filename_prefix="video/x"))
    (pack / "Workflows" / setup.H3_WORKFLOW.name).write_text(
        json.dumps({"nodes": nodes}), encoding="utf-8")
    return pack


def _make_comfy(root):
    comfy = root / "ComfyUI"
    workflows = comfy / "user" / "default" / "workflows"
    workflows.mkdir(parents=True)
    template = {"nodes": [
        {"id": 4, "type": "TextEncodeQwenImage21", "widgets_values": ["apple", "", 1024]},
        {"id": 5, "type": "EmptyLatentImage", "widgets_values": [1024, 1024, 1]},
        {"id": 8, "type": "SaveImage", "widgets_values": ["Qwen"]},
    ]}
    (workflows / setup.QWEN_T2I_NAME).write_text(json.dumps(template), encoding="utf-8")
    return comfy


class TestSetupAnimeScene(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.pack, self.comfy = _make_pack(root), _make_comfy(root)
        self.folder = self.comfy / "user" / "default" / "workflows" / setup.FOLDER

    def tearDown(self):
        self._tmp.cleanup()

    def test_prompt_sections_are_split_and_blank_runs_collapsed(self):
        prompts = setup.read_prompts(self.pack / "Prompt.md")
        self.assertEqual(prompts["Car prompt"], "white car")
        self.assertEqual(prompts[setup.VIDEO_PROMPT],
                         "Image 1 is the girl.\n\nKeep the camera locked.")

    def test_video_workflow_gets_images_in_prompt_order_and_the_prompt(self):
        setup.install(self.pack, self.comfy)
        wf = json.loads((self.folder / setup.VIDEO_WORKFLOW).read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in wf["nodes"]}
        self.assertEqual(nodes[137]["widgets_values"][0], "anime-scene/Girl wearing walkman.png")
        self.assertEqual(nodes[150]["widgets_values"][0], "anime-scene/skateboard.png")
        self.assertIn("Image 1 is the girl", nodes[setup.PROMPT_NODE]["widgets_values"][0])

    def test_image_workflows_use_the_pack_prompts_on_qwen_image(self):
        setup.install(self.pack, self.comfy)
        wf = json.loads((self.folder / "3 - Qwen-Image white car sheet.json")
                        .read_text(encoding="utf-8"))
        nodes = {n["type"]: n for n in wf["nodes"]}
        self.assertEqual(nodes["TextEncodeQwenImage21"]["widgets_values"][:2],
                         ["white car", setup.NEGATIVE])
        self.assertEqual(nodes["EmptyLatentImage"]["widgets_values"][:2], [1664, 928])
        self.assertEqual(nodes["SaveImage"]["widgets_values"][0], "anime-scene/car")

    def test_rerun_never_overwrites_a_workflow_edited_in_the_ui(self):
        setup.install(self.pack, self.comfy)
        edited = self.folder / setup.VIDEO_WORKFLOW
        edited.write_text('{"edited": true}', encoding="utf-8")

        added, _ = setup.install(self.pack, self.comfy)

        self.assertEqual(added, [])
        self.assertEqual(edited.read_text(encoding="utf-8"), '{"edited": true}')

    def test_force_rebuilds(self):
        setup.install(self.pack, self.comfy)
        edited = self.folder / setup.VIDEO_WORKFLOW
        edited.write_text('{"edited": true}', encoding="utf-8")

        setup.install(self.pack, self.comfy, force=True)

        self.assertIn("nodes", json.loads(edited.read_text(encoding="utf-8")))

    def test_reports_missing_h3_models(self):
        _, missing = setup.install(self.pack, self.comfy)
        self.assertEqual(missing, setup.H3_MODELS)

        for model in setup.H3_MODELS:
            path = self.comfy / "models" / model
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        _, missing = setup.install(self.pack, self.comfy)
        self.assertEqual(missing, [])

    def test_without_the_qwen_template_only_the_video_workflow_is_written(self):
        (self.comfy / "user" / "default" / "workflows" / setup.QWEN_T2I_NAME).unlink()
        setup.install(self.pack, self.comfy)
        self.assertEqual([p.name for p in self.folder.iterdir()], [setup.VIDEO_WORKFLOW])


if __name__ == "__main__":
    unittest.main()
