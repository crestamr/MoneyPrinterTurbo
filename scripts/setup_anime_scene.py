"""Install the "MiniMax H3 - Anime Scene from reference" pack into ComfyUI.

Run by start-all.bat on every start, so it only ever *adds* what is missing:
workflows you have edited in the ComfyUI UI are never overwritten. Pass
--force to rebuild the workflows from the pack.

What it does:
  * copies the four reference images into ComfyUI/input/anime-scene/
  * writes ready-to-run workflows into the ComfyUI workflow browser, under
    "Anime Scene (MiniMax H3)", with images, prompts and models filled in
  * reports any model file the H3 video workflow needs but cannot find

The pack itself (resource/Minimax H3 - Anime Scene from reference/) is
third-party material and stays out of git; without it this is a no-op.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "resource" / "Minimax H3 - Anime Scene from reference"
FOLDER = "Anime Scene (MiniMax H3)"
INPUT_SUBDIR = "anime-scene"
H3_WORKFLOW = PACK / "Workflows" / "ref2vid-minimaxh3+turboLora.json"
QWEN_T2I_NAME = "Qwen-Image-2.1-GGUF-T2I.json"

# The H3 prompt says "Image 1 is the girl, Image 2 the store ..."; these are
# the LoadImage node ids wired to ref_image_0..3, in that order.
REFERENCES = [
    (137, "Girl wearing walkman.png"),
    (139, "Store front background.png"),
    (143, "white car.png"),
    (150, "skateboard.png"),
]
PROMPT_NODE = 138
SAVE_VIDEO_NODE = 92

# Qwen-Image's native 16:9 size, matching the pack's 16:9 reference images.
IMAGE_SIZE = (1664, 928)
NEGATIVE = ("photorealistic, 3d render, CGI, glossy digital painting, text, labels, "
            "watermark, logo, blurry, deformed, extra limbs, duplicated parts")

H3_MODELS = [
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]
MODEL_SOURCE = "https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main"

# (workflow file, Prompt.md section, output prefix) for the image generators.
IMAGE_WORKFLOWS = [
    ("1 - Qwen-Image girl character sheet.json", "Girl Character Sheet Prompt", "girl"),
    ("2 - Qwen-Image store background.json", "Background Scene Prompt", "background"),
    ("3 - Qwen-Image white car sheet.json", "Car prompt", "car"),
    ("4 - Qwen-Image skateboard.json", "Skateboard Prompt", "skateboard"),
]
VIDEO_WORKFLOW = "5 - H3 ref2video (girl, store, car, skateboard).json"
VIDEO_PROMPT = "Minimax H3 Reference to video Prompt"


def read_prompts(prompt_md):
    """Split Prompt.md into {bold heading: body}."""
    text = prompt_md.read_text(encoding="utf-8")
    parts = re.split(r"^\*\*(.+?):?\*\*\s*$", text, flags=re.M)
    sections = {}
    for title, body in zip(parts[1::2], parts[2::2]):
        # The file pads paragraphs with runs of blank lines; keep one.
        sections[title.strip().rstrip(":")] = re.sub(r"\n\s*\n+", "\n\n", body.strip())
    return sections


def set_widget(node, name, value):
    names = list(node["widgets_values_named"])
    node["widgets_values"][names.index(name)] = value
    node["widgets_values_named"][name] = value


def build_video_workflow(pack, prompt):
    wf = json.loads((pack / "Workflows" / H3_WORKFLOW.name).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    for node_id, name in REFERENCES:
        set_widget(nodes[node_id], "image", f"{INPUT_SUBDIR}/{name}")
    set_widget(nodes[PROMPT_NODE], "value", prompt)
    set_widget(nodes[SAVE_VIDEO_NODE], "filename_prefix", "video/AnimeScene_H3")
    return wf


def build_image_workflow(qwen_template, prompt, prefix):
    """Image generator on the local Qwen-Image 2.1 GGUF setup."""
    wf = json.loads(qwen_template.read_text(encoding="utf-8"))
    nodes = {n["type"]: n for n in wf["nodes"]}
    nodes["TextEncodeQwenImage21"]["widgets_values"][:2] = [prompt, NEGATIVE]
    nodes["EmptyLatentImage"]["widgets_values"][:2] = list(IMAGE_SIZE)
    nodes["SaveImage"]["widgets_values"][0] = f"{INPUT_SUBDIR}/{prefix}"
    return wf


def install(pack, comfy, force=False):
    """Install into `comfy`; returns (files added, model files missing)."""
    added = []
    image_dir = comfy / "input" / INPUT_SUBDIR
    image_dir.mkdir(parents=True, exist_ok=True)
    for _, name in REFERENCES:
        dest = image_dir / name
        if force or not dest.exists():
            shutil.copy2(pack / name, dest)
            added.append(dest)

    workflows = comfy / "user" / "default" / "workflows"
    target = workflows / FOLDER
    target.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(pack / "Prompt.md")

    builds = {VIDEO_WORKFLOW: lambda: build_video_workflow(pack, prompts[VIDEO_PROMPT])}
    qwen_template = workflows / QWEN_T2I_NAME
    if qwen_template.exists():
        for filename, section, prefix in IMAGE_WORKFLOWS:
            builds[filename] = (lambda s=section, p=prefix:
                                build_image_workflow(qwen_template, prompts[s], p))
    for filename, build in builds.items():
        dest = target / filename
        if force or not dest.exists():
            dest.write_text(json.dumps(build(), indent=2), encoding="utf-8")
            added.append(dest)

    missing = [m for m in H3_MODELS if not (comfy / "models" / m).exists()]
    return added, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comfy", default=os.environ.get("COMFYUI_DIR", r"D:\Developer\ComfyUI"))
    parser.add_argument("--force", action="store_true",
                        help="rebuild workflows and images even if they exist")
    args = parser.parse_args(argv)
    comfy = Path(args.comfy)

    if not PACK.is_dir():
        print(f"Anime Scene pack not found at {PACK} - skipping.")
        return 0
    if not (comfy / "main.py").exists():
        print(f"ComfyUI not found at {comfy} - skipping the Anime Scene pack.")
        return 0

    added, missing = install(PACK, comfy, force=args.force)
    for path in added:
        print(f"  added {path}")
    print(f"Anime Scene workflows: ComfyUI > Workflows > {FOLDER}")
    if missing:
        print("  Missing H3 models (workflow 5 will not run until downloaded from")
        print(f"  {MODEL_SOURCE}):")
        for model in missing:
            print(f"    models/{model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
