# Running the local stack

Everyday start-up for this machine: the MoneyPrinterTurbo WebUI plus the local
ComfyUI server that backs the Qwen-Image 2.1 video source.

## One command

From the repo root:

```bat
start-all.bat
```

It starts ComfyUI in its own window (skipping that if ComfyUI already answers),
waits for it to finish loading, and then hands off to `webui.bat`.

| Address | What |
|---|---|
| http://127.0.0.1:8501 | MoneyPrinterTurbo WebUI |
| http://127.0.0.1:8188 | ComfyUI |
| http://127.0.0.1:11434 | Ollama |
| http://localhost:5175 | OpenShorts dashboard (Podman, own window) |

If port 8501 is taken, `webui.bat` moves to the next free port in 8502-8599 and
prints the address it picked.

Stop everything by closing the two console windows, or pressing Ctrl+C in each.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `COMFYUI_DIR` | `D:\Developer\ComfyUI` | Where ComfyUI is installed |
| `COMFYUI_HOST` | `127.0.0.1` | Host ComfyUI binds to |
| `COMFYUI_PORT` | `8188` | Port ComfyUI binds to |
| `SKIP_COMFYUI` | unset | Set to `1` to skip ComfyUI |
| `OLLAMA_PORT` | `11434` | Port Ollama binds to |
| `SKIP_OLLAMA` | unset | Set to `1` to skip Ollama |
| `OPENSHORTS_DIR` | `D:\Developer\openshorts` | Where OpenShorts and its `start-openshorts.bat` live |
| `SKIP_OPENSHORTS` | unset | Set to `1` to skip OpenShorts |
| `SKIP_ANIME_SCENE` | unset | Set to `1` to skip installing the Anime Scene workflows |

OpenShorts starts in its own minimised window because, right after the Podman
VM restarts, it can take about three minutes to answer. Its launcher also
repairs the VM's stale view of `D:` (the dashboard dies with `EIO` after a day
or so of uptime). Finished clips are kept 30 days and downloaded sources 3
days (`JOB_RETENTION_SECONDS` / `SOURCE_RETENTION_SECONDS` in OpenShorts'
`docker-compose.override.yml`); the upstream default deletes clips after 24 h.
| `MPT_WEBUI_HOST` | `127.0.0.1` | Host the WebUI binds to (read by `webui.bat`) |
| `MPT_WEBUI_PORT` | `8501` | Preferred WebUI port (read by `webui.bat`) |

Example - WebUI only, because you are using a stock-footage video source:

```bat
set SKIP_COMFYUI=1
start-all.bat
```

## Anime Scene workflows (MiniMax H3)

`start-all.bat` runs `scripts/setup_anime_scene.py`, which installs the pack in
`resource/Minimax H3 - Anime Scene from reference/` into ComfyUI: the four
reference images go to `input/anime-scene/`, and five workflows appear under
**Workflows > Anime Scene (MiniMax H3)** - four Qwen-Image 2.1 generators for
the reference images and the H3 reference-to-video workflow. It only adds
missing files, so workflows edited in the UI survive; `--force` rebuilds them.
The pack is third-party material and is not in git; without it the step is a
no-op.

The H3 models (about 44 GB, from `Comfy-Org/MiniMax-H3`) are larger than the
16 GB card, so ComfyUI streams them from system RAM: a 10-second clip at
0.6 MP takes 5-12 minutes per sampling step, depending on how much RAM is free.
Close other heavy apps first, or shorten the duration / lower the megapixels.

## Starting the pieces on their own

```bat
webui.bat                       rem WebUI only
omnivoice.bat                   rem OmniVoice TTS service, http://127.0.0.1:8890
```

```bat
cd /d D:\Developer\ComfyUI
venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188
```

The API server (`main.py`, port 8080 per `config.toml`) is separate from the
WebUI and only needed if you drive the app over HTTP instead of the browser.

## The local LLM (Ollama + Qwen3.8-27B abliterated)

Scripts, keywords and the expanded Qwen-Image prompts all come from a local
model, so no cloud LLM key is needed and nothing is billed per video.

`config.toml`:

```toml
llm_provider = "ollama"
ollama_base_url = ""              # empty = auto-detect http://localhost:11434
ollama_model_name = "qwen3.8-unc:27b"
```

The model is `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, UD-IQ3_XXS quant
(11.0 GB), imported from a local GGUF rather than pulled from the Ollama
library. Weights live on `D:` via the persistent user variable
`OLLAMA_MODELS=D:\Developer\ollama-models`.

To rebuild it after editing the Modelfile:

```bat
ollama create qwen3.8-unc:27b -f D:\Developer\ollama-models\gguf\Modelfile.qwen38-unc
```

Note that `ollama pull hf.co/<repo>:<quant>` fails on Ollama 0.34.3 with
`realm host "huggingface.co" does not match original host "hf.co"`. Download the
`.gguf` with curl and `ollama create` from a Modelfile instead.

### Sharing the RTX by taking turns

Ollama and ComfyUI both want the whole 16 GB card, and prompt expansion runs
immediately before each image, so they collide. The answer is not to split the
card or to move one of them off it - it is to make both release it when idle:

- `OLLAMA_KEEP_ALIVE=0` unloads the model as soon as a request finishes
- ComfyUI runs with `--disable-smart-memory`, offloading to RAM instead of
  holding VRAM between generations

Reloading is cheap enough to make this work: about 3.6s from page cache, against
53 tok/s on the GPU. Measured on the same two-clip pipeline:

| LLM placement | Speed | Two-clip pipeline |
|---|---|---|
| RTX, both sides holding VRAM | 25 tok/s | **0 clips - every image timed out** |
| CPU only | 2.7 tok/s | 474s |
| Radeon 890M via Vulkan | 5.4 tok/s | 266s |
| **RTX, both sides releasing VRAM** | **53 tok/s** | **160-207s** |

Both halves are required. If either process squats on VRAM the other starves,
and image generation times out completely rather than merely slowing down.

### The integrated GPU is a fallback, not the answer

The Radeon 890M works and is documented here because it was the best option
before the take-turns approach was found. Ollama hides integrated GPUs by
default, logging `dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1`.
Enabling that is not sufficient - Ollama still prefers the NVIDIA card, and
hiding it from CUDA alone just makes Vulkan pick it up instead. All three are
needed, and `CUDA_VISIBLE_DEVICES` must never be set as a user-level variable
because it would blind ComfyUI too:

```bat
set "CUDA_VISIBLE_DEVICES=-1"      rem hide NVIDIA from the CUDA backend
set "GGML_VK_VISIBLE_DEVICES=1"    rem pick AMD from the Vulkan list
set "OLLAMA_IGPU_ENABLE=1"         rem allow integrated GPUs at all
```

That path costs about 10x the tokens-per-second, so use it only if the RTX is
needed exclusively for something else.

### Thinking is disabled

`app/services/llm.py` sends `extra_body={"think": False}` for the Ollama
provider. This model's chat template defaults to `xhigh` reasoning effort, its
most verbose setting, and `_normalize_text_response` discards the whole `<think>`
trace anyway - so it was pure latency. Script generation went 83s to 55s and
keyword generation 306s to 152s. Ollama ignores the flag for models without the
thinking capability, so it is safe for any Ollama model.

### Watch for orphaned runners

Killing `ollama.exe` does **not** kill its `llama-server.exe` child processes,
and those keep holding GPU memory. Five orphans had accumulated during testing
and were squatting on 11.5 GB of the RTX. If ComfyUI starts timing out, check:

```bat
tasklist | findstr llama-server
taskkill /F /IM llama-server.exe
```

## When does ComfyUI actually matter?

Only for `video_source = "qwen_image"`, which is what `config.toml` is currently
set to. With it, every scene is generated locally on the GPU: roughly 2 minutes
per 2048px image, or about a minute at 1088x1920 portrait.

If ComfyUI is not running, the WebUI stops before generating and shows
"Could Not Connect to ComfyUI". Either start it, or switch Video Source in the
UI to a stock-footage provider such as Pexels.

## Things that are easy to forget

- **If the LLM is unreachable, video still gets made.** Image prompts fall back
  to the raw search term, logged as a warning rather than an error. So a quiet
  drop in visual quality is the symptom of a stopped Ollama, not a crash.
- **First ComfyUI start after a reboot is slow.** It loads a 7.6 GB GGUF diffusion
  model and a 17.5 GB text encoder, and the encoder is larger than the 16 GB of
  VRAM, so part of it is served from system RAM.
- **OmniVoice is optional and lives in its own environment.** See
  `services/omnivoice_server/README.md`.
