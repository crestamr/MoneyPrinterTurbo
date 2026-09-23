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
| `MPT_WEBUI_HOST` | `127.0.0.1` | Host the WebUI binds to (read by `webui.bat`) |
| `MPT_WEBUI_PORT` | `8501` | Preferred WebUI port (read by `webui.bat`) |

Example - WebUI only, because you are using a stock-footage video source:

```bat
set SKIP_COMFYUI=1
start-all.bat
```

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

### Two GPUs, one job each

This machine has an RTX 5070 Ti (16 GB) **and** a Radeon 890M integrated GPU with
32 GB carved out of system RAM. Ollama is pinned to the 890M so the RTX belongs
entirely to ComfyUI.

This matters more than it sounds. The card has 16 GB and ComfyUI needs
essentially all of it to generate an image, while prompt expansion runs
immediately *before* each image. Measured, same two-clip pipeline each time:

| LLM placement | Speed | Pipeline result |
|---|---|---|
| RTX 5070 Ti (11.1 GB VRAM) | 25 tok/s | **0 clips - every image timed out** |
| CPU only | 2.7 tok/s | 2 clips in 474s |
| Radeon 890M via Vulkan | **5.2 tok/s** | **2 clips in 266s** |

Ollama hides integrated GPUs by default, logging
`dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1`. Enabling that is
not enough on its own: Ollama still prefers the NVIDIA card, and hiding it from
CUDA alone just makes Vulkan pick it up instead. `start-all.bat` launches the
Ollama server with all three of these set, then clears the device variables so
ComfyUI - started later by the same script - can still see CUDA:

```bat
set "CUDA_VISIBLE_DEVICES=-1"      rem hide NVIDIA from the CUDA backend
set "GGML_VK_VISIBLE_DEVICES=1"    rem pick the AMD device from the Vulkan list
set "OLLAMA_IGPU_ENABLE=1"         rem allow integrated GPUs at all
```

Never set `CUDA_VISIBLE_DEVICES` as a user-level environment variable. It would
blind ComfyUI too.

The Vulkan device index is machine-specific: index 0 is the NVIDIA card here.
Confirm the choice in the server log, which should read
`using device Vulkan0 (AMD Radeon(TM) 890M Graphics) ... 77452 MiB free`.

ROCm would likely beat Vulkan, but the server logs
`AMD driver is too old. Update your AMD driver to enable GPU inference.`
Updating the Radeon driver is the obvious next experiment.

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
