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

### Why the LLM is pinned to the CPU

The Modelfile sets `PARAMETER num_gpu 0`, which looks wasteful and is not.

The card has 16 GB and ComfyUI needs essentially all of it to generate an image.
Prompt expansion runs immediately *before* each image, so the two land on the GPU
at the same moment. Measured on this machine:

| Configuration | Result |
|---|---|
| LLM on GPU (11.1 GB VRAM, 25 tok/s) | **every image times out - 0 clips produced** |
| LLM on CPU (2.7 tok/s) | 2 clips in 474s, ~70s LLM + ~130s image per scene |

A fast LLM that produces no video is worth less than a slow one that does. If
the GPU is ever dedicated to the LLM instead, drop the `num_gpu 0` line and
expect roughly 25 tok/s.

For reference, the censored `qwen3.8:27b` from the Ollama library is still
installed at 17.7 GB. At Q4_K_M it does not fit in VRAM and measured **0.6
tok/s**, so it is unusable here; `ollama rm qwen3.8:27b` reclaims the space.

### Sizing rule for any replacement model

Anything at or under ~11 GB can run fully on the GPU; above that Ollama splits it
with system RAM and throughput collapses by roughly 40x. But GPU residency only
helps if ComfyUI is not also generating - see the table above.

Useful commands:

```bat
ollama list                     rem installed models
ollama ps                       rem what is loaded, and the CPU/GPU split
ollama run qwen3.8-unc:27b      rem chat with it directly
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
