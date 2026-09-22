# OmniVoice TTS service

A small HTTP wrapper that exposes [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
over the OpenAI `/v1/audio/speech` contract, so MoneyPrinterTurbo can talk to it
with the same client code it already uses for Kokoro and Chatterbox.

## What and why

The default TTS provider is Edge TTS. It is free, but it is also a public
endpoint that intermittently times out mid-render, which kills a task that has
already spent minutes on video assembly. OmniVoice trades that for a local
dependency you control:

- **Runs entirely on your machine.** No API key, no network call, no rate limit,
  nothing to time out.
- **Apache-2.0 for both the code and the weights**, so there is no license
  question about the audio it produces.
- **Zero-shot voice cloning.** You do not train anything. Drop a few seconds of
  reference audio in a folder and that voice becomes selectable in the WebUI.

The cost is that it needs a GPU and a PyTorch install, which is why it lives in
its own service with its own dependencies instead of inside the main app.

## Install

Install into a **separate environment**. The main app's venv deliberately has no
`torch` — adding a multi-gigabyte CUDA stack to it would make every
`uv sync` slow and would pin the app to a CUDA build it does not otherwise need.

On an RTX 50-series (Blackwell, `sm_120`) card such as the 5070 Ti, you must use
the **cu128** wheels. The default PyPI `torch` wheel does not include `sm_120`
kernels and will fail at runtime with a "no kernel image is available" error.

```bash
# From the repository root.
python -m venv .venv-omnivoice

# Windows
.venv-omnivoice\Scripts\activate
# macOS / Linux
source .venv-omnivoice/bin/activate

# Install PyTorch FIRST, from the CUDA 12.8 index. Doing this before the
# requirements file stops pip from pulling the CPU/default-CUDA wheel.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Then the rest of the service dependencies.
pip install -r services/omnivoice_server/requirements.txt
```

`requirements.txt` pins `torch>=2.8.0`; OmniVoice itself is the `omnivoice`
package on PyPI, and is installed by the file above. The other dependencies are
`fastapi`, `uvicorn[standard]`, `soundfile`, and `numpy`.

Verify the install picked up your GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The service also shells out to **ffmpeg** to encode its output as MP3. The main
app already requires ffmpeg, so it is normally on `PATH` already. If yours is
not, set the `FFMPEG_BINARY` environment variable to its full path.

Model weights are not part of the install. They are downloaded from Hugging Face
on the first synthesis and cached in your Hugging Face cache directory. If that
download is blocked for you, set `HF_ENDPOINT` to a mirror before starting the
service — both launch scripts have a commented line ready for this.

## Add voices

A voice is just a reference clip in `storage/voices/`:

```
storage/voices/
├── narrator.wav          -> voice ID "narrator"
├── narrator.txt          -> optional transcript of narrator.wav
└── calm_female.mp3       -> voice ID "calm_female"
```

- Supported extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`.
- **The filename stem is the voice ID.** Keep it simple — it is what you will
  pick in the WebUI dropdown.
- **3 to 10 seconds** of clean speech is the useful range. Use a clip with no
  music, no background noise, and no second speaker; the model clones whatever
  it hears, including room echo.
- The optional **`<stem>.txt` sidecar** holds the transcript of that clip. When
  it is present, OmniVoice uses it directly. When it is absent, OmniVoice must
  transcribe the clip with Whisper first, which is slow and is the single
  biggest cause of a long first synthesis. Writing one line of text per clip is
  worth it.

The directory is scanned on every request, so you can add a clip without
restarting the service — just reload the WebUI page to refresh the dropdown.

## Run

From the repository root:

```bash
# Windows
omnivoice.bat

# macOS / Linux / Git Bash
./omnivoice.sh
```

Both scripts set `PYTHONPATH` to the repository root and run
`python -m services.omnivoice_server`. Because the service needs its own
environment, point them at it with the `OMNIVOICE_PYTHON` environment variable:

```bash
# Windows (PowerShell)
$env:OMNIVOICE_PYTHON = "$PWD\.venv-omnivoice\Scripts\python.exe"; .\omnivoice.bat

# macOS / Linux
OMNIVOICE_PYTHON="$PWD/.venv-omnivoice/bin/python" ./omnivoice.sh
```

Without that variable the scripts fall back to the repository's `.venv` and then
to `python` on `PATH`.

Any arguments you pass to the script are forwarded to the service. The defaults
below apply unless `config.toml` overrides them — see
[Point the app at it](#point-the-app-at-it):

| Flag | Default | Notes |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8890` | Must match `base_url` in `config.toml`. |
| `--voices-dir` | `storage/voices` | Created on startup if missing. |
| `--model-id` | `k2-fsa/OmniVoice` | Hugging Face repo to load weights from. |
| `--device` | `cuda:0` | Use `cpu` to run without a GPU — very slow. |
| `--num-step` | `32` | Diffusion steps. `16` is roughly twice as fast with some quality loss. |
| `--idle-unload-seconds` | `600` | Unload the model after this long idle. `0` disables it. |
| `--api-key` | `$OMNIVOICE_API_KEY` or empty | When empty, auth is disabled. |

Example:

```bash
./omnivoice.sh --num-step 16 --idle-unload-seconds 300
```

Endpoints, once it is up:

- `GET  /health` — status, voice count, and whether the model is currently
  loaded. Never requires the API key; this is what the WebUI guard probes.
- `GET  /v1/audio/voices` — the list of discovered voice IDs.
- `POST /v1/audio/speech` — OpenAI-compatible synthesis, returns MP3.
- `POST /unload` — drop the model and release VRAM now.
- `GET  /docs` — FastAPI's interactive docs.

If you set `--api-key`, every endpoint except `/health` requires an
`Authorization: Bearer <key>` header.

Quick check:

```bash
curl http://127.0.0.1:8890/health
curl http://127.0.0.1:8890/v1/audio/voices
```

## Point the app at it

In `config.toml`:

```toml
[omnivoice]
base_url = "http://127.0.0.1:8890/v1"
api_key = ""          # must match --api-key if you set one
model_id = "omnivoice"
```

Then in the WebUI, under **Audio Settings**, set the TTS server to
**OmniVoice TTS** and pick one of your voices. The dropdown is populated by
calling `/v1/audio/voices`, so the service must be running when you open that
page.

The `[omnivoice]` section also carries `voices_dir`, `num_step`, and
`idle_unload_seconds`. Those configure **this service**, not the client: on
startup it reads `config.toml` from the repository root and uses them as its
defaults, so setting `num_step = 16` there is enough — no launch flag needed.

Resolution order, highest first:

1. a flag you passed on the command line,
2. `$OMNIVOICE_API_KEY` (for `--api-key` only),
3. the `[omnivoice]` value in `config.toml`,
4. the built-in default from the table above.

`api_key` is read by both sides, which is what keeps them in agreement. An empty
value means "not set", so `voices_dir = ""` falls back to `storage/voices`.
`base_url` and `model_id` are client-only — the service never reads them, and it
deliberately ignores `model_id` because there it means the name the client puts
in the request payload (`omnivoice`), not the Hugging Face weights repo. A
missing or malformed `config.toml` is ignored and the service still starts.

## Subtitles

Set this in `config.toml`:

```toml
subtitle_provider = "whisper"
```

The OpenAI speech contract returns audio only — there are no word-level
timestamps in the response. Without them the app falls back to estimating
subtitle timing from the full text, which drifts on longer scripts. This is the
same limitation Kokoro and Chatterbox have, and the same fix applies: let
Whisper transcribe the rendered audio and derive real timings from it.

## VRAM

Expect roughly **2-4 GB** of VRAM while the model is loaded, at `float16`.

That matters on a machine that also runs ComfyUI, since both want the same GPU.
Two ways to hand it back:

```bash
# Immediately, when auth is disabled:
curl -X POST http://127.0.0.1:8890/unload

# With an API key set:
curl -X POST http://127.0.0.1:8890/unload -H "Authorization: Bearer YOUR_KEY"
```

Or just leave it alone: `--idle-unload-seconds` (default `600`) unloads the
model automatically after ten idle minutes. A background thread checks every 30
seconds, so the actual unload lands within about half a minute of the deadline.
The model reloads by itself on the next request. A synthesis that runs longer
than the timeout will not be unloaded out from under itself.

## Performance expectation

**The first synthesis with a new voice is slow.** It pays for, in order:

1. Downloading the weights, the very first time only.
2. Loading the model onto the GPU — tens of seconds.
3. Building the clone prompt for that reference clip, which runs Whisper over
   the clip if there is no `<stem>.txt` sidecar.

**Subsequent syntheses are fast.** The model stays warm until the idle timeout,
and the clone prompt is cached on disk next to its clip as
`<stem>.<size>-<mtime>.pt`. That cache survives restarts, so step 3 is paid once
per voice, not once per session. The key includes the clip's size and mtime, so
editing or replacing a clip invalidates it and rebuilds automatically.

Practical consequence: after adding a new voice, synthesize one short test
sentence with it before you start a real video task. That way the slow path
happens while you are watching instead of in the middle of a render.

To delete a cache entry, just remove the matching `.pt` file from
`storage/voices/`; it will be rebuilt on next use.

## Troubleshooting

**"The local OmniVoice service is not responding. Start it with omnivoice.bat
(or omnivoice.sh), then try again."**

The WebUI guard could not reach `GET /health`. Check that the service is
actually running, that its port matches `base_url` in `config.toml` (default
`8890`), and that `base_url` ends in `/v1`. If the service window exited on
startup, read its output — a missing `torch` or a CUDA mismatch shows up there.

**No voices in the dropdown**

`storage/voices/` is empty, or holds nothing with a supported extension. Confirm
with `curl http://127.0.0.1:8890/v1/audio/voices` — an empty list means the
service sees no clips. Check you are looking at the directory the service is
actually using: it prints `reference clips: <path>` on startup, resolved from
`--voices-dir`, then `[omnivoice] voices_dir` in `config.toml`, then the
default.

**CUDA out of memory**

Almost always ComfyUI (or another local model) already holding the GPU. Unload
one of them: `curl -X POST http://127.0.0.1:8890/unload` for this service, or
ComfyUI's own unload/free-memory control for that one. Synthesis returns HTTP
500 with the underlying error in the response detail, so the actual exception is
visible rather than a bare "Internal Server Error". Lowering `--num-step` reduces
compute time but not peak VRAM; if the card is genuinely too small, run this
service on `--device cpu` and accept it being much slower.

**"no kernel image is available for execution on the device"**

The installed `torch` has no kernels for your GPU architecture. On RTX 50-series
cards, reinstall from the cu128 index (see [Install](#install)).

**ffmpeg errors on synthesis**

The service encodes MP3 by shelling out to `ffmpeg`. If it is not on `PATH`, set
`FFMPEG_BINARY` to its full path before starting the service.

**Numbers and abbreviations are read out oddly (text normalization)**

OmniVoice can spell out numbers, currency and abbreviations ("$12" → "twelve
dollars") before synthesis, but that step needs `WeTextProcessing`, which is
built on `pynini`. `pynini` publishes no Windows wheel, and its source build
passes GCC-only flags (`-Wno-register`, `-funsigned-char`) that MSVC rejects, so
**it cannot be installed on Windows**.

The service detects this and simply skips normalization rather than failing the
request — without the check, every synthesis returns HTTP 500 with
`ImportError: Text normalization (normalize_text=True) requires
WeTextProcessing`. Nothing to do on Windows; write numbers as words in the
script if their pronunciation matters.

On Linux (or WSL) you can install it and normalization switches on
automatically, no config change needed:

```bash
pip install WeTextProcessing
```
