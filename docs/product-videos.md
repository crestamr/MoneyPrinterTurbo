# TikTok Shop product videos

Turn a product and its photos into a vertical UGC-style ad: a presenter
physically demonstrating the product, speaking to camera, lip-synced.

This is a **separate pipeline** from the main topic-to-video flow. That one goes
topic → script → search terms → stock footage → TTS → subtitles. This one
inverts it: the visuals come from your product photos, and the model speaks the
dialogue on camera rather than a TTS track being laid over silent footage.

## In the WebUI

Pick **TikTok Shop Product Video** under *Video Source -> AI Video*. The product
form appears in the Video Settings panel:

| Field | Notes |
|---|---|
| TikTok Shop product URL | Optional. *Fetch product details* fills the name and image |
| Product name | Say it the way a person would, not the listing title |
| Key features | Comma separated; the dialogue is written around these |
| Who it is for | Steers tone and setting |
| Product images | Your own photos beat a listing thumbnail |
| Presenter reference | Photo or video; keeps the same person across scenes |
| Scenes / Seconds per scene | Each scene is one paid generation |

Script, voice and subtitle settings do not apply to this source - the model
speaks on camera. Progress, logs and the finished video appear in the task
manager exactly as for any other source.

## Quick start

```bat
rem See the prompts. Costs nothing.
.venv\Scripts\python.exe make_product_video.py ^
    --url "https://shop.tiktok.com/jp/pdp/1731596597173191895" --dry-run

rem Real run with your own photos (better input than a listing thumbnail)
.venv\Scripts\python.exe make_product_video.py ^
    --name "Dowinx gaming chair" ^
    --image photos\hero.png --image photos\in-use.png ^
    --features "lumbar support,165 degree recline,linked armrests" ^
    --scenes 4
```

| Flag | What it does |
|---|---|
| `--url` | TikTok Shop product link; fills name and image automatically |
| `--name` | Overrides the listing title (recommended - see below) |
| `--image` | Product image, repeatable. Local path or URL |
| `--features` | Comma-separated; the LLM writes dialogue around these |
| `--audience` | Who it is for, e.g. "people who sit all day" |
| `--character` | Photo *or video* of the presenter, kept consistent across scenes |
| `--character-at` | Seconds into a `--character` video to take the frame from |
| `--scenes` | Scenes to generate. **Each one is a paid API call** |
| `--seconds` | Seconds per scene (MiniMax allows 4-15) |
| `--dry-run` | Build and print prompts, generate nothing, spend nothing |
| `--yes` | Skip the cost confirmation prompt |

## Cost

Every scene is one MiniMax generation billed to `minimax_api_key`. A 4-scene
video is 4 generations. The CLI prints the count and asks before starting, and
a failed scene can still bill, so `generations_spent` counts attempts, not
successes. **Always `--dry-run` first** - it exercises the whole chain
(listing fetch, image download, validation, LLM scene writing) for free.

## Always override the product name

Marketplace titles are keyword-stuffed for search:

> Dowinx ゲーミング座椅子 ゲーミングチェア 腰が痛くならない 回転座椅子 おしゃれ ゲーム
> パソコンチェア 連動アームレスト ランバーサポート 165°リクライニング チェア ハイバック
> グレー LS-6679

That string appears three times in every scene prompt and crowds out the parts
that shape the shot. `display_name` trims it to 48 characters automatically, but
a human-chosen `--name "Dowinx gaming chair"` is always better.

## Use your own photos

`--url` gives you the listing's hero image, which is enough to try the format.
Real product photography - several angles, in use, in context - produces
noticeably better video. Pass several with repeated `--image`; up to 9
reference images go into a single request.

Images must be JPG/PNG/WEBP/HEIC/HEIF, at most 30 MB, 256-5760 px on both
sides, aspect ratio between 0.4 and 2.5. They are validated locally before
upload, so a rejection names the limit instead of costing an API round trip.

## Pin the presenter

Without `--character` the model invents a different person for every scene, so
a four-scene ad reads as four unrelated clips. Pass a reference and the same
creator carries through:

```bat
--character photos/creator.jpg
--character resource/renoise/creator-clip.mp4 --character-at 2.0
```

A video is fine - a frame is taken from the body of the clip (35% in by
default, avoiding black openings and end cards). `reference_video` is
deliberately not used: it caps clips at 15 s, and combining video and image
references in one request is not something the API docs confirm. Two
`reference_image` entries - character first, then product - is the documented
shape.

**Check the extracted frame.** A clip from a previous ad will have *that*
product in shot, and it can bleed into the new video. Cropping to head and
shoulders is the reliable fix:

```bat
ffmpeg -y -ss 2.0 -i creator-clip.mp4 -frames:v 1 ^
    -vf "crop=1080:920:0:60" character-headshot.png
```

Then pass the crop with `--character`. The CLI prints a warning whenever it
extracts from a video, because this is easy to miss.

## How a prompt is built

Modelled on `resource/renoise/prompts.json`. Nine blocks, only three of which
vary per scene:

1. reference declaration - which image is the product, which the person
2. style + camera - vertical UGC, phone texture, one continuous take
3. audio - no music, so the spoken line survives
4. **action** - how the product is handled *(varies)*
5. **dialogue** - the spoken line *(varies)*
6. **timing beats** - what happens when *(varies)*
7. authenticity - not a commercial, no overacting
8. capture texture - real skin, phone exposure, no beauty filters
9. negative prompt - ~25 failure modes to avoid

Blocks 7-9 are most of the length and are what stop the output looking like an
advert. When a prompt approaches the API's 7000-character limit the *action*
is truncated, never the boilerplate.

## Audio

The presenter speaks on camera and the model lip-syncs, which is what sells the
format. The video's own audio is kept - MPT's TTS is not layered on top.
Subtitles can still be burned, because the dialogue is known before generation
rather than transcribed after.

## Troubleshooting

**"could not download image (HTTP 400)"** - ByteDance CDN URLs require a
`~tplv-<token>-...` template directive; the bare path and the listing's signed
query are both rejected. The parser rewrites thumbnails to the origin asset
automatically. If a URL still fails, save the image and pass `--image`.

**"no product details found on the page"** - TikTok renders listings with
JavaScript and changes its markup without notice. Share links (copied from the
app) carry title and image in an `og_info` parameter and need no scraping;
otherwise enter the details manually.

**"no scenes were produced"** - the LLM provider failed or returned
unparseable output. Check `llm_provider` in `config.toml`. With a Qwen3 model
on Ollama, make sure thinking is disabled or calls will time out.

**Scene failed but others succeeded** - the run continues and assembles what it
has. Partial output beats none; `failures` in the result names the scene.

## Configuration

`[minimax_product]` in `config.example.toml` holds the defaults (`clip_seconds`,
`scene_count`, `presenter`). The API key, base URL, model, resolution and
polling settings are shared with `[minimax_video]`.
