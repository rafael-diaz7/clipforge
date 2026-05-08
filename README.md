# clipforge

Automating the boring parts of clipping so I can try to turn this into a minimal effort task.

The goal of this project is to take a Twitch clip and turn it into multiple short-form vertical edits that are ready for YouTube Shorts, TikTok, etc. Instead of manually editing every clip, this pipeline generates a few good candidates that I can quickly review and tweak.

This is not meant to fully replace editing. It is meant to speed it up.


## What this does (MVP)

Given a Twitch clip URL:

- Get a downloadable video (via Clipr or similar)
- Download the clip locally
- Generate multiple vertical edits:
  - centered gameplay crop
  - facecam-focused crop
  - hybrid layout (facecam + gameplay section)
- Save all outputs locally
- Save layout metadata so edits can be tweaked later without starting over

This is a human-in-the-loop workflow: clipforge creates candidates, then I review, tweak, and post manually.


## Project structure

```
clipforge/
  src/clipforge/        # main code
  data/
    downloads/          # raw clips
    renders/            # final video outputs
    metadata/           # layout + edit configs (json)
      captions/         # optional timed caption metadata
  examples/
    layouts/            # example layout templates
  README.md
```

## How this is meant to be used

1. Input a Twitch clip URL
2. Script downloads the clip
3. Script generates a few vertical edit candidates
4. I review them
5. If needed, I tweak layout values or re-render
6. Post manually

## Setup

Requirements:

- Python 3.11+
- FFmpeg installed and available in PATH
- yt-dlp is installed with Clipforge as a Python dependency
- Twitch developer app credentials for clip discovery:
  - `TWITCH_CLIENT_ID`
  - `TWITCH_CLIENT_SECRET`
- OpenAI API credentials for optional caption generation:
  - `OPENAI_API_KEY`
  - `OPENAI_TRANSCRIPTION_MODEL` defaults to `whisper-1`
- A downloader backend:
  - Clipr: `CLIPR_API_KEY` set and `CLIPFORGE_DOWNLOADER=clipr`
  - yt-dlp: `CLIPFORGE_DOWNLOADER=ytdlp` or unset
- `CLIPFORGE_DOWNLOADER` unset uses the default yt-dlp downloader

### FFmpeg

Check whether FFmpeg is already available:

```bash
ffmpeg -version
```

If that command fails, install FFmpeg first:

- Windows: install FFmpeg, then make sure the `ffmpeg` executable is on PATH before opening Git Bash.
- Ubuntu/Linux: use your distro package manager, for example `sudo apt install ffmpeg`.
- macOS: install with Homebrew using `brew install ffmpeg`.

### Python Environment

Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e .
```

Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Clipr API Key

Copy `.env.example` to `.env`. If you use Clipr, replace the placeholder with
your real key:

```bash
cp .env.example .env
```

`.env`:

```text
CLIPR_API_KEY=your_key_here
CLIPFORGE_DOWNLOADER=clipr
```

`CLIPFORGE_DOWNLOADER` selects the Twitch clip downloader backend. The only
supported backends are `clipr` and `ytdlp`. `ytdlp` remains the default when the
variable is unset, even if `CLIPR_API_KEY` is present.

### yt-dlp

Clipforge installs `yt-dlp` as a Python dependency. To download Twitch clips
directly without Clipr, leave `CLIPFORGE_DOWNLOADER` unset or select the backend:

`.env`:

```text
CLIPFORGE_DOWNLOADER=ytdlp
```

When `ytdlp` is selected, `CLIPR_API_KEY` is not required. Clipforge runs
`yt-dlp` from the active Python environment.

### Twitch Clip Discovery

Clip discovery uses the Twitch Helix API with app credentials. Add these values
to `.env`:

```text
TWITCH_CLIENT_ID=your_twitch_client_id_here
TWITCH_CLIENT_SECRET=your_twitch_client_secret_here
```

List clips for a channel without downloading or rendering:

```bash
clipforge clips --channel "<channel_login>" --limit 10
```

Without explicit dates, `clips` searches the last 7 days in UTC.

You can narrow discovery with UTC ISO-8601 date filters:

```bash
clipforge clips \
  --channel "<channel_login>" \
  --limit 20 \
  --started-at "2026-05-01T00:00:00Z" \
  --ended-at "2026-05-06T00:00:00Z"
```

The command prints tab-separated `created_at`, `view_count`, `duration`, `url`,
and `title` fields. Discovery is read-only and does not trigger downloads or
renders.

Pass `--format json` to write a queue-friendly discovery export instead of the
tab-separated list:

```bash
clipforge clips --channel "<channel_login>" --limit 20 --format json
```

By default, JSON exports are written to
`data/metadata/discovered_clips/<channel>/<date>-<channel>.json`. Use `--output`
to choose a specific path:

```bash
clipforge clips \
  --channel "<channel_login>" \
  --limit 20 \
  --format json \
  --output "data/metadata/discovered_clips/review_queue.json"
```

## Running

Run the full Twitch-clip-to-candidates flow:

```bash
python -m clipforge.pipeline.cli --url "<twitch_clip_url>"
```

After reinstalling with `python -m pip install -e .`, the same full pipeline is available through the console script:

```bash
clipforge --url "<twitch_clip_url>"
```

The command:

- validates the Twitch clip URL
- downloads through the configured downloader backend, defaulting to yt-dlp
- downloads the source clip to `data/downloads/<clip_slug>/<backend>/`
- optionally generates caption metadata before rendering
- burns generated captions into the rendered candidates when caption generation is enabled
- renders the `center_gameplay`, `facecam_focus`, and `hybrid` candidates to `data/renders/<clip_slug>/<backend>/`
- writes run metadata to `data/metadata/`
- prints the source, render, and metadata paths

Caption metadata uses the `clipforge.caption_metadata` JSON schema with a
`clip_id` and ordered `segments`, where each segment has `start_time`,
`end_time`, and `text`. Caption files are saved deterministically under
`data/metadata/captions/<clip_id>.json`. Run metadata may reference a caption
metadata file with `caption_metadata_path`; when that field is absent, the clip
has no caption metadata. A caption file with an empty `segments` list is valid
and means captions were intentionally saved as empty.
Before upload, clipforge extracts temporary speech-optimized MP3 audio with
FFmpeg instead of sending the full MP4 to OpenAI.
Caption rendering uses a conservative vertical-safe style by default: smaller
two-line captions, safe horizontal margins, and capped display time so short
captions do not linger through long pauses in a transcription segment.

Set `CLIPFORGE_CAPTION_FONT_FILE` to a local `.ttf` or `.otf` path to change the
burned-in caption font:

```text
CLIPFORGE_CAPTION_FONT_FILE=C:/Windows/Fonts/arial.ttf
```

Use `--captions <caption_metadata.json>` with `render` or `render-all` to burn
existing caption metadata into local renders:

```bash
clipforge render-all \
  --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" \
  --clip-id "<clip_id>" \
  --captions "data/metadata/captions/<clip_id>.json"
```

Full pipeline render filenames are layout-only inside the scoped render
directory, for example `data/renders/<clip_slug>/ytdlp/hybrid.mp4`. Direct
`render` and `render-all` commands still write flat filenames based on the
source clip stem.

Use `--verbose` to show progress logs from Clipforge. For the `ytdlp` backend,
verbose mode logs when URL processing starts and when the download begins.

You can also run each pipeline step directly:

```bash
python -m clipforge.pipeline.cli resolve-url --url "<twitch_clip_url>"
python -m clipforge.pipeline.cli download --media-url "<direct_media_url>" --clip-id "<clip_id>"
python -m clipforge.pipeline.cli captions --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --clip-id "<clip_id>"
python -m clipforge.pipeline.cli render --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --layout center_gameplay --captions "data/metadata/captions/<clip_id>.json"
python -m clipforge.pipeline.cli render-all --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --captions "data/metadata/captions/<clip_id>.json"
python -m clipforge.pipeline.cli process --url "<twitch_clip_url>"
```

`resolve-url` uses Clipr because it resolves a direct media URL without
downloading. The full `process` flow and `clipforge --url` shortcut use the
configured downloader backend.

The same subcommands are available through the installed `clipforge` command:

```bash
clipforge --url "<twitch_clip_url>"
clipforge resolve-url --url "<twitch_clip_url>"
clipforge download --media-url "<direct_media_url>" --clip-id "<clip_id>"
clipforge captions --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --clip-id "<clip_id>"
clipforge render --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --layout center_gameplay --captions "data/metadata/captions/<clip_id>.json"
clipforge render-all --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --captions "data/metadata/captions/<clip_id>.json"
clipforge process --url "<twitch_clip_url>"
```

For development, the installed console script points at
`clipforge.pipeline.cli:main`.

## Testing

Run the test suite from the repository root:

```bash
pytest .\tests\
```

The pytest config adds the repo root to Python's import path so shared test
helpers such as `tests.constants` resolve consistently across `pytest` and
`python -m pytest`.

## Design notes

Layouts are defined using normalized coordinates (0 to 1). This makes it easy to tweak crops without re-editing from scratch.

Each render is generated from a layout config, not hardcoded logic.

The idea is to:
- generate a few good options automatically
- make small adjustments quickly
- avoid full manual edits every time


## Future ideas

- automatic clip selection (Twitch API)
- captions using Whisper
- better facecam detection
- dynamic crops based on audio or transcript
- auto-generated titles and descriptions
- upload integration


## Why I'm building this

Clipping is mostly repetitive work. The creative part is deciding what is funny or worth posting.

Everything else should be as fast as possible.
