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

## Running

Run the full Twitch-clip-to-candidates flow:

```bash
python -m clipforge.render_clip --url "<twitch_clip_url>"
```

After reinstalling with `python -m pip install -e .`, the same full pipeline is available through the console script:

```bash
clipforge --url "<twitch_clip_url>"
```

The command:

- validates the Twitch clip URL
- downloads through the configured downloader backend, defaulting to yt-dlp
- downloads the source clip to `data/downloads/<clip_slug>/<backend>/`
- renders the `center_gameplay`, `facecam_focus`, and `hybrid` candidates to `data/renders/<clip_slug>/<backend>/`
- writes run metadata to `data/metadata/`
- prints the source, render, and metadata paths

Full pipeline render filenames are layout-only inside the scoped render
directory, for example `data/renders/<clip_slug>/ytdlp/hybrid.mp4`. Direct
`render` and `render-all` commands still write flat filenames based on the
source clip stem.

Use `--verbose` to show progress logs from Clipforge. For the `ytdlp` backend,
verbose mode logs when URL processing starts and when the download begins.

You can also run each pipeline step directly:

```bash
python -m clipforge.render_clip resolve-url --url "<twitch_clip_url>"
python -m clipforge.render_clip download --media-url "<direct_media_url>" --clip-id "<clip_id>"
python -m clipforge.render_clip render --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --layout center_gameplay
python -m clipforge.render_clip render-all --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4"
python -m clipforge.render_clip process --url "<twitch_clip_url>"
```

`resolve-url` uses Clipr because it resolves a direct media URL without
downloading. The full `process` flow and `clipforge --url` shortcut use the
configured downloader backend.

The same subcommands are available through the installed `clipforge` command:

```bash
clipforge --url "<twitch_clip_url>"
clipforge resolve-url --url "<twitch_clip_url>"
clipforge download --media-url "<direct_media_url>" --clip-id "<clip_id>"
clipforge render --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4" --layout center_gameplay
clipforge render-all --source "data/downloads/<clip_slug>/<backend>/<clip_slug>.mp4"
clipforge process --url "<twitch_clip_url>"
```

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
