# Clipforge

Clipforge is a local, CLI-driven workflow for discovering Twitch clips, ranking
them for review, and rendering short-form vertical candidates.

It is built around a stateful loop:

```text
discover -> persist -> rerank -> review -> process -> reprocess
```

Clipforge does not try to replace editorial judgment. It automates the repetitive
download, caption, layout, and render work so a human can decide what is worth
posting.

## Features

- Discover recent Twitch clips for a channel through the Twitch Helix API.
- Persist discovered clips in local SQLite state at `data/state/clipforge.sqlite`.
- Rank clips using Twitch metadata so review starts with the strongest candidates.
- Review pending clips before spending time on downloads and renders.
- Process the highest-ranked pending clips or a specific saved clip.
- Reprocess an already-rendered clip intentionally with `--force`.
- Download Twitch clips with `yt-dlp` by default, with Clipr available as an optional backend.
- Render vertical candidates from reusable layout JSON files.
- Automatically apply configured streamer PNG watermarks during renders.
- Optionally generate timed caption metadata with the OpenAI transcription API.
- Burn caption metadata into renders with FFmpeg `drawtext` or generated `.ass` subtitle files.

## Workflow Overview

The primary workflow is stateful:

1. Discover clips for a Twitch channel.
2. Clipforge writes or refreshes those clips in SQLite.
3. Ranking scores are stored with each clip.
4. You review pending clips from the local state table.
5. You process selected clips into vertical render candidates.
6. If an edit needs another pass, you reprocess that clip explicitly.

Processing remains human-in-the-loop. Clipforge creates candidates; you still
review, tweak layouts or caption settings when needed, and post manually.
Clipforge also tracks clip lifecycle state in SQLite so processed clips are not
re-rendered accidentally.

## Installation

Requirements:

- Python 3.11+
- FFmpeg available on `PATH`
- Twitch app credentials for clip discovery
- OpenAI API credentials only if generating captions
- Clipr API key only if using the Clipr downloader backend

Check FFmpeg:

```powershell
ffmpeg -version
```

Create a virtual environment and install Clipforge:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For a non-development install, use `python -m pip install -e .`.

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

## Environment Variables

Clipforge loads `.env` from the project root.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `TWITCH_CLIENT_ID` | For `clips` discovery | none | Twitch app client ID. |
| `TWITCH_CLIENT_SECRET` | For `clips` discovery | none | Twitch app client secret. |
| `CLIPFORGE_DOWNLOADER` | No | `ytdlp` | Downloader backend. Supported values: `ytdlp`, `clipr`. |
| `CLIPR_API_KEY` | Only when `CLIPFORGE_DOWNLOADER=clipr` | none | RapidAPI Clipr key. |
| `OPENAI_API_KEY` | Only when generating captions | none | OpenAI API key for transcription. |
| `OPENAI_TRANSCRIPTION_MODEL` | No | `whisper-1` | OpenAI transcription model used for captions. |
| `CLIPFORGE_GENERATE_CAPTIONS` | No | `false` | Enables caption generation during full processing. |
| `CLIPFORGE_CAPTION_FONT_FILE` | No | none | Optional local `.ttf` or `.otf` font path for burned-in captions. |
| `CLIPFORGE_CAPTION_RENDERER` | No | `drawtext` | Caption renderer. Supported values: `drawtext`, `ass`. |
| `CLIPFORGE_ASS_TEMP_DIR` | No | `data/metadata/ass` | Directory for generated `.ass` files when using the `ass` renderer. |
| `CLIPFORGE_CAPTION_FONT_FALLBACKS` | No | `Arial` | Comma-separated fallback font names for generated ASS styles. |
| `{CHANNEL_NAME}_WATERMARK` | No | none | Optional PNG watermark path for one streamer, using an uppercase env-safe channel key. |

`yt-dlp` is installed as a Python dependency and is the default downloader even
if `CLIPR_API_KEY` is present. Set `CLIPFORGE_DOWNLOADER=clipr` only when you
want Clipforge to use Clipr.

## Streamer Watermarks

Clipforge can automatically apply a PNG watermark for a streamer during normal
rendering. Configure one environment variable per channel:

```text
OHNEPIXEL_WATERMARK=C:/watermarks/ohnepixel.png
DOUBLELIFT_WATERMARK=assets/watermarks/doublelift.png
```

The key is the channel name normalized to an uppercase env-safe value plus
`_WATERMARK`. For example, `ohnepixel` becomes `OHNEPIXEL_WATERMARK`,
`JasonTheWeen` becomes `JASONTHEWEEN_WATERMARK`, and `doublelift` becomes
`DOUBLELIFT_WATERMARK`.

When a saved clip is rendered for that streamer, Clipforge looks up the matching
env var and applies the PNG automatically. Review and export workflows inherit
this through the normal processing pipeline, so no watermark CLI flag is needed.
If no matching env var is set, renders behave as before.

Watermarks are placed at the bottom center with a 32px bottom margin. The
PNG aspect ratio is preserved, the rendered width is capped at roughly 34% of
the output width, and smaller PNGs are not upscaled.

## Quick Start

Discover and persist clips for a channel:

```powershell
clipforge clips --channel "<channel_login>" --limit 20
```

Review the saved pending list:

```powershell
clipforge clips pending --channel "<channel_login>"
```

Discover, process, and choose one final render per top clip:

```powershell
clipforge clips review --streamer "<channel_login>" --count 3
```

Process the top ranked pending clip:

```powershell
clipforge clips process --top 1
```

Reprocess a rendered clip after changing layout or caption settings:

```powershell
clipforge clips process --clip-id "<clip_id>" --force
```

Reset one clip back to the pending/discovered queue:

```powershell
clipforge clips reset --clip-id "<clip_id>"
```

Outputs are written under `data/downloads/`, `data/renders/`, and
`data/metadata/`. The SQLite state database tracks clip status and artifact
paths between runs.

## Stateful Clip Workflow

### Discover

`clipforge clips` searches Twitch clips and persists every returned clip to
SQLite. Without date filters, discovery searches the last 7 days in UTC.

```powershell
clipforge clips --channel "<channel_login>" --limit 20
```

Use UTC ISO-8601 filters when you want a specific window:

```powershell
clipforge clips --channel "<channel_login>" --started-at "2026-05-01T00:00:00Z" --ended-at "2026-05-06T00:00:00Z" --limit 50
```

The command prints a compact tab-separated list and updates local state. JSON
discovery exports are still available for manual inspection or external tools:

```powershell
clipforge clips --channel "<channel_login>" --limit 20 --format json
```

Pass `--output "<path>"` with `--format json` to choose the export path.

### Rerank

Discovery stores a deterministic rank score based on available Twitch metadata.
Refresh scores without calling Twitch again:

```powershell
clipforge clips rerank --channel "<channel_login>"
```

### Review

List unprocessed clips ordered by rank:

```powershell
clipforge clips pending --channel "<channel_login>" --limit 10
```

Add `--show-url` when you want the full clip URLs in the table.

Run the manual final-render review workflow:

```powershell
clipforge clips review --streamer "<channel_login>" --count 3
```

This discovers recent clips, updates SQLite state, selects the top-ranked
eligible clips from the DB, processes them through the normal pipeline, prompts
for one render per clip, and copies only selected renders to:

```text
data/exports/ready/<streamer>/<clip_id>/<layout_name>.mp4
```

Already exported or posted clips are not selected again. Existing ready exports
are preserved unless `--force` is passed.

### Process

Process by rank:

```powershell
clipforge clips process --top 3
```

Or process a specific saved clip:

```powershell
clipforge clips process --clip-id "<clip_id>"
```

Use `--continue-on-error` when processing a batch should continue after a single
clip fails.

### Reprocess

Rendered clips are excluded from pending and top-ranked processing. Reprocessing
is opt-in so an existing render is not replaced by accident:

```powershell
clipforge clips process --clip-id "<clip_id>" --force
```

Force reprocessing and regenerate captions in the same pass:

```powershell
clipforge clips process --clip-id "<clip_id>" --force --generate-captions
```

`--force` is only supported with `--clip-id`.

### Reset State

Move one saved clip back to `discovered` and clear stored processing, render,
failure, and export artifact paths:

```powershell
clipforge clips reset --clip-id "<clip_id>"
```

Reset every saved clip when you want to rebuild the full queue:

```powershell
clipforge clips reset --all
```

## Direct Pipeline Commands

The stateful clip workflow is the normal path. Direct commands are available for
manual or advanced use:

| Command | Use |
| --- | --- |
| `clipforge process --url "<twitch_clip_url>"` | Run the full URL-to-candidates pipeline for one Twitch clip. |
| `clipforge --url "<twitch_clip_url>"` | Shortcut for the full direct pipeline. |
| `clipforge resolve-url --url "<twitch_clip_url>"` | Resolve a Twitch clip URL to a direct media URL through Clipr. |
| `clipforge download --media-url "<direct_media_url>" --clip-id "<clip_id>"` | Download a direct media URL. |
| `clipforge render --source "<path>" --layout center_gameplay` | Render one layout from a local source clip. |
| `clipforge render-all --source "<path>"` | Render all default candidate layouts from a local source clip. |
| `clipforge captions --source "<path>" --clip-id "<clip_id>"` | Generate caption metadata for a local source clip. Add `--output "<path>"` to choose the JSON path. |

Default candidate layouts live in `examples/layouts/` and currently include
`center_gameplay`, `facecam_focus`, and `hybrid`.

## Caption Workflow

Captions are optional. Enable them for the full processing flow with either an
environment variable:

```text
CLIPFORGE_GENERATE_CAPTIONS=true
```

or per command:

```powershell
clipforge clips process --clip-id "<clip_id>" --generate-captions
```

Caption generation extracts temporary speech-optimized audio with FFmpeg,
transcribes it with the configured OpenAI transcription model, and writes JSON
metadata to:

```text
data/metadata/captions/<clip_id>.json
```

Existing caption metadata can be reused during manual renders:

```powershell
clipforge render-all --source "data/downloads/<clip_id>/ytdlp/<clip_id>.mp4" --clip-id "<clip_id>" --captions "data/metadata/captions/<clip_id>.json"
```

Caption rendering defaults to FFmpeg `drawtext`. Set
`CLIPFORGE_CAPTION_RENDERER=ass` to render from generated ASS subtitle files
under `CLIPFORGE_ASS_TEMP_DIR`. Rerenders with existing caption JSON do not need
another transcription request.

Set `CLIPFORGE_CAPTION_FONT_FILE` to a local font file when you want explicit
caption font control:

```text
CLIPFORGE_CAPTION_FONT_FILE=C:/Windows/Fonts/arial.ttf
```

## Project Structure

```text
clipforge/
  src/clipforge/        # application code
  data/
    downloads/          # downloaded source clips
    renders/            # rendered vertical candidates
    metadata/           # run metadata, discovery exports, captions, ASS files
      captions/         # caption metadata JSON
      ass/              # generated ASS subtitle files
    state/
      clipforge.sqlite  # local clip workflow state
  examples/
    layouts/            # reusable render layouts
  tests/                # automated tests
```

The main CLI entry point is `clipforge.pipeline.cli:main`.

## Development And Testing

Run the test suite from the repository root:

```powershell
python -m pytest .\tests\
```

Use the global `--verbose` flag before a subcommand when you want progress logs.

The pytest configuration adds the repo root to Python's import path so shared
test helpers resolve consistently.

## Future Ideas

- Better review controls for approving, skipping, and posting clips from state.
- Smarter layout selection based on clip content.
- Facecam or subject-aware crop assistance.
- Caption styling presets.
- Title, description, and thumbnail draft generation.
- Upload or publishing integrations after human approval.
