# clipforge Design

## Project Purpose

clipforge is a local Python tool for reducing the repetitive editing work involved in turning Twitch clips into short-form vertical videos for YouTube Shorts, TikTok, Instagram Reels, and similar platforms.

The project is a semi-automated editing pipeline, not a fully autonomous content farm. It should help a human editor move faster by generating useful render candidates and preserving editable layout metadata. The human still decides whether a clip is worth posting, reviews the rendered candidates, tweaks layout values when needed, and manually posts the final video.

The intended workflow is:

1. A user provides a Twitch clip URL.
2. clipforge downloads the source clip locally.
3. clipforge generates several vertical edit candidates from configurable layouts.
4. The user reviews the candidates.
5. The user edits layout metadata and re-renders if a candidate needs adjustment.
6. The user manually posts the final result.

## MVP Scope

The first working version should support this flow:

`Twitch clip URL -> Clipr download URL -> local MP4 download -> three vertical renders -> local metadata JSON`

The three MVP render candidates are:

1. Center gameplay crop.
2. Facecam-focused crop.
3. Hybrid layout with facecam plus a gameplay section.

The MVP should be local-only, CLI-driven, FFmpeg-based, and configured with editable JSON layout files. Clip downloading should use the Clipr API. The Clipr API key must be read from the `CLIPR_API_KEY` environment variable and must never be hardcoded.

MVP non-goals:

- Automatic Twitch clip discovery.
- Twitch API integration.
- Captions or Whisper transcription.
- Auto-generated titles or descriptions.
- Automatic uploading to YouTube, TikTok, Instagram, or other platforms.
- AI-based clip scoring.
- Fully automated posting.
- Cloud infrastructure.
- Database storage.
- A review UI.
- Batch queue processing.

## Architecture

Planned modules under `src/clipforge/`:

### `config.py`

Loads environment variables and centralizes settings used across the app.

Responsibilities:

- Load `.env` values for local development.
- Read `CLIPR_API_KEY`.
- Define project paths such as `data/downloads/`, `data/renders/`, `data/metadata/`, and `examples/layouts/`.
- Define render defaults such as target width `1080`, target height `1920`, and output format `mp4`.
- Provide a single place for future timeout and logging defaults.

### `clipr.py`

Wraps Clipr API access.

Responsibilities:

- Accept a Twitch clip URL.
- Call Clipr with the configured API key.
- Parse the Clipr response.
- Return a direct downloadable video URL.
- Raise clear errors for missing credentials, invalid responses, and API failures.

This module should not download video bytes. It should only translate a Twitch clip URL into a downloadable media URL.

### `download.py`

Downloads video files into `data/downloads/`.

Responsibilities:

- Accept a direct video URL and output path or clip identifier.
- Stream the media file to disk.
- Avoid loading full videos into memory.
- Use safe filenames.
- Return the local source clip path.
- Raise clear errors for network failures, non-success responses, and incomplete downloads.

### `layouts.py`

Loads and validates editable JSON layout files.

Responsibilities:

- Load layout templates from `examples/layouts/`.
- Load per-clip layout metadata from `data/metadata/` when re-rendering later.
- Validate required fields, normalized coordinate bounds, and supported layout concepts.
- Provide layout objects or dictionaries that the renderer can consume.

Layouts should be editable without changing Python code.

### `render.py`

Builds and runs FFmpeg commands.

Responsibilities:

- Convert normalized layout coordinates into pixel crop, scale, pad, and overlay values.
- Build FFmpeg argument lists for each render candidate.
- Run FFmpeg with `subprocess.run([...])`.
- Export vertical videos at `1080x1920`.
- Return output paths and useful render metadata.
- Raise clear errors when FFmpeg is missing or returns a non-zero exit code.

Python should pass FFmpeg arguments as lists, not shell strings.

### `cli.py`

CLI entrypoint that wires the flow together.

Responsibilities:

- Parse CLI arguments such as `--url`.
- Validate the input Twitch clip URL.
- Load configuration.
- Request a Clipr download URL.
- Download the source clip.
- Load the three candidate layouts.
- Render all three candidates.
- Save metadata describing the source, layouts, outputs, and timestamps.
- Print concise local output paths for the user.

The planned command is:

```bash
python -m clipforge.pipeline.cli --url "<twitch_clip_url>"
```

### `utils.py`

Shared helpers used by the other modules.

Responsibilities:

- Path helpers built on `pathlib`.
- Safe filename generation.
- Clip ID extraction or slug generation.
- Timestamp helpers.
- Subprocess helpers.
- Small validation helpers shared by CLI, downloader, and renderer.

## Data Flow

1. The user runs the CLI with a Twitch clip URL.
2. The CLI validates that the input is shaped like a Twitch clip URL.
3. The CLI loads settings and confirms `CLIPR_API_KEY` is available.
4. The Clipr client sends the Twitch clip URL to Clipr and receives a direct downloadable video URL.
5. The downloader saves the source video to `data/downloads/`.
6. The layout loader loads the three candidate layout templates from `examples/layouts/`.
7. The renderer converts each layout from normalized coordinates into FFmpeg crop, scale, pad, and overlay operations.
8. The renderer writes three vertical MP4 outputs to `data/renders/`.
9. The metadata writer saves a JSON document in `data/metadata/` with:
   - Clip ID or slug.
   - Original Twitch clip URL.
   - Clipr download URL.
   - Local source path.
   - Render output paths.
   - Layout names and values used for each candidate.
   - Target resolution.
   - Created and rendered timestamps.

## Layout System

Layouts should be JSON files using normalized coordinates from `0` to `1`. Normalized coordinates make layouts resolution-independent, so the same layout can apply to clips with different source dimensions.

A region should describe a rectangle with normalized values:

```json
{
  "x": 0.0,
  "y": 0.0,
  "width": 1.0,
  "height": 1.0
}
```

Planned region concepts:

- `source_region`: The portion of the input video to crop from.
- `output_region`: The portion of the final vertical canvas where the processed source region should appear.
- `gameplay`: A region intended to emphasize the main gameplay area.
- `facecam`: A region intended to emphasize the streamer camera area.
- `background`: A full-canvas or blurred/scaled base layer behind foreground regions.
- `overlay`: A foreground element placed on top of another region.
- `effect`: An optional per-region render effect. The current supported value is `"blur"` for blurred background layers.

An MVP layout template should include enough information for the renderer to generate one vertical candidate:

```json
{
  "name": "center_gameplay",
  "description": "Centered vertical crop focused on gameplay.",
  "output": {
    "width": 1080,
    "height": 1920
  },
  "regions": [
    {
      "name": "gameplay",
      "source_region": {
        "x": 0.25,
        "y": 0.0,
        "width": 0.5,
        "height": 1.0
      },
      "output_region": {
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0
      }
    }
  ]
}
```

Future layout templates can add more fields, but the MVP should keep the schema simple and easy to edit by hand.

## Rendering Approach

FFmpeg is the rendering backend. Python should build FFmpeg commands as argument lists and call them with `subprocess.run([...])`. The code should avoid shell strings and should not use `os.system()`.

The default output target is `1080x1920` for vertical shorts. The renderer should support:

- Cropping from normalized source regions.
- Scaling cropped regions to output regions.
- Padding or composing onto a `1080x1920` canvas.
- Overlaying facecam/gameplay sections for hybrid layouts.
- Exporting MP4 files into `data/renders/`.

The renderer should assume `ffmpeg` is available in `PATH`, but it should fail with a clear message if FFmpeg is missing.

## Cross-Platform Requirements

clipforge should work on:

- Windows using Git Bash.
- Ubuntu/Linux.
- macOS.

Requirements:

- Use `pathlib.Path` for filesystem paths.
- Avoid hardcoded Windows or Unix path separators.
- Avoid shell-specific commands inside Python.
- Use `subprocess.run([...])` instead of shell strings or `os.system()`.
- Assume `ffmpeg` is available in `PATH`.
- Keep generated outputs under project-local `data/` directories.

## Error Handling and Logging

Expected behavior:

- Missing `CLIPR_API_KEY`: stop before making Clipr requests and print a clear configuration error.
- Invalid Twitch clip URL: stop before Clipr requests and tell the user the URL is not supported.
- Clipr API failure: show a concise error with status or response context, without printing secrets.
- Download failure: remove incomplete output when practical and report the failed URL/path.
- FFmpeg not installed: report that `ffmpeg` must be installed and available in `PATH`.
- FFmpeg render failure: report which candidate failed and include useful stderr context.
- Missing layout JSON: report the missing layout file path.
- Invalid layout JSON: report the file path and validation issue.

Logging should be useful but local. Logs must not include API keys or large response bodies. A future logging task can decide whether logs go only to stderr or also to ignored `.log` files.

## File and Git Hygiene

Generated videos, generated metadata, local environments, and logs should not be committed.

The following should stay ignored:

- `.env`
- `.venv/`
- `data/downloads/`
- `data/renders/`
- `data/clips/`
- `data/metadata/`
- `*.mp4`
- `*.mov`
- `*.log`

`examples/layouts/` is safe to commit because it stores hand-written layout templates, not generated outputs.

Do not commit API keys, downloaded source clips, rendered videos, or generated per-clip metadata.

## Future Extensions

Possible later additions:

- Twitch API clip discovery.
- Whisper captions.
- Facecam detection.
- Dynamic crops based on transcript or audio.
- Review UI for choosing and tweaking candidates.
- Upload integrations for YouTube, TikTok, Instagram, and other platforms.
- Queue and batch processing.

These are intentionally out of scope for the MVP design.
