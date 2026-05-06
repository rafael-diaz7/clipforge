# clipforge Task Roadmap

The MVP pipeline is working end to end: a Twitch clip URL resolves through Clipr, downloads locally, renders three vertical candidates with FFmpeg, and writes metadata.

This roadmap tracks post-MVP product improvements. Tasks are ordered so each one leaves the project in a usable state. Keep each task small enough to review as one focused change, and preserve the existing one-shot `clipforge --url "<twitch_clip_url>"` workflow unless the task explicitly says otherwise.

Task status markers:

- `[x]` Complete
- `[ ]` Not started

Task fields:

- Goal: the product or engineering outcome.
- Files likely touched: starting points, not a required or exhaustive list.
- Acceptance criteria: testable behavior that should be true when the task is done.
- Out of scope: tempting adjacent work that should stay out of the task.

## Baseline

### [x] 0. MVP: URL to rendered candidates

Goal: Preserve the known-good baseline before adding intelligence or polish.

Current behavior:

- Accepts a Twitch clip URL through `clipforge --url "<twitch_clip_url>"`.
- Resolves a direct download URL through Clipr.
- Prints the resolved download URL before downloading.
- Downloads the source clip to `data/downloads/`.
- Renders `center_gameplay`, `facecam_focus`, and `hybrid` candidates to `data/renders/`.
- Writes run metadata to `data/metadata/`.
- Keeps the workflow human-in-the-loop.

Verification command:

```text
.\.venv\Scripts\python -m pytest
```

## Milestone 1: Reliable Downloading

### [ ] 1. feat: add pluggable downloader backend

Goal: Let Clipforge select a downloader provider while preserving the existing Clipr-backed workflow.

Files likely touched:

- `src/clipforge/download.py`
- `src/clipforge/clipr.py`
- `src/clipforge/config.py`
- `src/clipforge/render_clip.py`
- `tests/test_download.py`
- `tests/test_config.py`
- `.env.example`
- `README.md`

Acceptance criteria:

- Existing `clipforge --url "<twitch_clip_url>"` workflow still works.
- Current Clipr behavior is moved behind a `CliprDownloader`.
- Downloader backend selection is centralized and config/env-backed, with Clipr as the default.
- Invalid downloader backend names fail with a clear configuration error.
- Downloaded output path is returned consistently through the downloader abstraction.
- Existing render and metadata behavior still works after download.
- Tests cover downloader selection without calling live Clipr or Twitch.

Out of scope:

- No Twitch clip discovery yet.
- No captions yet.
- No custom Twitch extractor.
- No batch processing.
- No `yt-dlp` execution yet; that is handled by the next task.

Suggested commit message:

```text
feat: add pluggable clip downloader
```

### [ ] 2. feat: add yt-dlp downloader backend

Goal: Add a `yt-dlp` provider so Clipforge can download Twitch clip URLs without Clipr.

Files likely touched:

- `src/clipforge/download.py`
- `src/clipforge/config.py`
- `src/clipforge/render_clip.py`
- `tests/test_download.py`
- `tests/test_config.py`
- `.env.example`
- `README.md`

Acceptance criteria:

- New `YtDlpDownloader` can download a Twitch clip URL directly to `data/downloads/`.
- Downloader backend is selected by config/env, e.g. `CLIPFORGE_DOWNLOADER=clipr|ytdlp`.
- If `yt-dlp` is selected but missing, the command fails with a clear install/help message.
- Downloaded output path is returned consistently regardless of backend.
- Existing render and metadata behavior still works after download.
- Tests cover `yt-dlp` selection and command construction without calling live Twitch or `yt-dlp`.

Out of scope:

- No Twitch clip discovery yet.
- No captions yet.
- No custom Twitch extractor.
- No batch processing.

Suggested commit message:

```text
feat: add yt-dlp clip downloader
```

## Milestone 2: Clip Selection

### [ ] 3. feat: add Twitch clip discovery client

Goal: Discover candidate clips from Twitch instead of requiring one URL at a time.

Files likely touched:

- `src/clipforge/twitch.py`
- `src/clipforge/config.py`
- `src/clipforge/render_clip.py`
- `tests/test_twitch.py`
- `.env.example`
- `README.md`

Acceptance criteria:

- Twitch credentials are loaded from environment variables without hardcoding secrets.
- CLI can list recent clips for a channel.
- Clip search supports a small set of practical filters such as date range and result limit.
- API errors are clear and do not expose credentials.
- No automatic rendering is triggered by discovery alone.
- HTTP interactions are testable without calling the live Twitch API.

Out of scope:

- No ranking or queue processing yet.
- No downloading or rendering discovered clips.

Suggested commit message:

```text
feat: discover Twitch clips
```

### [ ] 4. feat: rank discovered clips

Goal: Sort discovered clips into a useful review queue.

Files likely touched:

- `src/clipforge/twitch.py`
- `src/clipforge/ranking.py`
- `tests/test_ranking.py`
- `README.md`

Acceptance criteria:

- Ranking uses available metadata such as views, age, duration, creator, and title.
- Scoring is deterministic, transparent, and easy to tune.
- CLI prints a concise ranked list with clip URLs.
- Ranking does not call rendering or downloading automatically.

Out of scope:

- No AI scoring.
- No automatic queue processing.

Suggested commit message:

```text
feat: rank discovered clips
```

### [ ] 5. feat: process selected clips from a review queue

Goal: Let the user choose clips from discovery results and render only selected items.

Files likely touched:

- `src/clipforge/render_clip.py`
- `src/clipforge/twitch.py`
- `src/clipforge/ranking.py`
- `tests/test_render_clip.py`
- `README.md`

Acceptance criteria:

- CLI can save a discovered and ranked queue to metadata.
- CLI can process one selected clip from that queue by stable ID, URL, or list index.
- Already-downloaded clips are reused when possible.
- Outputs remain organized by clip slug.
- The workflow stays human-in-the-loop.

Out of scope:

- No automatic batch rendering.
- No upload integration.

Suggested commit message:

```text
feat: process selected clips from queue
```

## Milestone 3: Captions and Render Polish

### [ ] 6. feat: add caption metadata model

Goal: Define the on-disk caption schema and helpers before wiring in transcription or render overlays.

Files likely touched:

- `src/clipforge/captions.py`
- `src/clipforge/render_clip.py`
- `tests/test_captions.py`
- `tests/test_render_clip.py`
- `README.md`

Acceptance criteria:

- Caption segments include start time, end time, and text.
- Caption JSON round-trips through load/save helpers with stable ordering and readable formatting.
- Caption metadata is saved under `data/metadata/` with a deterministic path derived from the clip ID.
- Existing run metadata can optionally reference a caption metadata path.
- Empty caption lists are valid and distinct from missing caption metadata.
- Existing download and render flows still work when captions are absent.

Out of scope:

- No transcription backend is added.
- No captions are burned into video.

Suggested commit message:

```text
feat: add caption metadata model
```

### [ ] 7. feat: generate captions from downloaded clips

Goal: Transcribe the downloaded source clip into timed caption segments.

Files likely touched:

- `src/clipforge/captions.py`
- `src/clipforge/config.py`
- `src/clipforge/render_clip.py`
- `tests/test_captions.py`
- `.env.example`
- `README.md`

Acceptance criteria:

- CLI can generate caption metadata for a local source clip.
- Full pipeline can optionally generate captions after download and before rendering through an explicit opt-in flag or config setting.
- Caption generation uses the schema from task 1 and saves metadata before rendering starts.
- If caption generation is requested and fails, the command exits with a clear error before rendering.
- Secrets, source paths, backend names, and external-service failures are logged safely and usefully.
- New dependencies, model downloads, or external services are documented before adoption.

Out of scope:

- Rendering captions onto videos remains separate.
- Automatic clip discovery remains separate.

Suggested commit message:

```text
feat: generate clip captions
```

### [ ] 8. feat: burn captions into renders

Goal: Overlay timed captions onto rendered vertical candidates.

Files likely touched:

- `src/clipforge/render.py`
- `src/clipforge/captions.py`
- `src/clipforge/render_clip.py`
- `tests/test_render.py`
- `tests/test_render_clip.py`

Acceptance criteria:

- Renderer accepts optional caption metadata produced by task 1 or task 2.
- FFmpeg command builder can include caption overlays without using shell strings.
- Caption style is readable on mobile vertical video and can be tuned from one place.
- Captions stay inside a safe area and avoid obvious layout collisions with known templates.
- Rendering without captions remains supported and remains the default.
- Tests cover command generation with and without captions.

Out of scope:

- No transcription changes.
- No per-platform caption style presets yet.

Suggested commit message:

```text
feat: render captions onto candidates
```

### [ ] 9. feat: add render diagnostics

Goal: Print and record basic output diagnostics after rendering so bad outputs are easier to spot.

Files likely touched:

- `src/clipforge/render.py`
- `src/clipforge/render_clip.py`
- `src/clipforge/probe.py`
- `tests/test_render_clip.py`
- `tests/test_probe.py`
- `README.md`

Acceptance criteria:

- CLI can print basic output diagnostics: duration, resolution, audio presence, and file size.
- Diagnostics run after each render.
- Metadata records diagnostics when they are available.
- Missing audio is called out clearly.
- Diagnostics use FFprobe when available and warn without failing the render flow when FFprobe is missing.

Out of scope:

- No thumbnail, contact sheet, or browser-based preview UI yet.

Suggested commit message:

```text
feat: add render diagnostics
```

## Milestone 4: Smarter Layouts

### [ ] 10. feat: sample frames from source clips

Goal: Extract representative frames that later layout logic can analyze.

Files likely touched:

- `src/clipforge/analyze.py`
- `src/clipforge/render_clip.py`
- `tests/test_analyze.py`
- `.gitignore`
- `README.md`

Acceptance criteria:

- CLI can sample frames from a local source clip.
- Samples are saved under an ignored data directory, such as `data/analysis/<clip_id>/frames/`.
- Sampling frequency or count is configurable with a simple default.
- FFmpeg failures include useful context.
- No layout behavior changes yet.

Out of scope:

- No facecam detection or layout generation yet.
- No committed generated frames.

Suggested commit message:

```text
feat: sample frames for layout analysis
```

### [ ] 11. feat: detect likely facecam region

Goal: Find a stable face or facecam area from sampled frames without training a custom model yet.

Files likely touched:

- `src/clipforge/analyze.py`
- `src/clipforge/layouts.py`
- `tests/test_analyze.py`
- `README.md`

Acceptance criteria:

- Analysis returns a normalized rectangle for the likely facecam region.
- Detection results include confidence or enough metadata to explain fallbacks.
- Failure to detect a facecam falls back cleanly to static layouts.
- Detected regions are saved in metadata.
- Detection runs locally with no API keys or external services.
- Any new computer-vision dependency is justified in the README before adoption.

Out of scope:

- No custom model training.
- No dynamic layout generation yet.

Suggested commit message:

```text
feat: detect facecam region
```

### [ ] 12. feat: generate dynamic layout candidates

Goal: Create layout JSON dynamically from detected facecam/gameplay regions.

Files likely touched:

- `src/clipforge/layouts.py`
- `src/clipforge/analyze.py`
- `src/clipforge/render_clip.py`
- `tests/test_layouts.py`
- `tests/test_render_clip.py`

Acceptance criteria:

- Dynamic layouts use the existing normalized layout schema.
- Generated layouts are saved to metadata for manual review and tweaking.
- Static layout templates remain available as fallbacks.
- Candidate names make it clear whether they are static or detected.
- Tests cover detected and fallback layout generation.

Out of scope:

- No frame sampling or detection changes beyond consuming saved analysis results.
- No review UI.

Suggested commit message:

```text
feat: generate detected layout candidates
```

## Milestone 5: Workflow UX

### [ ] 13. feat: add resumable pipeline state

Goal: Make failed or interrupted runs easier to continue.

Files likely touched:

- `src/clipforge/render_clip.py`
- `src/clipforge/state.py`
- `tests/test_render_clip.py`
- `tests/test_state.py`

Acceptance criteria:

- Each major stage records completion status.
- A failed run preserves resolved URLs, source paths, caption paths, and partial output metadata.
- CLI can resume from the latest completed stage.
- Resume avoids repeating completed download/render work unless an explicit force option is used.
- Existing one-shot `clipforge --url` behavior remains available.

Out of scope:

- No background worker or scheduler.
- No multi-clip batch orchestration.

Suggested commit message:

```text
feat: add resumable pipeline state
```

### [ ] 14. docs: refresh README for post-MVP workflow

Goal: Keep the README aligned as downloading, discovery, captions, analysis, and pipeline state land.

Files likely touched:

- `README.md`
- `docs/tasks.md`

Acceptance criteria:

- README documents the current recommended workflow.
- Optional features are clearly marked as optional.
- Required external tools and credentials are listed.
- The docs do not claim unfinished features exist.

Out of scope:

- No new runtime behavior.

Suggested commit message:

```text
docs: refresh post-MVP usage
```

## Backlog Notes

### [ ] research: collect bad layout examples

Goal: Build a small local evidence set before considering custom model training.

Files likely touched:

- `docs/layout-evaluation.md`
- `docs/tasks.md`

Acceptance criteria:

- Document common layout failure modes.
- Capture representative examples without committing large videos or secrets.
- Decide whether heuristics are enough or a trained model is justified.
- List what labels would be needed for a training set if ML becomes worthwhile.

Out of scope:

- No model training.
- No committed source clips or rendered videos.

Suggested commit message:

```text
docs: track layout failure cases
```
