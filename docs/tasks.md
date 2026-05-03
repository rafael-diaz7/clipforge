# clipforge MVP Task Plan

This plan is organized as small milestones. Each task should be small enough to land as one git commit. Future agents should complete tasks in order unless the user explicitly changes priority.

Task status markers:

- `[x]` Complete
- `[ ]` Not started

## Milestone 1: Documentation Foundation

### [x] 1. docs: add initial design and task plan

Goal: Add the project design and implementation task plan before runtime implementation begins.

Files likely touched:

- `docs/design.md`
- `docs/tasks.md`

Acceptance criteria:

- `docs/design.md` explains project purpose, MVP scope, architecture, data flow, layout system, rendering approach, cross-platform requirements, error handling, git hygiene, and future extensions.
- `docs/tasks.md` lists milestone tasks with goal, files likely touched, acceptance criteria, and suggested commit message.
- No runtime code is implemented.
- No external APIs are called.
- No secrets, generated videos, or large binary files are added.

Suggested commit message:

```text
docs: add initial design and task plan
```

## Milestone 2: Configuration and Paths

### [x] 2. chore: add shared config and path helpers

Goal: Centralize environment loading, path constants, render defaults, and shared path helpers.

Status: Complete in this commit.

Files likely touched:

- `src/clipforge/config.py`
- `src/clipforge/utils.py`
- `.env.example`

Acceptance criteria:

- `CLIPR_API_KEY` is read from the environment without hardcoding secrets.
- Project paths for downloads, renders, metadata, and example layouts are defined with `pathlib`.
- Target render resolution defaults to `1080x1920`.
- Missing required configuration can be reported clearly.
- `.env.example` documents required variables without real secrets.

Suggested commit message:

```text
chore: add shared config and path helpers
```

### [ ] 3. feat: add Clipr API client

Goal: Add a small client that exchanges a Twitch clip URL for a direct downloadable video URL using Clipr.

Files likely touched:

- `src/clipforge/clipr.py`
- `src/clipforge/config.py`
- `src/clipforge/utils.py`

Acceptance criteria:

- The client accepts a Twitch clip URL and API key.
- The API key is passed from configuration and is never logged.
- The client returns a direct downloadable video URL.
- Clipr HTTP errors, malformed responses, and missing download URLs raise clear exceptions.
- No downloading of video bytes happens in this module.

Suggested commit message:

```text
feat: add Clipr API client
```

### [ ] 4. feat: add local clip downloader

Goal: Download direct media URLs to `data/downloads/` safely and predictably.

Files likely touched:

- `src/clipforge/download.py`
- `src/clipforge/utils.py`
- `src/clipforge/config.py`

Acceptance criteria:

- The downloader streams video content to disk instead of loading the full file into memory.
- Downloaded filenames are safe for local filesystems.
- Downloads are saved under `data/downloads/`.
- Failed or incomplete downloads raise clear errors.
- The function returns the local source clip path.

Suggested commit message:

```text
feat: add local clip downloader
```

## Milestone 3: Layouts

### [ ] 5. feat: define layout schema and example layouts

Goal: Define editable JSON layout templates for the three MVP candidates.

Files likely touched:

- `src/clipforge/layouts.py`
- `examples/layouts/center_gameplay.json`
- `examples/layouts/facecam_focus.json`
- `examples/layouts/hybrid.json`

Acceptance criteria:

- Layout JSON uses normalized coordinates from `0` to `1`.
- Three committed templates exist for center gameplay, facecam-focused, and hybrid candidates.
- The loader validates required fields and coordinate bounds.
- Invalid or missing layout files raise clear errors.
- Layouts are editable without changing Python code.

Suggested commit message:

```text
feat: define layout schema and example layouts
```

## Milestone 4: Rendering

### [ ] 6. feat: add FFmpeg render command builder

Goal: Convert layout data into FFmpeg argument lists without running shell strings.

Files likely touched:

- `src/clipforge/render.py`
- `src/clipforge/layouts.py`
- `src/clipforge/utils.py`

Acceptance criteria:

- Renderer builds FFmpeg commands as `list[str]`.
- Commands target `1080x1920` MP4 output.
- Normalized source and output regions are converted to pixel operations.
- FFmpeg execution uses `subprocess.run([...])`.
- Missing FFmpeg and non-zero FFmpeg exits produce clear errors.

Suggested commit message:

```text
feat: add FFmpeg render command builder
```

### [ ] 7. feat: render center crop candidate

Goal: Generate the center gameplay crop candidate from its layout template.

Files likely touched:

- `src/clipforge/render.py`
- `examples/layouts/center_gameplay.json`
- `src/clipforge/layouts.py`

Acceptance criteria:

- The center gameplay layout renders a vertical `1080x1920` MP4.
- Output is written to `data/renders/`.
- The source crop is centered around gameplay according to layout values.
- Render output path is returned to the caller.

Suggested commit message:

```text
feat: render center crop candidate
```

### [ ] 8. feat: render facecam-focused candidate

Goal: Generate the facecam-focused vertical candidate from its layout template.

Files likely touched:

- `src/clipforge/render.py`
- `examples/layouts/facecam_focus.json`
- `src/clipforge/layouts.py`

Acceptance criteria:

- The facecam-focused layout renders a vertical `1080x1920` MP4.
- Output is written to `data/renders/`.
- The source crop emphasizes the facecam area according to layout values.
- Render output path is returned to the caller.

Suggested commit message:

```text
feat: render facecam-focused candidate
```

### [ ] 9. feat: render hybrid candidate

Goal: Generate the hybrid candidate with facecam plus gameplay sections.

Files likely touched:

- `src/clipforge/render.py`
- `examples/layouts/hybrid.json`
- `src/clipforge/layouts.py`

Acceptance criteria:

- The hybrid layout renders a vertical `1080x1920` MP4.
- Output is written to `data/renders/`.
- Facecam and gameplay sections are composed from layout regions.
- Overlay ordering is deterministic.
- Render output path is returned to the caller.

Suggested commit message:

```text
feat: render hybrid candidate
```

## Milestone 5: CLI and Metadata

### [ ] 10. feat: add CLI entrypoint for URL to rendered candidates

Goal: Wire the full MVP flow into `python -m clipforge.render_clip --url "<twitch_clip_url>"`.

Files likely touched:

- `src/clipforge/render_clip.py`
- `src/clipforge/config.py`
- `src/clipforge/clipr.py`
- `src/clipforge/download.py`
- `src/clipforge/layouts.py`
- `src/clipforge/render.py`
- `src/clipforge/utils.py`

Acceptance criteria:

- CLI accepts a Twitch clip URL with `--url`.
- CLI validates the URL before calling Clipr.
- CLI retrieves a Clipr download URL, downloads the source clip, loads three layouts, and renders three candidates.
- CLI saves metadata under `data/metadata/`.
- Metadata includes clip ID or slug, source URL, Clipr download URL, local source path, output paths, layout values, target resolution, and timestamps.
- CLI prints concise output paths for the user.

Suggested commit message:

```text
feat: add CLI entrypoint for URL to rendered candidates
```

### [ ] 11. chore: improve error handling and logging

Goal: Make failures understandable and safe without exposing secrets.

Files likely touched:

- `src/clipforge/render_clip.py`
- `src/clipforge/config.py`
- `src/clipforge/clipr.py`
- `src/clipforge/download.py`
- `src/clipforge/layouts.py`
- `src/clipforge/render.py`
- `src/clipforge/utils.py`

Acceptance criteria:

- Missing `CLIPR_API_KEY` produces a clear configuration error.
- Invalid Twitch clip URLs fail before network calls.
- Clipr, download, layout, and FFmpeg failures include useful context.
- API keys are not logged.
- CLI exits with non-zero status on failures.
- Logging works cross-platform and does not rely on shell-specific behavior.

Suggested commit message:

```text
chore: improve error handling and logging
```

## Milestone 6: Usage Documentation

### [ ] 12. docs: update README with working usage

Goal: Update the README once the MVP CLI flow is functional.

Files likely touched:

- `README.md`
- `.env.example`

Acceptance criteria:

- README documents install steps for Windows Git Bash, Ubuntu/Linux, and macOS.
- README explains FFmpeg as a requirement.
- README documents `CLIPR_API_KEY`.
- README shows the working CLI command.
- README explains where downloads, renders, and metadata are saved.
- README keeps the human-in-the-loop framing clear.
- README does not claim non-MVP features exist.

Suggested commit message:

```text
docs: update README with working usage
```
