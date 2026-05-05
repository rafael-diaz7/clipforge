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

### [x] 3. feat: add Clipr API client

Goal: Add a small client that exchanges a Twitch clip URL for a direct downloadable video URL using Clipr.

Status: Complete in this commit.

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

### [x] 4. feat: add local clip downloader

Goal: Download direct media URLs to `data/downloads/` safely and predictably.

Status: Complete in this commit.

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

### [x] 5. feat: define layout schema and example layouts

Goal: Define editable JSON layout templates for the three MVP candidates.

Status: Complete in this commit.

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

### [x] 6. feat: add FFmpeg render command builder

Goal: Convert layout data into FFmpeg argument lists without running shell strings.

Status: Complete in this commit.

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

### [x] 7. not needed: render center crop candidate

Goal: Previously intended to generate the center gameplay crop candidate from its layout template.

Status: Not needed. Task 6 already added the generic renderer, and the existing `render_candidate` / `render_all_candidates` orchestration renders named layout templates without per-layout code.

Files touched:

- `docs/tasks.md`

Resolution:

- Keep `render.py` layout-agnostic.
- Keep candidate behavior in JSON layout templates.
- Use `render_candidate(..., layout_ref="center_gameplay")` or `render_all_candidates(...)`.
- Do not add a center-specific render function.

Suggested commit message:

```text
docs: mark separate candidate render tasks unnecessary
```

### [x] 8. not needed: render facecam-focused candidate

Goal: Previously intended to generate the facecam-focused vertical candidate from its layout template.

Status: Not needed. The generic renderer already handles this candidate through the `facecam_focus` layout template.

Files touched:

- `docs/tasks.md`

Resolution:

- Keep candidate behavior in `examples/layouts/facecam_focus.json`.
- Use `render_candidate(..., layout_ref="facecam_focus")` or `render_all_candidates(...)`.
- Do not add a facecam-specific render function.

Suggested commit message:

```text
docs: mark separate candidate render tasks unnecessary
```

### [x] 9. not needed: render hybrid candidate

Goal: Previously intended to generate the hybrid candidate with facecam plus gameplay sections.

Status: Not needed. The generic renderer already composes multiple regions in layout order, so the hybrid candidate is data-driven by the `hybrid` layout template.

Files touched:

- `docs/tasks.md`

Resolution:

- Keep candidate behavior in `examples/layouts/hybrid.json`.
- Use `render_candidate(..., layout_ref="hybrid")` or `render_all_candidates(...)`.
- Do not add a hybrid-specific render function.

Suggested commit message:

```text
docs: mark separate candidate render tasks unnecessary
```

## Milestone 5: CLI and Metadata

### [x] 10. feat: add CLI entrypoint for URL to rendered candidates

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

Status: Done. The CLI entrypoint already wires the MVP flow through
`python -m clipforge.render_clip --url "<twitch_clip_url>"` and the installed
`clipforge --url "<twitch_clip_url>"` script.

Files touched:

- `docs/tasks.md`

Resolution:

- Validates supported Twitch clip URLs before calling Clipr via
  `twitch_clip_slug_from_url`.
- Resolves the Clipr download URL, downloads the source clip, renders the three
  default layout candidates, writes metadata under `data/metadata/`, and prints
  concise source/output/metadata paths.
- Verified existing coverage with the full test suite.

Tests:

```text
.\.venv\Scripts\python -m pytest tests\test_render_clip.py tests\test_clipr.py tests\test_download.py tests\test_layouts.py tests\test_render.py tests\test_config.py tests\test_utils.py
```

Suggested commit message:

```text
docs: mark CLI render pipeline task complete
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
