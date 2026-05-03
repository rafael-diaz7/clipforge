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

This is a human-in-the-loop workflow, not full automation.



## Setup

Requirements:

- Python 3.11+
- FFmpeg installed and available in PATH

Check FFmpeg:


```bash
ffmpeg -version
```

Install:

```bash
python -m venv .venv

Windows (Git Bash):
source .venv/Scripts/activate

Linux / Mac:
source .venv/bin/activate

pip install -e .
```
Environment variables:

Copy .env.example to .env and fill in:

```bash
CLIPR_API_KEY=your_key_here
```

## Running (planned)

```bash
python -m clipforge.render_clip --url "<twitch_clip_url>"
```
This should:

- download the clip
- generate multiple vertical edits
- save them to data/renders/
- save layout metadata to data/metadata/

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


## Why I’m building this

Clipping is mostly repetitive work. The creative part is deciding what is funny or worth posting.

Everything else should be as fast as possible.