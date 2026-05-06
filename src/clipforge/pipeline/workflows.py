"""Reusable pipeline workflows."""

from __future__ import annotations

import logging
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.core.utils import ensure_directory, safe_filename, twitch_clip_slug_from_url
from clipforge.integrations.clipr import CliprClient
from clipforge.media.download import download_clip, download_twitch_clip
from clipforge.media.layouts import (
    DEFAULT_LAYOUT_NAMES,
    Layout,
    load_example_layouts,
    load_layout,
)
from clipforge.media.render import render_layout
from clipforge.pipeline.artifacts import write_metadata
from clipforge.pipeline.state_sync import record_rendered_clip


LOGGER = logging.getLogger("clipforge.pipeline.workflows")


def resolve_download_url(twitch_clip_url: str, *, config: ClipforgeConfig | None = None) -> str:
    """Resolve a Twitch clip URL to a direct downloadable media URL."""

    config = config or load_config()
    LOGGER.info("Resolving Twitch clip URL with Clipr.")
    return CliprClient.from_config(config).get_download_url(twitch_clip_url)


def download_media_url(
    media_url: str,
    *,
    clip_id: str | None = None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Download a direct media URL into the configured downloads directory."""

    config = config or load_config()
    LOGGER.info("Downloading clip media to %s.", config.downloads_dir)
    return download_clip(
        media_url,
        downloads_dir=config.downloads_dir,
        filename_stem=clip_id,
    )


def render_candidate(
    source_path: Path,
    *,
    layout_ref: str,
    clip_id: str | None = None,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Render one local source clip with one example layout or layout file."""

    config = config or load_config()
    layout = _load_layout_ref(layout_ref, config=config)
    output_path = _render_output_path(source_path, layout, clip_id=clip_id, config=config)
    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return render_layout(source_path, output_path, layout)


def render_all_candidates(
    source_path: Path,
    *,
    clip_id: str | None = None,
    config: ClipforgeConfig | None = None,
) -> tuple[Path, ...]:
    """Render the default MVP candidate layouts for one source clip."""

    config = config or load_config()
    layouts = load_example_layouts(DEFAULT_LAYOUT_NAMES, layouts_dir=config.example_layouts_dir)
    return tuple(
        _render_candidate_layout(source_path, layout, clip_id=clip_id, config=config)
        for layout in layouts
    )


def process_clip(
    twitch_clip_url: str,
    *,
    config: ClipforgeConfig | None = None,
) -> Path:
    """Run the full MVP pipeline and return the metadata path."""

    config = config or load_config()
    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)
    download_result = download_twitch_clip(
        twitch_clip_url,
        clip_id=clip_id,
        config=config,
        on_media_url_resolved=lambda media_url: print(f"download_url: {media_url}"),
    )
    source_path = download_result.source_path
    layouts = load_example_layouts(DEFAULT_LAYOUT_NAMES, layouts_dir=config.example_layouts_dir)

    outputs = []
    for layout in layouts:
        output_path = _render_candidate_layout(
            source_path,
            layout,
            clip_id=clip_id,
            backend=download_result.backend,
            config=config,
        )
        outputs.append({"layout": layout.name, "path": str(output_path)})

    metadata_path = write_metadata(
        clip_id=clip_id,
        twitch_clip_url=twitch_clip_url,
        download_result=download_result,
        source_path=source_path,
        layouts=layouts,
        outputs=outputs,
        config=config,
    )
    record_rendered_clip(
        clip_id=clip_id,
        twitch_clip_url=twitch_clip_url,
        render_dir=Path(outputs[0]["path"]).parent,
        metadata_path=metadata_path,
        config=config,
    )

    print(f"source: {source_path}")
    for output in outputs:
        print(f"{output['layout']}: {output['path']}")
    print(f"metadata: {metadata_path}")
    return metadata_path


def _load_layout_ref(layout_ref: str, *, config: ClipforgeConfig) -> Layout:
    path = Path(layout_ref)
    if path.suffix.lower() == ".json" or path.exists():
        LOGGER.info("Loading layout from %s.", path)
        return load_layout(path)

    layout_path = config.example_layouts_dir / f"{layout_ref}.json"
    LOGGER.info("Loading example layout %s from %s.", layout_ref, layout_path)
    return load_layout(layout_path)


def _render_output_path(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    backend: str | None = None,
    config: ClipforgeConfig,
) -> Path:
    stem = clip_id or source_path.stem
    if backend is not None:
        output_dir = ensure_directory(
            config.renders_dir / safe_filename(stem) / safe_filename(backend)
        )
        return output_dir / f"{layout.name}.{config.output_format}"

    output_dir = ensure_directory(config.renders_dir)
    return output_dir / f"{stem}_{layout.name}.{config.output_format}"


def _render_candidate_layout(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    backend: str | None = None,
    config: ClipforgeConfig,
) -> Path:
    output_path = _render_output_path(
        source_path,
        layout,
        clip_id=clip_id,
        backend=backend,
        config=config,
    )
    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return render_layout(source_path, output_path, layout)
