"""Command line entry points for the clipforge pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from clipforge.clipr import CliprClient
from clipforge.config import (
    ClipforgeConfig,
    ConfigError,
    load_config,
)
from clipforge.download import download_clip
from clipforge.layouts import Layout, load_example_layouts, load_layout
from clipforge.render import render_layout
from clipforge.utils import ensure_directory, twitch_clip_slug_from_url, utc_timestamp


DEFAULT_LAYOUT_NAMES = ("center_gameplay", "facecam_focus", "hybrid")
LOGGER = logging.getLogger(__name__)


class CLIError(RuntimeError):
    """Raised when CLI arguments cannot be mapped to a pipeline operation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="Resolve, download, and render Twitch clips into vertical candidates.",
    )
    parser.add_argument(
        "--url",
        help="Run the full URL-to-renders pipeline for a Twitch clip URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress logging on stderr.",
    )

    subparsers = parser.add_subparsers(dest="command")

    resolve_parser = subparsers.add_parser(
        "resolve-url",
        help="Resolve a Twitch clip URL to a direct downloadable media URL.",
    )
    resolve_parser.add_argument("--url", required=True, help="Twitch clip URL.")

    download_parser = subparsers.add_parser(
        "download",
        help="Download a direct media URL into data/downloads/.",
    )
    download_parser.add_argument("--media-url", required=True, help="Direct media URL.")
    download_parser.add_argument(
        "--clip-id",
        help="Filename stem to use for the downloaded clip.",
    )

    render_parser = subparsers.add_parser(
        "render",
        help="Render one layout from a local source clip.",
    )
    render_parser.add_argument("--source", required=True, help="Local source video path.")
    render_parser.add_argument(
        "--layout",
        required=True,
        help="Example layout name or path to a layout JSON file.",
    )
    render_parser.add_argument(
        "--clip-id",
        help="Output filename prefix. Defaults to the source filename stem.",
    )

    render_all_parser = subparsers.add_parser(
        "render-all",
        help="Render the three MVP layout candidates from a local source clip.",
    )
    render_all_parser.add_argument(
        "--source",
        required=True,
        help="Local source video path.",
    )
    render_all_parser.add_argument(
        "--clip-id",
        help="Output filename prefix. Defaults to the source filename stem.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Run the full Twitch URL to rendered candidates pipeline.",
    )
    process_parser.add_argument("--url", required=True, help="Twitch clip URL.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    try:
        if args.url and args.command is None:
            process_clip(args.url)
            return 0

        if args.command is None:
            parser.print_help()
            return 0

        if args.command == "resolve-url":
            print(resolve_download_url(args.url))
            return 0

        if args.command == "download":
            print(download_media_url(args.media_url, clip_id=args.clip_id))
            return 0

        if args.command == "render":
            output_path = render_candidate(
                Path(args.source),
                layout_ref=args.layout,
                clip_id=args.clip_id,
            )
            print(output_path)
            return 0

        if args.command == "render-all":
            for output_path in render_all_candidates(Path(args.source), clip_id=args.clip_id):
                print(output_path)
            return 0

        if args.command == "process":
            process_clip(args.url)
            return 0

        raise CLIError(f"Unsupported command: {args.command}")
    except (CLIError, ConfigError, RuntimeError, ValueError) as exc:
        print(f"clipforge: {exc}", file=sys.stderr)
        return 1


def resolve_download_url(twitch_clip_url: str, *, config: ClipforgeConfig | None = None) -> str:
    """Resolve a Twitch clip URL to a direct downloadable media URL."""

    config = config or load_config(require_clipr_api_key=True)
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

    config = config or load_config(require_clipr_api_key=True)
    clip_id = twitch_clip_slug_from_url(twitch_clip_url)
    LOGGER.info("Starting clip pipeline for clip %s.", clip_id)
    download_url = resolve_download_url(twitch_clip_url, config=config)
    source_path = download_media_url(download_url, clip_id=clip_id, config=config)
    layouts = load_example_layouts(DEFAULT_LAYOUT_NAMES, layouts_dir=config.example_layouts_dir)

    outputs = []
    for layout in layouts:
        output_path = _render_candidate_layout(
            source_path,
            layout,
            clip_id=clip_id,
            config=config,
        )
        outputs.append({"layout": layout.name, "path": str(output_path)})

    metadata_path = write_metadata(
        clip_id=clip_id,
        twitch_clip_url=twitch_clip_url,
        clipr_download_url=download_url,
        source_path=source_path,
        layouts=layouts,
        outputs=outputs,
        config=config,
    )

    print(f"source: {source_path}")
    for output in outputs:
        print(f"{output['layout']}: {output['path']}")
    print(f"metadata: {metadata_path}")
    return metadata_path


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="clipforge: %(levelname)s: %(message)s")


def write_metadata(
    *,
    clip_id: str,
    twitch_clip_url: str,
    clipr_download_url: str,
    source_path: Path,
    layouts: Sequence[Layout],
    outputs: Sequence[dict[str, str]],
    config: ClipforgeConfig,
) -> Path:
    """Persist metadata for a full pipeline run."""

    metadata_dir = ensure_directory(config.metadata_dir)
    metadata_path = metadata_dir / f"{clip_id}.json"
    created_at = utc_timestamp()
    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "twitch_clip_url": twitch_clip_url,
        "clipr_download_url": clipr_download_url,
        "source_path": str(source_path),
        "outputs": list(outputs),
        "layouts": [asdict(layout) for layout in layouts],
        "target_resolution": {
            "width": config.target_width,
            "height": config.target_height,
        },
        "created_at": created_at,
        "rendered_at": created_at,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    config: ClipforgeConfig,
) -> Path:
    output_dir = ensure_directory(config.renders_dir)
    stem = clip_id or source_path.stem
    return output_dir / f"{stem}_{layout.name}.{config.output_format}"


def _render_candidate_layout(
    source_path: Path,
    layout: Layout,
    *,
    clip_id: str | None,
    config: ClipforgeConfig,
) -> Path:
    output_path = _render_output_path(source_path, layout, clip_id=clip_id, config=config)
    LOGGER.info("Rendering layout %s to %s.", layout.name, output_path)
    return render_layout(source_path, output_path, layout)


if __name__ == "__main__":
    raise SystemExit(main())
