"""Command line entry points for the clipforge pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from clipforge.core.config import ConfigError, load_config
from clipforge.integrations.twitch import list_channel_clips
from clipforge.media.captions import generate_caption_metadata
from clipforge.pipeline.artifacts import write_clip_discovery_export, write_metadata
from clipforge.pipeline.state_sync import record_discovered_clips, record_rendered_clip
from clipforge.pipeline.workflows import (
    download_media_url,
    process_clip,
    render_all_candidates,
    render_candidate,
    resolve_download_url,
)
from clipforge.storage.state import UNPROCESSED_STATUSES, get_clip, get_unprocessed_clips


LOGGER = logging.getLogger("clipforge.pipeline.cli")


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
    parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=None,
        help="Generate caption metadata after download and before rendering.",
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
    render_parser.add_argument(
        "--captions",
        help="Optional caption metadata JSON path to burn into the render.",
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
    render_all_parser.add_argument(
        "--captions",
        help="Optional caption metadata JSON path to burn into each render.",
    )

    captions_parser = subparsers.add_parser(
        "captions",
        help="Generate caption metadata for a local source clip.",
    )
    captions_parser.add_argument(
        "--source",
        required=True,
        help="Local source video path.",
    )
    captions_parser.add_argument(
        "--clip-id",
        help="Caption metadata clip ID. Defaults to the source filename stem.",
    )
    captions_parser.add_argument(
        "--output",
        help="Optional caption metadata JSON output path.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Run the full Twitch URL to rendered candidates pipeline.",
    )
    process_parser.add_argument("--url", required=True, help="Twitch clip URL.")
    process_parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=None,
        help="Generate caption metadata after download and before rendering.",
    )

    clips_parser = subparsers.add_parser(
        "clips",
        help="Discover clips or process saved clips from SQLite state.",
    )
    clips_subparsers = clips_parser.add_subparsers(dest="clips_command")
    clips_parser.add_argument("--channel", help="Twitch channel login.")
    clips_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum clips to list, from 1 to 100. Defaults to 10.",
    )
    clips_parser.add_argument(
        "--started-at",
        help="Optional UTC ISO-8601 start timestamp, e.g. 2026-05-01T00:00:00Z.",
    )
    clips_parser.add_argument(
        "--ended-at",
        help="Optional UTC ISO-8601 end timestamp, e.g. 2026-05-06T00:00:00Z.",
    )
    clips_parser.add_argument(
        "--format",
        choices=("json",),
        help="Export format. Passing this writes a discovery export file.",
    )
    clips_parser.add_argument(
        "--output",
        help="Optional JSON export path. Only used with --format json.",
    )
    clips_subparsers.add_parser(
        "pending",
        help="List unprocessed saved clips by rank.",
    )
    clips_process_parser = clips_subparsers.add_parser(
        "process",
        help="Process one saved clip from SQLite state.",
    )
    clips_process_group = clips_process_parser.add_mutually_exclusive_group(required=True)
    clips_process_group.add_argument(
        "--top",
        type=int,
        help="Process the highest-ranked unprocessed clip. Only --top 1 is supported.",
    )
    clips_process_group.add_argument(
        "--clip-id",
        help="Process a specific unprocessed saved clip ID.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    try:
        if args.url and args.command is None:
            if args.generate_captions is None:
                process_clip(args.url)
            else:
                process_clip(args.url, generate_captions=args.generate_captions)
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
            render_kwargs = {
                "layout_ref": args.layout,
                "clip_id": args.clip_id,
            }
            if args.captions:
                render_kwargs["caption_metadata_path"] = Path(args.captions)
            output_path = render_candidate(Path(args.source), **render_kwargs)
            print(output_path)
            return 0

        if args.command == "render-all":
            render_kwargs = {"clip_id": args.clip_id}
            if args.captions:
                render_kwargs["caption_metadata_path"] = Path(args.captions)
            for output_path in render_all_candidates(Path(args.source), **render_kwargs):
                print(output_path)
            return 0

        if args.command == "captions":
            caption_path = generate_caption_metadata(
                Path(args.source),
                clip_id=args.clip_id or Path(args.source).stem,
                output_path=Path(args.output) if args.output else None,
                config=load_config(),
            )
            print(caption_path)
            return 0

        if args.command == "process":
            if args.generate_captions is None:
                process_clip(args.url)
            else:
                process_clip(args.url, generate_captions=args.generate_captions)
            return 0

        if args.command == "clips":
            return _handle_clips_command(args)

        raise CLIError(f"Unsupported command: {args.command}")
    except (CLIError, ConfigError, RuntimeError, ValueError) as exc:
        print(f"clipforge: {exc}", file=sys.stderr)
        return 1


def _handle_clips_command(args: argparse.Namespace) -> int:
    if args.clips_command == "pending":
        return _handle_clips_pending_command(args)

    if args.clips_command == "process":
        return _handle_clips_process_command(args)

    if not args.channel:
        raise CLIError("clips discovery requires --channel.")

    if args.output and not args.format:
        raise CLIError("--output requires --format json.")

    started_at, ended_at = _clip_date_filters(
        started_at=args.started_at,
        ended_at=args.ended_at,
    )
    config = load_config()
    clips = list_channel_clips(
        args.channel,
        limit=args.limit,
        started_at=started_at,
        ended_at=ended_at,
        config=config,
    )
    record_discovered_clips(clips=clips, channel=args.channel, config=config)
    if args.format == "json":
        # TODO: Add more formats.
        export_path = write_clip_discovery_export(
            clips=clips,
            channel=args.channel,
            limit=args.limit,
            started_at=started_at,
            ended_at=ended_at,
            output_path=Path(args.output) if args.output else None,
            config=config,
        )
        print(f"export: {export_path}")
        return 0

    for clip in clips:
        print(
            "\t".join(
                (
                    clip.created_at,
                    str(clip.view_count),
                    f"{clip.duration:g}s",
                    clip.url,
                    clip.title,
                )
            )
        )
    return 0


def _handle_clips_pending_command(args: argparse.Namespace) -> int:
    config = load_config()
    for clip in get_unprocessed_clips(db_path=config.state_db_path):
        print(_format_state_clip(clip))
    return 0


def _handle_clips_process_command(args: argparse.Namespace) -> int:
    config = load_config()
    if args.top is not None:
        if args.top != 1:
            raise CLIError("clips process only supports --top 1.")
        clips = get_unprocessed_clips(db_path=config.state_db_path, limit=1)
        if not clips:
            raise CLIError("No unprocessed clips found.")
        clip = clips[0]
    else:
        clip = get_clip(args.clip_id, db_path=config.state_db_path)
        if clip is None:
            raise CLIError(f"Clip not found: {args.clip_id}.")
        if clip.status not in UNPROCESSED_STATUSES:
            raise CLIError(f"Clip is not unprocessed: {args.clip_id}.")

    process_clip(clip.url, config=config)
    return 0


def _format_state_clip(clip) -> str:
    rank_score = "" if clip.rank_score is None else f"{clip.rank_score:g}"
    view_count = "" if clip.view_count is None else str(clip.view_count)
    duration = "" if clip.duration_seconds is None else f"{clip.duration_seconds:g}s"
    title = clip.title or ""
    return "\t".join((clip.clip_id, rank_score, view_count, duration, clip.url, title))


def _configure_logging(*, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="clipforge: %(levelname)s: %(message)s")


def _clip_date_filters(
    *,
    started_at: str | None,
    ended_at: str | None,
) -> tuple[str | None, str | None]:
    if ended_at and not started_at:
        raise CLIError("--ended-at requires --started-at for Twitch clip discovery.")
    if started_at:
        return started_at, ended_at

    now = datetime.now(UTC).replace(microsecond=0)
    one_week_ago = now - timedelta(days=7)
    return _format_twitch_timestamp(one_week_ago), _format_twitch_timestamp(now)


def _format_twitch_timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
