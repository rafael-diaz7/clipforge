"""Command line entry points for the clipforge pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from clipforge.core.config import ConfigError, load_config
from clipforge.media.analyze import sample_frames
from clipforge.integrations.twitch import list_channel_clips, twitch_channel_login_from_input
from clipforge.media.captions import generate_caption_metadata
from clipforge.media.layouts import generate_detected_layout_candidates
from clipforge.media.overlay import analyze_overlay, write_overlay_debug_images
from clipforge.pipeline.artifacts import write_clip_discovery_export
from clipforge.pipeline.processing import (
    SavedClipProcessingError,
    process_saved_clips,
    select_saved_clips_for_processing,
)
from clipforge.pipeline.prepare import prepare_streamer_clips
from clipforge.pipeline.state_sync import (
    record_discovered_clips,
    rerank_persisted_clips,
)
from clipforge.pipeline.workflows import (
    download_media_url,
    process_clip,
    render_all_candidates,
    render_candidate,
    resolve_download_url,
)
from clipforge.pipeline.review import review_streamer_clips
from clipforge.server.http import serve_review_app
from clipforge.storage.state import (
    get_clip,
    get_unprocessed_clips,
    reset_all_clips_to_discovered,
    reset_clip_to_discovered,
)


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
    parser.add_argument(
        "--force-captions",
        action="store_true",
        help="Regenerate caption metadata even when deterministic metadata already exists.",
    )
    parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="With --url, ignore generated analysis layouts and render static layouts.",
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
        help="Render the default layout candidates from a local source clip.",
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
    render_all_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
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
    process_parser.add_argument(
        "--force-captions",
        action="store_true",
        help="Regenerate caption metadata even when deterministic metadata already exists.",
    )
    process_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
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
    clips_pending_parser = clips_subparsers.add_parser(
        "pending",
        help="List unprocessed saved clips by rank.",
    )
    clips_pending_parser.add_argument(
        "--limit",
        type=int,
        dest="pending_limit",
        help="Maximum pending clips to list. Defaults to 10.",
    )
    clips_pending_parser.add_argument(
        "--channel",
        dest="pending_channel",
        help="Only list pending clips for this Twitch channel login.",
    )
    clips_pending_parser.add_argument(
        "--show-url",
        action="store_true",
        help="Include full clip URLs in the pending clips table.",
    )
    clips_process_parser = clips_subparsers.add_parser(
        "process",
        help="Process saved clips from SQLite state.",
    )
    clips_process_group = clips_process_parser.add_mutually_exclusive_group(required=True)
    clips_process_group.add_argument(
        "--top",
        type=int,
        help="Process the highest-ranked unprocessed clips.",
    )
    clips_process_group.add_argument(
        "--clip-id",
        help="Process a specific saved clip ID.",
    )
    clips_process_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow reprocessing a rendered clip when used with --clip-id.",
    )
    clips_process_parser.add_argument(
        "--rerender",
        action="store_true",
        help=(
            "Rebuild visual/render artifacts for --clip-id while reusing existing "
            "source video and captions."
        ),
    )
    clips_process_parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=None,
        help="Generate caption metadata before rendering.",
    )
    clips_process_parser.add_argument(
        "--force-captions",
        action="store_true",
        help="Regenerate caption metadata even when deterministic metadata already exists.",
    )
    clips_process_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
    )
    clips_process_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining clips after a clip fails.",
    )
    clips_rerender_parser = clips_subparsers.add_parser(
        "rerender",
        help="Rebuild visual/render artifacts for one saved clip without transcription.",
    )
    clips_rerender_parser.add_argument(
        "--clip-id",
        required=True,
        help="Saved clip ID to rerender.",
    )
    clips_rerender_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
    )
    clips_rerank_parser = clips_subparsers.add_parser(
        "rerank",
        help="Refresh saved clip rank scores from SQLite state.",
    )
    clips_rerank_parser.add_argument(
        "--channel",
        dest="rerank_channel",
        help="Only rerank clips for this Twitch channel login.",
    )
    clips_reset_parser = clips_subparsers.add_parser(
        "reset",
        help="Reset saved clip state back to discovered.",
    )
    clips_reset_group = clips_reset_parser.add_mutually_exclusive_group(required=True)
    clips_reset_group.add_argument(
        "--clip-id",
        help="Reset one saved clip ID back to discovered.",
    )
    clips_reset_group.add_argument(
        "--all",
        action="store_true",
        help="Reset every saved clip back to discovered.",
    )
    clips_review_parser = clips_subparsers.add_parser(
        "review",
        help="Discover, process, and manually export final renders for a streamer.",
    )
    clips_review_parser.add_argument(
        "--streamer",
        required=True,
        help="Twitch streamer login to discover and review.",
    )
    clips_review_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of top-ranked eligible clips to review. Defaults to 3.",
    )
    clips_review_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ready-to-post exports.",
    )
    clips_review_parser.add_argument(
        "--rerender",
        action="store_true",
        help=(
            "Regenerate visual/render artifacts while preserving source video and "
            "reusing existing captions."
        ),
    )
    clips_review_parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=None,
        help="Generate caption metadata before rendering.",
    )
    clips_review_parser.add_argument(
        "--force-captions",
        action="store_true",
        help="Regenerate caption metadata even when deterministic metadata already exists.",
    )
    clips_review_parser.add_argument(
        "--clip-id",
        action="append",
        help="Review a specific saved clip ID. May be passed more than once.",
    )
    clips_review_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
    )
    clips_prepare_parser = clips_subparsers.add_parser(
        "prepare",
        help="Discover, process, and render review-ready candidates without prompting.",
    )
    clips_prepare_parser.add_argument(
        "--streamer",
        required=True,
        help="Twitch streamer login to discover and prepare.",
    )
    clips_prepare_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of top-ranked eligible clips to prepare. Defaults to 3.",
    )
    clips_prepare_parser.add_argument(
        "--generate-captions",
        action="store_true",
        default=None,
        help="Generate caption metadata before rendering.",
    )
    clips_prepare_parser.add_argument(
        "--force-captions",
        action="store_true",
        help="Regenerate caption metadata even when deterministic metadata already exists.",
    )
    clips_prepare_parser.add_argument(
        "--clip-id",
        action="append",
        help="Prepare a specific saved clip ID. May be passed more than once.",
    )
    clips_prepare_parser.add_argument(
        "--static-layouts",
        action="store_true",
        help="Ignore generated analysis layouts and render static layouts.",
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Review clips already prepared into the local review queue.",
    )
    review_subparsers = review_parser.add_subparsers(dest="review_command")
    review_server_parser = review_subparsers.add_parser(
        "server",
        help="Start a local web server for reviewing prepared rendered clips.",
    )
    review_server_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind. Defaults to 127.0.0.1.",
    )
    review_server_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port to bind. Defaults to 8080.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Create lightweight local analysis artifacts.",
    )
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command")
    analyze_frames_parser = analyze_subparsers.add_parser(
        "frames",
        help="Sample representative frames from a local source clip.",
    )
    analyze_frames_parser.add_argument(
        "--source",
        required=True,
        help="Local source video path.",
    )
    analyze_frames_parser.add_argument(
        "--clip-id",
        required=True,
        help="Clip ID used for analysis artifact paths.",
    )
    analyze_frames_parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of frames to sample. Defaults to 12.",
    )
    analyze_frames_parser.add_argument(
        "--interval-seconds",
        type=float,
        help="Seconds between sampled frames. Defaults to 2 seconds.",
    )
    analyze_overlay_parser = analyze_subparsers.add_parser(
        "overlay",
        help="Infer the most likely streamer overlay from sampled frames.",
    )
    analyze_overlay_parser.add_argument(
        "--clip-id",
        required=True,
        help="Clip ID used for analysis artifact paths.",
    )
    analyze_overlay_parser.add_argument(
        "--debug-raw-faces",
        action="store_true",
        help="Write raw detector JSON and annotated raw subject/face debug frames.",
    )
    analyze_overlay_debug_parser = analyze_subparsers.add_parser(
        "overlay-debug",
        help="Draw overlay inference candidates onto sampled frames.",
    )
    analyze_overlay_debug_parser.add_argument(
        "--clip-id",
        required=True,
        help="Clip ID used for analysis artifact paths.",
    )
    analyze_layouts_parser = analyze_subparsers.add_parser(
        "layouts",
        help="Generate detected vertical layout candidates from overlay analysis.",
    )
    analyze_layouts_parser.add_argument(
        "--clip-id",
        required=True,
        help="Clip ID used for analysis artifact paths.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    try:
        if args.url and args.command is None:
            process_clip(args.url, **_process_clip_kwargs(args))
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
            if args.static_layouts:
                render_kwargs["use_generated_layouts"] = False
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
            process_clip(args.url, **_process_clip_kwargs(args))
            return 0

        if args.command == "clips":
            return _handle_clips_command(args)

        if args.command == "review":
            return _handle_review_command(args)

        if args.command == "analyze":
            return _handle_analyze_command(args)

        raise CLIError(f"Unsupported command: {args.command}")
    except (CLIError, ConfigError, RuntimeError, ValueError) as exc:
        print(f"clipforge: {exc}", file=sys.stderr)
        return 1


def _handle_analyze_command(args: argparse.Namespace) -> int:
    if args.analyze_command == "frames":
        metadata_path = sample_frames(
            Path(args.source),
            clip_id=args.clip_id,
            count=args.count,
            interval_seconds=args.interval_seconds,
        )
        print(metadata_path)
        return 0

    if args.analyze_command == "overlay":
        if args.debug_raw_faces:
            overlay_path = analyze_overlay(
                clip_id=args.clip_id,
                debug_raw_faces=True,
            )
        else:
            overlay_path = analyze_overlay(clip_id=args.clip_id)
        print(overlay_path)
        return 0

    if args.analyze_command == "overlay-debug":
        debug_dir = write_overlay_debug_images(clip_id=args.clip_id)
        print(debug_dir)
        return 0

    if args.analyze_command == "layouts":
        for layout_path in generate_detected_layout_candidates(clip_id=args.clip_id):
            print(layout_path)
        return 0

    raise CLIError("analyze requires a subcommand.")


def _handle_clips_command(args: argparse.Namespace) -> int:
    if args.clips_command == "pending":
        return _handle_clips_pending_command(args)

    if args.clips_command == "process":
        return _handle_clips_process_command(args)

    if args.clips_command == "rerender":
        return _handle_clips_rerender_command(args)

    if args.clips_command == "rerank":
        return _handle_clips_rerank_command(args)

    if args.clips_command == "reset":
        return _handle_clips_reset_command(args)

    if args.clips_command == "review":
        return _handle_clips_review_command(args)

    if args.clips_command == "prepare":
        return _handle_clips_prepare_command(args)

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
    limit = args.pending_limit if args.pending_limit is not None else args.limit
    channel = args.pending_channel or args.channel
    streamer_login = twitch_channel_login_from_input(channel) if channel else None
    clips = get_unprocessed_clips(
        db_path=config.state_db_path,
        limit=limit,
        streamer_login=streamer_login,
    )
    for line in _format_state_clip_table(clips, show_url=args.show_url):
        print(line)
    return 0


def _handle_clips_process_command(args: argparse.Namespace) -> int:
    config = load_config()
    _reject_rerender_caption_generation_conflict(args)
    try:
        clips = select_saved_clips_for_processing(
            top=args.top,
            clip_id=args.clip_id,
            force=args.force,
            rerender=args.rerender,
            config=config,
        )
    except SavedClipProcessingError as exc:
        raise CLIError(str(exc)) from exc

    process_kwargs = _process_clip_kwargs(
        args,
        config=config,
        include_force=True,
        include_rerender=True,
    )
    results = process_saved_clips(
        clips,
        config=config,
        process_kwargs=process_kwargs,
        continue_on_error=args.continue_on_error,
        process_clip_fn=process_clip,
    )
    failures = 0
    for result in results:
        if result.succeeded:
            print(f"processed: {result.clip.clip_id}: {result.metadata_path}")
        else:
            failures += 1
            print(f"failed: {result.clip.clip_id}: {result.error_message}")

    return 1 if failures else 0


def _handle_clips_rerender_command(args: argparse.Namespace) -> int:
    config = load_config()
    clip = get_clip(args.clip_id, db_path=config.state_db_path)
    if clip is None:
        raise CLIError(f"Clip not found: {args.clip_id}.")

    process_kwargs = {
        "config": config,
        "rerender": True,
        "use_generated_layouts": not args.static_layouts,
    }
    metadata_path = process_clip(clip.url, **process_kwargs)
    print(f"rerendered: {clip.clip_id}: {metadata_path}")
    return 0


def _handle_clips_rerank_command(args: argparse.Namespace) -> int:
    config = load_config()
    channel = args.rerank_channel or args.channel
    count = rerank_persisted_clips(config=config, channel=channel)
    suffix = "clip" if count == 1 else "clips"
    print(f"Reranked {count} {suffix}")
    return 0


def _handle_clips_reset_command(args: argparse.Namespace) -> int:
    config = load_config()
    if args.all:
        count = reset_all_clips_to_discovered(db_path=config.state_db_path)
    else:
        reset_clip_to_discovered(args.clip_id, db_path=config.state_db_path)
        count = 1

    suffix = "clip" if count == 1 else "clips"
    print(f"Reset {count} {suffix} to discovered")
    return 0


def _handle_clips_review_command(args: argparse.Namespace) -> int:
    _reject_rerender_caption_generation_conflict(args)
    started_at, ended_at = _clip_date_filters(
        started_at=args.started_at,
        ended_at=args.ended_at,
    )
    config = load_config()
    exported_paths = review_streamer_clips(
        streamer=args.streamer,
        count=args.count,
        force=args.force,
        rerender=args.rerender,
        generate_captions=args.generate_captions,
        force_captions=args.force_captions,
        clip_ids=args.clip_id or (),
        started_at=started_at,
        ended_at=ended_at,
        discovery_limit=max(args.count, args.limit),
        use_generated_layouts=not args.static_layouts,
        config=config,
    )
    if exported_paths:
        print("ready exports:")
        for path in exported_paths:
            print(path)
    return 0


def _handle_clips_prepare_command(args: argparse.Namespace) -> int:
    started_at, ended_at = _clip_date_filters(
        started_at=args.started_at,
        ended_at=args.ended_at,
    )
    config = load_config()
    result = prepare_streamer_clips(
        streamer=args.streamer,
        count=args.count,
        generate_captions=args.generate_captions,
        force_captions=args.force_captions,
        clip_ids=args.clip_id or (),
        started_at=started_at,
        ended_at=ended_at,
        discovery_limit=max(args.count, args.limit),
        use_generated_layouts=not args.static_layouts,
        config=config,
    )
    print(f"discovered/upserted: {result.discovered_count}")
    print(f"reranked: {result.reranked_count}")
    print(f"selected: {result.selected_count}")
    print(f"rendered: {result.rendered_count}")
    print(f"failed: {len(result.failed)}")
    if result.prepared:
        print("prepared clips:")
        for prepared in result.prepared:
            print(f"{prepared.clip_id}: {prepared.metadata_path}")
    if result.failed:
        print("failed clips:")
        for failed in result.failed:
            print(f"{failed.clip_id}: {failed.error_message}")
    return 1 if result.failed else 0


def _handle_review_command(args: argparse.Namespace) -> int:
    if args.review_command == "server":
        return _handle_review_server_command(args)
    raise CLIError("review requires a subcommand.")


def _handle_review_server_command(args: argparse.Namespace) -> int:
    if args.port < 1 or args.port > 65535:
        raise CLIError("--port must be between 1 and 65535.")

    config = load_config()
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Clipforge review server: http://{display_host}:{args.port}")
    if args.host == "0.0.0.0":
        print(
            "Bound to 0.0.0.0. From your phone, open "
            f"http://<tailscale-or-lan-ip>:{args.port}"
        )
    print("Reviews prepared rendered clips only. Fill the queue with clips prepare.")
    serve_review_app(host=args.host, port=args.port, config=config)
    return 0


def _format_state_clip_table(clips, *, show_url: bool = False) -> tuple[str, ...]:
    rows: list[list[str]] = []
    for index, clip in enumerate(clips, start=1):
        row = [
            str(index),
            clip.streamer_login or "",
            _format_optional_score(clip.rank_score),
            "" if clip.view_count is None else str(clip.view_count),
            _format_optional_duration(clip.duration_seconds),
            clip.status,
            clip.clip_id,
        ]
        if show_url:
            row.append(clip.url)
        row.append(clip.title or "")
        rows.append(row)

    headers = ["rank", "streamer", "score", "views", "duration", "status", "clip_id"]
    if show_url:
        headers.append("url")
    headers.append("title")

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    formatted_rows = [_format_table_row(headers, widths)]
    formatted_rows.append(_format_table_row(("-" * width for width in widths), widths))
    formatted_rows.extend(_format_table_row(row, widths) for row in rows)
    return tuple(formatted_rows)


def _process_clip_kwargs(
    args: argparse.Namespace,
    *,
    config: object | None = None,
    include_force: bool = False,
    include_rerender: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if config is not None:
        kwargs["config"] = config
    if args.generate_captions is not None:
        kwargs["generate_captions"] = args.generate_captions
    if args.force_captions:
        kwargs["force_captions"] = True
    if include_force and args.force:
        kwargs["force"] = True
    if include_rerender and getattr(args, "rerender", False):
        kwargs["rerender"] = True
    if args.static_layouts:
        kwargs["use_generated_layouts"] = False
    return kwargs


def _reject_rerender_caption_generation_conflict(args: argparse.Namespace) -> None:
    if not getattr(args, "rerender", False):
        return
    if getattr(args, "generate_captions", None) is True or getattr(
        args,
        "force_captions",
        False,
    ):
        raise CLIError(
            "--rerender reuses existing captions and does not regenerate "
            "transcriptions. Run once with --generate-captions first."
        )


def _format_table_row(values, widths: list[int]) -> str:
    cells = [str(value).ljust(widths[index]) for index, value in enumerate(values)]
    cells[-1] = cells[-1].rstrip()
    return "  ".join(cells)


def _format_optional_score(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _format_optional_duration(value: float | None) -> str:
    return "" if value is None else f"{value:g}s"


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
