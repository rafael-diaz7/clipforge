"""Build and run FFmpeg commands for layout-based vertical renders."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from clipforge.media.caption_rendering import (
    DEFAULT_CAPTION_RENDERER_BACKEND,
    DEFAULT_CAPTION_STYLE,
    CaptionRenderingError,
    CaptionStyle,
    CaptionVerticalSafeArea,
    ass_subtitle_path,
    caption_filter_parts,
    require_caption_renderer_backend,
)
from clipforge.media.captions import CaptionMetadata
from clipforge.media.layouts import Layout, LayoutRegion, NormalizedRect, OutputSize
from clipforge.media.render_settings import (
    DEFAULT_FFMPEG_RENDER_SETTINGS,
    DEFAULT_X264_PRESET,
    FFmpegRenderSettings,
    NVENC_H264_ENCODER,
)


class RenderError(RuntimeError):
    """Raised when an FFmpeg command cannot be built or executed."""


class FFmpegCommandError(RenderError):
    """Raised when FFmpeg exits unsuccessfully."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        detail = f": {stderr}" if stderr else "."
        super().__init__(f"FFmpeg failed with exit code {returncode}{detail}")


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_WATERMARK_MARGIN = 32
DEFAULT_WATERMARK_MAX_WIDTH_RATIO = 0.45
DEFAULT_WATERMARK_OPACITY = 1.0
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PixelRect:
    """A rectangle described in output pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Watermark:
    """PNG watermark render settings."""

    path: Path
    native_width: int
    native_height: int
    margin: int = DEFAULT_WATERMARK_MARGIN
    max_width_ratio: float = DEFAULT_WATERMARK_MAX_WIDTH_RATIO
    opacity: float = DEFAULT_WATERMARK_OPACITY

    def __post_init__(self) -> None:
        if self.native_width <= 0 or self.native_height <= 0:
            raise RenderError("Watermark PNG dimensions must be positive.")
        if self.margin < 0:
            raise RenderError("Watermark margin must be non-negative.")
        if self.max_width_ratio <= 0:
            raise RenderError("Watermark max width ratio must be positive.")
        if not 0 <= self.opacity <= 1:
            raise RenderError("Watermark opacity must be between 0 and 1.")


@dataclass(frozen=True)
class WatermarkPlacement:
    """Resolved watermark dimensions and bottom-center position."""

    x: int
    y: int
    width: int
    height: int


def build_ffmpeg_command(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_temp_dir: Path | None = None,
    watermark: Watermark | None = None,
    render_settings: FFmpegRenderSettings = DEFAULT_FFMPEG_RENDER_SETTINGS,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build an FFmpeg argv list that renders one layout to an MP4."""

    output_size = layout.output
    render_settings = render_settings.normalized()
    caption_segments = caption_metadata.segments if caption_metadata is not None else ()
    ass_path = ass_subtitle_path(
        output_path,
        ass_temp_dir=ass_temp_dir,
        caption_segments=caption_segments,
        caption_renderer_backend=caption_renderer_backend,
    )
    filter_complex = build_filter_complex(
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        caption_renderer_backend=caption_renderer_backend,
        ass_subtitle_path=ass_path,
        watermark=watermark,
    )

    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(source_path),
    ]
    if watermark is not None:
        command.extend(("-i", str(watermark.path)))
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-map",
            "0:a?",
            "-c:v",
            render_settings.encoder,
            *_video_encoder_args(render_settings),
            *_thread_args(render_settings),
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            "-s",
            f"{output_size.width}x{output_size.height}",
            "-f",
            "mp4",
            str(output_path),
        ]
    )
    return command


def build_filter_complex(
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_subtitle_path: Path | None = None,
    watermark: Watermark | None = None,
) -> str:
    """Build the FFmpeg filter graph for a layout."""

    if not layout.regions:
        raise RenderError(f"Layout {layout.name!r} must contain at least one region.")

    _require_caption_renderer_backend(caption_renderer_backend)

    output_size = layout.output
    caption_style = _caption_style_for_layout(layout, caption_style)
    caption_segments = caption_metadata.segments if caption_metadata is not None else ()
    needs_watermark = watermark is not None
    filter_parts = [
        f"color=c=black:s={output_size.width}x{output_size.height}:r=30[base]",
        f"[0:v]split={len(layout.regions)}{_split_labels(len(layout.regions))}",
    ]

    overlay_input = "[base]"
    for index, region in enumerate(layout.regions):
        region_label = f"region{index}"
        if index == len(layout.regions) - 1:
            if caption_segments:
                composed_label = "captionbase"
            elif needs_watermark:
                composed_label = "watermarkbase"
            else:
                composed_label = "out"
        else:
            composed_label = f"composed{index}"
        output_rect = rect_to_pixels(region.output_region, output_size)

        filter_parts.append(
            f"[src{index}]{_region_filter(region, output_rect)}[{region_label}]"
        )
        filter_parts.append(
            f"{overlay_input}[{region_label}]overlay="
            f"{output_rect.x}:{output_rect.y}:format=auto:shortest=1[{composed_label}]"
        )
        overlay_input = f"[{composed_label}]"

    if caption_segments:
        filter_parts.extend(
            _caption_filters(
                caption_segments,
                caption_style=caption_style,
                output_size=output_size,
                caption_renderer_backend=caption_renderer_backend,
                ass_subtitle_path=ass_subtitle_path,
                input_label="captionbase",
                output_label="watermarkbase" if needs_watermark else "out",
            )
        )

    if watermark is not None:
        filter_parts.extend(
            _watermark_filters(
                watermark,
                output_size=output_size,
                input_label="watermarkbase",
                output_label="out",
            )
        )

    return ";".join(filter_parts)


def rect_to_pixels(rect: NormalizedRect, output_size: OutputSize) -> PixelRect:
    """Convert a normalized output rectangle to integer pixel coordinates."""

    x = round(rect.x * output_size.width)
    y = round(rect.y * output_size.height)
    width = round(rect.width * output_size.width)
    height = round(rect.height * output_size.height)

    if x + width > output_size.width:
        width = output_size.width - x
    if y + height > output_size.height:
        height = output_size.height - y

    if width <= 0 or height <= 0:
        raise RenderError("Output region resolved to an empty pixel rectangle.")

    return PixelRect(x=x, y=y, width=width, height=height)


def run_ffmpeg_command(command: list[str]) -> None:
    """Run an FFmpeg argv list and raise clear errors for common failures."""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        binary = command[0] if command else "ffmpeg"
        raise RenderError(
            f"{binary} was not found. Install FFmpeg and make sure it is available in PATH."
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise FFmpegCommandError(completed.returncode, stderr)


def render_layout(
    source_path: Path,
    output_path: Path,
    layout: Layout,
    *,
    caption_metadata: CaptionMetadata | None = None,
    caption_style: CaptionStyle = DEFAULT_CAPTION_STYLE,
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND,
    ass_temp_dir: Path | None = None,
    watermark: Watermark | None = None,
    render_settings: FFmpegRenderSettings = DEFAULT_FFMPEG_RENDER_SETTINGS,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Render one layout and return the output path."""

    if not source_path.is_file():
        raise RenderError(f"Source video not found: {source_path}")

    render_settings = render_settings.normalized()
    command = build_ffmpeg_command(
        source_path,
        output_path,
        layout,
        caption_metadata=caption_metadata,
        caption_style=caption_style,
        caption_renderer_backend=caption_renderer_backend,
        ass_temp_dir=ass_temp_dir,
        watermark=watermark,
        render_settings=render_settings,
        ffmpeg_binary=ffmpeg_binary,
    )
    start_time = time.perf_counter()
    rendered = False
    _log_render_settings(layout, output_path, render_settings)
    try:
        run_ffmpeg_command(command)
        rendered = True
    except FFmpegCommandError as exc:
        if _can_retry_with_software_encoder(render_settings, exc):
            fallback_settings = _software_fallback_settings(render_settings)
            LOGGER.warning(
                "FFmpeg encoder %s failed for layout %s; retrying with %s. "
                "output=%s",
                render_settings.encoder,
                layout.name,
                fallback_settings.encoder,
                output_path,
            )
            fallback_command = build_ffmpeg_command(
                source_path,
                output_path,
                layout,
                caption_metadata=caption_metadata,
                caption_style=caption_style,
                caption_renderer_backend=caption_renderer_backend,
                ass_temp_dir=ass_temp_dir,
                watermark=watermark,
                render_settings=fallback_settings,
                ffmpeg_binary=ffmpeg_binary,
            )
            _log_render_settings(layout, output_path, fallback_settings)
            try:
                run_ffmpeg_command(fallback_command)
                rendered = True
            except RenderError as fallback_exc:
                raise RenderError(
                    f"Could not render layout {layout.name!r} from {source_path} "
                    f"to {output_path}: {fallback_exc}"
                ) from fallback_exc
        else:
            raise RenderError(
                f"Could not render layout {layout.name!r} from {source_path} "
                f"to {output_path}: {exc}"
            ) from exc
    except RenderError as exc:
        raise RenderError(
            f"Could not render layout {layout.name!r} from {source_path} "
            f"to {output_path}: {exc}"
        ) from exc
    finally:
        if rendered:
            elapsed_seconds = time.perf_counter() - start_time
            LOGGER.info(
                "Rendered layout %s in %.2fs. output=%s",
                layout.name,
                elapsed_seconds,
                output_path,
            )

    return output_path


def streamer_watermark_env_key(channel_name: str) -> str:
    """Return the env var key used for a streamer's watermark."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", channel_name.strip()).strip("_")
    if not normalized:
        raise RenderError("Streamer channel name is required for watermark lookup.")
    return f"{normalized.upper()}_WATERMARK"


def streamer_watermark_path(
    channel_name: str,
    *,
    environ: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
) -> Path | None:
    """Return the configured watermark path for a streamer, if one exists."""

    env = environ if environ is not None else os.environ
    env_key = streamer_watermark_env_key(channel_name)
    value = env.get(env_key)
    if value is None or not value.strip():
        return None

    path = Path(value.strip()).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path


def load_streamer_watermark(
    channel_name: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
    margin: int = DEFAULT_WATERMARK_MARGIN,
    max_width_ratio: float = DEFAULT_WATERMARK_MAX_WIDTH_RATIO,
    opacity: float = DEFAULT_WATERMARK_OPACITY,
) -> Watermark | None:
    """Load a configured streamer PNG watermark, or return None when absent."""

    if channel_name is None or not channel_name.strip():
        return None

    watermark_path = streamer_watermark_path(
        channel_name,
        environ=environ,
        base_dir=base_dir,
    )
    if watermark_path is None:
        return None
    if not watermark_path.is_file():
        env_key = streamer_watermark_env_key(channel_name)
        raise RenderError(
            f"Configured watermark file for {env_key} was not found: {watermark_path}"
        )

    width, height = _png_dimensions(watermark_path)
    return Watermark(
        path=watermark_path,
        native_width=width,
        native_height=height,
        margin=margin,
        max_width_ratio=max_width_ratio,
        opacity=opacity,
    )


def watermark_placement(
    watermark: Watermark,
    output_size: OutputSize,
) -> WatermarkPlacement:
    """Resolve watermark size and bottom-center position for an output frame."""

    max_width = max(1, round(output_size.width * watermark.max_width_ratio))
    target_width = min(watermark.native_width, max_width, output_size.width)
    target_height = max(
        1,
        round(watermark.native_height * target_width / watermark.native_width),
    )

    if target_height > output_size.height:
        target_height = output_size.height
        target_width = max(
            1,
            round(watermark.native_width * target_height / watermark.native_height),
        )

    x = max(0, (output_size.width - target_width) // 2)
    y = max(0, output_size.height - target_height - watermark.margin)
    return WatermarkPlacement(x=x, y=y, width=target_width, height=target_height)


def _split_labels(count: int) -> str:
    return "".join(f"[src{index}]" for index in range(count))


def _region_filter(region: LayoutRegion, output_rect: PixelRect) -> str:
    source = region.source_region
    filter_chain = (
        f"crop=iw*{_fmt(source.width)}:ih*{_fmt(source.height)}:"
        f"iw*{_fmt(source.x)}:ih*{_fmt(source.y)},"
        f"scale={output_rect.width}:{output_rect.height}:"
        "force_original_aspect_ratio=increase,"
        f"crop={output_rect.width}:{output_rect.height},"
        "setsar=1"
    )
    if region.effect == "blur":
        filter_chain += ",boxblur=20:1"
    return filter_chain


def _video_encoder_args(render_settings: FFmpegRenderSettings) -> tuple[str, ...]:
    if render_settings.encoder == NVENC_H264_ENCODER:
        args: list[str] = []
        if render_settings.preset is not None:
            args.extend(("-preset", render_settings.preset))
        quality = render_settings.quality
        if quality is None:
            quality = render_settings.crf
        if quality is not None:
            args.extend(("-rc", "vbr", "-cq", str(quality)))
        return tuple(args)

    args = []
    if render_settings.preset is not None:
        args.extend(("-preset", render_settings.preset))
    if render_settings.crf is not None:
        args.extend(("-crf", str(render_settings.crf)))
    return tuple(args)


def _thread_args(render_settings: FFmpegRenderSettings) -> tuple[str, ...]:
    if render_settings.threads is None:
        return ()
    return ("-threads", str(render_settings.threads))


def _log_render_settings(
    layout: Layout,
    output_path: Path,
    render_settings: FFmpegRenderSettings,
) -> None:
    LOGGER.info(
        "Rendering layout %s with FFmpeg encoder=%s preset=%s crf=%s quality=%s "
        "threads=%s size=%sx%s. output=%s",
        layout.name,
        render_settings.encoder,
        render_settings.preset,
        render_settings.crf,
        render_settings.quality,
        render_settings.threads,
        layout.output.width,
        layout.output.height,
        output_path,
    )


def _can_retry_with_software_encoder(
    render_settings: FFmpegRenderSettings,
    error: FFmpegCommandError,
) -> bool:
    if (
        render_settings.encoder != NVENC_H264_ENCODER
        or not render_settings.fallback_to_software
    ):
        return False
    stderr = error.stderr.lower()
    return any(
        marker in stderr
        for marker in (
            "unknown encoder",
            "no capable devices found",
            "cannot load",
            "failed loading",
            "error initializing output stream",
            "device creation failed",
            "provided device doesn't support",
        )
    )


def _software_fallback_settings(
    render_settings: FFmpegRenderSettings,
) -> FFmpegRenderSettings:
    preset = render_settings.preset
    if preset is not None and re.fullmatch(r"p[1-7]", preset.lower()):
        preset = DEFAULT_X264_PRESET
    return replace(render_settings, encoder="libx264", preset=preset)


def _watermark_filters(
    watermark: Watermark,
    *,
    output_size: OutputSize,
    input_label: str,
    output_label: str,
) -> tuple[str, str]:
    placement = watermark_placement(watermark, output_size)
    opacity_filter = (
        ""
        if watermark.opacity == DEFAULT_WATERMARK_OPACITY
        else f",colorchannelmixer=aa={_fmt(watermark.opacity)}"
    )
    return (
        f"[1:v]scale={placement.width}:{placement.height},format=rgba"
        f"{opacity_filter}[watermark]",
        f"[{input_label}][watermark]overlay="
        f"{placement.x}:{placement.y}:format=auto:"
        f"eof_action=repeat:repeatlast=1[{output_label}]",
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:24]
    except OSError as exc:
        raise RenderError(f"Could not read configured watermark PNG: {path}") from exc

    if len(header) < 24 or not header.startswith(PNG_SIGNATURE):
        raise RenderError(f"Configured watermark file is not a valid PNG: {path}")
    if header[12:16] != b"IHDR":
        raise RenderError(f"Configured watermark PNG has an invalid IHDR chunk: {path}")

    width = int.from_bytes(header[16:20], byteorder="big")
    height = int.from_bytes(header[20:24], byteorder="big")
    if width <= 0 or height <= 0:
        raise RenderError(f"Configured watermark PNG has invalid dimensions: {path}")
    return width, height


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def _caption_filters(
    segments: tuple,
    *,
    caption_style: CaptionStyle,
    output_size: OutputSize,
    caption_renderer_backend: str,
    ass_subtitle_path: Path | None,
    input_label: str,
    output_label: str,
) -> tuple[str, ...]:
    try:
        return caption_filter_parts(
            segments,
            caption_style=caption_style,
            output_size=output_size,
            caption_renderer_backend=caption_renderer_backend,
            ass_subtitle_path=ass_subtitle_path,
            input_label=input_label,
            output_label=output_label,
            logger=LOGGER,
        )
    except CaptionRenderingError as exc:
        raise RenderError(str(exc)) from exc


def _require_caption_renderer_backend(caption_renderer_backend: str) -> None:
    try:
        require_caption_renderer_backend(caption_renderer_backend)
    except CaptionRenderingError as exc:
        raise RenderError(str(exc)) from exc


def _caption_style_for_layout(
    layout: Layout,
    caption_style: CaptionStyle,
) -> CaptionStyle:
    if layout.caption_region is None or caption_style.vertical_safe_area is not None:
        return caption_style

    caption_rect = rect_to_pixels(layout.caption_region, layout.output)
    return replace(
        caption_style,
        vertical_safe_area=CaptionVerticalSafeArea(
            top=caption_rect.y,
            bottom=layout.output.height - caption_rect.y - caption_rect.height,
            center=True,
        ),
    )
