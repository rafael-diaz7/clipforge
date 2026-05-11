"""FFmpeg render performance settings."""

from __future__ import annotations

from dataclasses import dataclass, replace


DEFAULT_FFMPEG_ENCODER = "libx264"
NVENC_H264_ENCODER = "h264_nvenc"
SUPPORTED_FFMPEG_ENCODERS = frozenset({DEFAULT_FFMPEG_ENCODER, NVENC_H264_ENCODER})
DEFAULT_X264_PRESET = "medium"
DEFAULT_X264_CRF = 23
DEFAULT_FFMPEG_THREADS = 0
DEFAULT_REVIEW_X264_PRESET = "veryfast"
DEFAULT_REVIEW_NVENC_PRESET = "p4"


@dataclass(frozen=True)
class FFmpegRenderSettings:
    """Encoder and quality settings for one FFmpeg render."""

    encoder: str = DEFAULT_FFMPEG_ENCODER
    preset: str | None = DEFAULT_X264_PRESET
    crf: int | None = DEFAULT_X264_CRF
    quality: int | None = None
    threads: int | None = DEFAULT_FFMPEG_THREADS
    fallback_to_software: bool = True

    def normalized(self) -> "FFmpegRenderSettings":
        encoder = self.encoder.strip().lower()
        preset = self.preset.strip() if self.preset is not None else None
        return replace(self, encoder=encoder, preset=preset or None)


DEFAULT_FFMPEG_RENDER_SETTINGS = FFmpegRenderSettings()
