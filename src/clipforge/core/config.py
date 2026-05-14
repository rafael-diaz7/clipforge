"""Shared configuration for clipforge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

from clipforge.media.render_settings import (
    DEFAULT_FFMPEG_RENDER_SETTINGS,
    DEFAULT_REVIEW_NVENC_PRESET,
    DEFAULT_REVIEW_X264_PRESET,
    FFmpegRenderSettings,
    NVENC_H264_ENCODER,
    SUPPORTED_FFMPEG_ENCODERS,
)
from clipforge.utils.config_validation import require_config_value, require_config_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
RENDERS_DIR = DATA_DIR / "renders"
METADATA_DIR = DATA_DIR / "metadata"
ANALYSIS_DIR = DATA_DIR / "analysis"
EXPORTS_DIR = DATA_DIR / "exports"
STATE_DIR = DATA_DIR / "state"
STATE_DB_PATH = STATE_DIR / "clipforge.sqlite"
EXAMPLE_LAYOUTS_DIR = PROJECT_ROOT / "examples" / "layouts"

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
OUTPUT_FORMAT = "mp4"
DEFAULT_DOWNLOADER_BACKEND = "ytdlp"
SUPPORTED_DOWNLOADER_BACKENDS = frozenset({DEFAULT_DOWNLOADER_BACKEND, "clipr"})
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_CAPTION_RENDERER_BACKEND = "drawtext"
SUPPORTED_CAPTION_RENDERER_BACKENDS = frozenset(
    {DEFAULT_CAPTION_RENDERER_BACKEND, "ass"}
)
ASS_TEMP_DIR = DATA_DIR / "metadata" / "ass"
DEFAULT_CAPTION_FONT_FALLBACKS = ("Arial",)
MAX_DISCOVERY_WINDOW_LIMIT = 100
DEFAULT_DISCOVERY_WINDOWS_SPEC = "7:100,31:100"
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigError(RuntimeError):
    """Raised when required clipforge configuration is missing or invalid."""


@dataclass(frozen=True)
class ClipDiscoveryWindow:
    """Configured Twitch clip discovery window."""

    days: int
    limit: int


DEFAULT_DISCOVERY_WINDOWS = (
    ClipDiscoveryWindow(days=7, limit=100),
    ClipDiscoveryWindow(days=31, limit=100),
)


@dataclass(frozen=True)
class ClipforgeConfig:
    """Runtime settings shared across clipforge modules."""

    clipr_api_key: str | None = None
    twitch_client_id: str | None = None
    twitch_client_secret: str | None = None
    openai_api_key: str | None = None
    openai_transcription_model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    generate_captions: bool = False
    caption_font_file: Path | None = None
    caption_renderer_backend: str = DEFAULT_CAPTION_RENDERER_BACKEND
    ass_temp_dir: Path = ASS_TEMP_DIR
    caption_font_fallbacks: tuple[str, ...] = DEFAULT_CAPTION_FONT_FALLBACKS
    project_root: Path = PROJECT_ROOT
    downloads_dir: Path = DOWNLOADS_DIR
    renders_dir: Path = RENDERS_DIR
    metadata_dir: Path = METADATA_DIR
    analysis_dir: Path = ANALYSIS_DIR
    exports_dir: Path = EXPORTS_DIR
    state_db_path: Path = STATE_DB_PATH
    example_layouts_dir: Path = EXAMPLE_LAYOUTS_DIR
    target_width: int = TARGET_WIDTH
    target_height: int = TARGET_HEIGHT
    output_format: str = OUTPUT_FORMAT
    downloader_backend: str = DEFAULT_DOWNLOADER_BACKEND
    ffmpeg_render_settings: FFmpegRenderSettings = field(
        default_factory=lambda: DEFAULT_FFMPEG_RENDER_SETTINGS
    )
    review_fast_render: bool = False
    review_ffmpeg_render_settings: FFmpegRenderSettings | None = None
    review_output_width: int | None = None
    discovery_windows: tuple[ClipDiscoveryWindow, ...] = DEFAULT_DISCOVERY_WINDOWS

    @property
    def target_resolution(self) -> tuple[int, int]:
        return (self.target_width, self.target_height)

    def require_downloader_backend(self) -> str:
        backend = self.downloader_backend.strip().lower()
        if backend not in SUPPORTED_DOWNLOADER_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_DOWNLOADER_BACKENDS))
            raise ConfigError(
                "Invalid downloader backend: "
                f"{self.downloader_backend!r}. Supported values: {supported}."
            )
        return backend

    def require_caption_renderer_backend(self) -> str:
        backend = self.caption_renderer_backend.strip().lower()
        if backend not in SUPPORTED_CAPTION_RENDERER_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_CAPTION_RENDERER_BACKENDS))
            raise ConfigError(
                "Invalid caption renderer backend: "
                f"{self.caption_renderer_backend!r}. Supported values: {supported}."
            )
        return backend

    def render_settings_for(self, *, review: bool) -> FFmpegRenderSettings:
        if review and self.review_ffmpeg_render_settings is not None:
            return self.review_ffmpeg_render_settings.normalized()
        return self.ffmpeg_render_settings.normalized()

    def review_resolution_for(self, *, width: int, height: int) -> tuple[int, int]:
        if self.review_output_width is None:
            return (width, height)
        if width <= 0 or height <= 0:
            raise ConfigError("Normal output dimensions must be positive.")
        review_height = round(height * self.review_output_width / width)
        return (self.review_output_width, max(1, review_height))

    def require_twitch_credentials(self) -> tuple[str, str]:
        client_id, client_secret = require_config_values(
            (
                ("TWITCH_CLIENT_ID", self.twitch_client_id),
                ("TWITCH_CLIENT_SECRET", self.twitch_client_secret),
            ),
            context="Twitch API",
            error_cls=ConfigError,
        )
        return client_id, client_secret

    def require_openai_api_key(self) -> str:
        return require_config_value(
            self.openai_api_key,
            "OPENAI_API_KEY",
            context="OpenAI API",
            error_cls=ConfigError,
        )

    def require_openai_transcription_model(self) -> str:
        return require_config_value(
            self.openai_transcription_model,
            "OPENAI_TRANSCRIPTION_MODEL",
            context="OpenAI API",
            error_cls=ConfigError,
        )


def load_config(*, load_dotenv_file: bool = True) -> ClipforgeConfig:
    """Load environment-backed settings for the local project."""

    if load_dotenv_file:
        load_dotenv(PROJECT_ROOT / ".env")
    ffmpeg_render_settings = _ffmpeg_settings_from_env("CLIPFORGE_FFMPEG_")
    review_fast_render = _env_bool("CLIPFORGE_REVIEW_FAST_RENDER", default=False)
    config = ClipforgeConfig(
        clipr_api_key=os.getenv("CLIPR_API_KEY"),
        twitch_client_id=os.getenv("TWITCH_CLIENT_ID"),
        twitch_client_secret=os.getenv("TWITCH_CLIENT_SECRET"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_transcription_model=os.getenv(
            "OPENAI_TRANSCRIPTION_MODEL",
            DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        ),
        generate_captions=_env_bool("CLIPFORGE_GENERATE_CAPTIONS", default=False),
        caption_font_file=_env_path("CLIPFORGE_CAPTION_FONT_FILE"),
        caption_renderer_backend=os.getenv(
            "CLIPFORGE_CAPTION_RENDERER",
            DEFAULT_CAPTION_RENDERER_BACKEND,
        ),
        ass_temp_dir=_env_path("CLIPFORGE_ASS_TEMP_DIR") or ASS_TEMP_DIR,
        caption_font_fallbacks=_env_list(
            "CLIPFORGE_CAPTION_FONT_FALLBACKS",
            default=DEFAULT_CAPTION_FONT_FALLBACKS,
        ),
        downloader_backend=os.getenv(
            "CLIPFORGE_DOWNLOADER",
            DEFAULT_DOWNLOADER_BACKEND,
        ),
        ffmpeg_render_settings=ffmpeg_render_settings,
        review_fast_render=review_fast_render,
        review_ffmpeg_render_settings=_review_ffmpeg_settings_from_env(
            ffmpeg_render_settings,
            fast_render=review_fast_render,
        ),
        review_output_width=_env_int("CLIPFORGE_REVIEW_OUTPUT_WIDTH"),
        discovery_windows=_env_discovery_windows("CLIPFORGE_DISCOVERY_WINDOWS"),
    )

    config.require_downloader_backend()
    config.require_caption_renderer_backend()
    _require_ffmpeg_settings(config.ffmpeg_render_settings)
    if config.review_ffmpeg_render_settings is not None:
        _require_ffmpeg_settings(config.review_ffmpeg_render_settings)
    if config.review_output_width is not None and config.review_output_width <= 0:
        raise ConfigError("CLIPFORGE_REVIEW_OUTPUT_WIDTH must be a positive integer.")
    _require_discovery_windows(config.discovery_windows)

    return config


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return Path(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False

    raise ConfigError(
        f"Invalid boolean configuration for {name}: {value!r}. "
        "Use one of: true, false, 1, 0, yes, no, on, off."
    )


def _env_list(name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default

    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _ffmpeg_settings_from_env(
    prefix: str,
    *,
    base: FFmpegRenderSettings = DEFAULT_FFMPEG_RENDER_SETTINGS,
) -> FFmpegRenderSettings:
    settings = base
    encoder = _env_str(f"{prefix}ENCODER")
    preset = _env_str(f"{prefix}PRESET")
    crf = _env_int(f"{prefix}CRF")
    quality = _env_int(f"{prefix}QUALITY")
    threads = _env_int(f"{prefix}THREADS")

    if encoder is not None:
        settings = replace(settings, encoder=encoder)
    if preset is not None:
        settings = replace(settings, preset=preset)
    if crf is not None:
        settings = replace(settings, crf=crf)
    if quality is not None:
        settings = replace(settings, quality=quality)
    if threads is not None:
        settings = replace(settings, threads=threads)
    return settings.normalized()


def _review_ffmpeg_settings_from_env(
    base: FFmpegRenderSettings,
    *,
    fast_render: bool,
) -> FFmpegRenderSettings | None:
    has_review_override = any(
        _env_str(name) is not None
        for name in (
            "CLIPFORGE_REVIEW_FFMPEG_ENCODER",
            "CLIPFORGE_REVIEW_FFMPEG_PRESET",
            "CLIPFORGE_REVIEW_FFMPEG_CRF",
            "CLIPFORGE_REVIEW_FFMPEG_QUALITY",
            "CLIPFORGE_REVIEW_FFMPEG_THREADS",
        )
    )
    if not fast_render and not has_review_override:
        return None

    settings = _ffmpeg_settings_from_env("CLIPFORGE_REVIEW_FFMPEG_", base=base)
    if fast_render and _env_str("CLIPFORGE_REVIEW_FFMPEG_PRESET") is None:
        fast_preset = (
            DEFAULT_REVIEW_NVENC_PRESET
            if settings.encoder == NVENC_H264_ENCODER
            else DEFAULT_REVIEW_X264_PRESET
        )
        settings = replace(settings, preset=fast_preset)
    return settings.normalized()


def _require_ffmpeg_settings(settings: FFmpegRenderSettings) -> None:
    normalized = settings.normalized()
    if normalized.encoder not in SUPPORTED_FFMPEG_ENCODERS:
        supported = ", ".join(sorted(SUPPORTED_FFMPEG_ENCODERS))
        raise ConfigError(
            "Invalid FFmpeg encoder: "
            f"{settings.encoder!r}. Supported values: {supported}."
        )
    if normalized.crf is not None and normalized.crf < 0:
        raise ConfigError("FFmpeg CRF must be non-negative.")
    if normalized.quality is not None and normalized.quality < 0:
        raise ConfigError("FFmpeg quality must be non-negative.")
    if normalized.threads is not None and normalized.threads < 0:
        raise ConfigError("FFmpeg thread count must be non-negative.")


def _env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_int(name: str) -> int | None:
    value = _env_str(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer configuration for {name}: {value!r}.") from exc


def _env_discovery_windows(name: str) -> tuple[ClipDiscoveryWindow, ...]:
    value = _env_str(name)
    if value is None:
        return DEFAULT_DISCOVERY_WINDOWS

    windows: list[ClipDiscoveryWindow] = []
    for item in value.split(","):
        spec = item.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) != 2:
            raise ConfigError(
                f"Invalid discovery window configuration for {name}: {value!r}. "
                "Use comma-separated days:limit entries, for example "
                f"{DEFAULT_DISCOVERY_WINDOWS_SPEC}."
            )
        days_text, limit_text = parts
        try:
            days = int(days_text)
            limit = int(limit_text)
        except ValueError as exc:
            raise ConfigError(
                f"Invalid discovery window configuration for {name}: {value!r}. "
                "Days and limit must be integers."
            ) from exc
        windows.append(ClipDiscoveryWindow(days=days, limit=limit))

    return tuple(windows)


def _require_discovery_windows(windows: tuple[ClipDiscoveryWindow, ...]) -> None:
    if not windows:
        raise ConfigError("CLIPFORGE_DISCOVERY_WINDOWS must include at least one window.")
    for window in windows:
        if window.days < 1:
            raise ConfigError("Discovery window days must be at least 1.")
        if window.limit < 1 or window.limit > MAX_DISCOVERY_WINDOW_LIMIT:
            raise ConfigError(
                "Discovery window limit must be between "
                f"1 and {MAX_DISCOVERY_WINDOW_LIMIT}."
            )
