"""Shared configuration for clipforge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from clipforge.utils import require_config_value, require_config_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
RENDERS_DIR = DATA_DIR / "renders"
METADATA_DIR = DATA_DIR / "metadata"
STATE_DIR = DATA_DIR / "state"
STATE_DB_PATH = STATE_DIR / "clipforge.sqlite"
EXAMPLE_LAYOUTS_DIR = PROJECT_ROOT / "examples" / "layouts"

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
OUTPUT_FORMAT = "mp4"
DEFAULT_DOWNLOADER_BACKEND = "ytdlp"
SUPPORTED_DOWNLOADER_BACKENDS = frozenset({DEFAULT_DOWNLOADER_BACKEND, "clipr"})
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigError(RuntimeError):
    """Raised when required clipforge configuration is missing or invalid."""


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
    project_root: Path = PROJECT_ROOT
    downloads_dir: Path = DOWNLOADS_DIR
    renders_dir: Path = RENDERS_DIR
    metadata_dir: Path = METADATA_DIR
    state_db_path: Path = STATE_DB_PATH
    example_layouts_dir: Path = EXAMPLE_LAYOUTS_DIR
    target_width: int = TARGET_WIDTH
    target_height: int = TARGET_HEIGHT
    output_format: str = OUTPUT_FORMAT
    downloader_backend: str = DEFAULT_DOWNLOADER_BACKEND

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


def load_config() -> ClipforgeConfig:
    """Load environment-backed settings for the local project."""

    load_dotenv(PROJECT_ROOT / ".env")
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
        downloader_backend=os.getenv(
            "CLIPFORGE_DOWNLOADER",
            DEFAULT_DOWNLOADER_BACKEND,
        ),
    )

    config.require_downloader_backend()

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
