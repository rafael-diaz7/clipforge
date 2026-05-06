"""Shared configuration for clipforge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
RENDERS_DIR = DATA_DIR / "renders"
METADATA_DIR = DATA_DIR / "metadata"
EXAMPLE_LAYOUTS_DIR = PROJECT_ROOT / "examples" / "layouts"

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
OUTPUT_FORMAT = "mp4"
DEFAULT_DOWNLOADER_BACKEND = "ytdlp"
SUPPORTED_DOWNLOADER_BACKENDS = frozenset({DEFAULT_DOWNLOADER_BACKEND, "clipr"})


class ConfigError(RuntimeError):
    """Raised when required clipforge configuration is missing or invalid."""


@dataclass(frozen=True)
class ClipforgeConfig:
    """Runtime settings shared across clipforge modules."""

    clipr_api_key: str | None = None
    twitch_client_id: str | None = None
    twitch_client_secret: str | None = None
    project_root: Path = PROJECT_ROOT
    downloads_dir: Path = DOWNLOADS_DIR
    renders_dir: Path = RENDERS_DIR
    metadata_dir: Path = METADATA_DIR
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
        client_id = (self.twitch_client_id or "").strip()
        client_secret = (self.twitch_client_secret or "").strip()
        missing = []
        if not client_id:
            missing.append("TWITCH_CLIENT_ID")
        if not client_secret:
            missing.append("TWITCH_CLIENT_SECRET")
        if missing:
            raise ConfigError(
                "Missing required Twitch API configuration: "
                f"{', '.join(missing)}. Set them in your environment or .env file."
            )
        return client_id, client_secret


def load_config() -> ClipforgeConfig:
    """Load environment-backed settings for the local project."""

    load_dotenv(PROJECT_ROOT / ".env")
    config = ClipforgeConfig(
        clipr_api_key=os.getenv("CLIPR_API_KEY"),
        twitch_client_id=os.getenv("TWITCH_CLIENT_ID"),
        twitch_client_secret=os.getenv("TWITCH_CLIENT_SECRET"),
        downloader_backend=os.getenv(
            "CLIPFORGE_DOWNLOADER",
            DEFAULT_DOWNLOADER_BACKEND,
        ),
    )

    config.require_downloader_backend()

    return config
