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


class ConfigError(RuntimeError):
    """Raised when required clipforge configuration is missing or invalid."""


@dataclass(frozen=True)
class ClipforgeConfig:
    """Runtime settings shared across clipforge modules."""

    clipr_api_key: str | None
    project_root: Path = PROJECT_ROOT
    downloads_dir: Path = DOWNLOADS_DIR
    renders_dir: Path = RENDERS_DIR
    metadata_dir: Path = METADATA_DIR
    example_layouts_dir: Path = EXAMPLE_LAYOUTS_DIR
    target_width: int = TARGET_WIDTH
    target_height: int = TARGET_HEIGHT
    output_format: str = OUTPUT_FORMAT

    @property
    def target_resolution(self) -> tuple[int, int]:
        return (self.target_width, self.target_height)

    def require_clipr_api_key(self) -> str:
        if not self.clipr_api_key:
            raise ConfigError(
                "Missing required configuration: CLIPR_API_KEY. "
                "Set it in your environment or in a local .env file."
            )
        return self.clipr_api_key


def load_config(*, require_clipr_api_key: bool = False) -> ClipforgeConfig:
    """Load environment-backed settings for the local project."""

    load_dotenv(PROJECT_ROOT / ".env")
    config = ClipforgeConfig(clipr_api_key=os.getenv("CLIPR_API_KEY"))

    if require_clipr_api_key:
        config.require_clipr_api_key()

    return config
