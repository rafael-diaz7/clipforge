from pathlib import Path

import pytest

from clipforge.config import (
    ClipforgeConfig,
    ConfigError,
    DOWNLOADS_DIR,
    EXAMPLE_LAYOUTS_DIR,
    METADATA_DIR,
    PROJECT_ROOT,
    RENDERS_DIR,
    load_config,
)


def test_load_config_uses_env_for_clipr_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIPR_API_KEY", "test-key")

    config = load_config()

    assert config.clipr_api_key == "test-key"


def test_config_defines_project_paths_with_pathlib() -> None:
    config = ClipforgeConfig(clipr_api_key=None)

    assert config.project_root == PROJECT_ROOT
    assert config.downloads_dir == DOWNLOADS_DIR
    assert config.renders_dir == RENDERS_DIR
    assert config.metadata_dir == METADATA_DIR
    assert config.example_layouts_dir == EXAMPLE_LAYOUTS_DIR
    assert all(
        isinstance(path, Path)
        for path in (
            config.project_root,
            config.downloads_dir,
            config.renders_dir,
            config.metadata_dir,
            config.example_layouts_dir,
        )
    )


def test_config_defaults_to_vertical_short_resolution() -> None:
    config = ClipforgeConfig(clipr_api_key=None)

    assert config.target_width == 1080
    assert config.target_height == 1920
    assert config.target_resolution == (1080, 1920)
    assert config.output_format == "mp4"


def test_require_clipr_api_key_reports_missing_config() -> None:
    config = ClipforgeConfig(clipr_api_key=None)

    with pytest.raises(ConfigError, match="CLIPR_API_KEY"):
        config.require_clipr_api_key()
