from pathlib import Path

import pytest

from clipforge.config import (
    ClipforgeConfig,
    ConfigError,
    DEFAULT_DOWNLOADER_BACKEND,
    DOWNLOADS_DIR,
    EXAMPLE_LAYOUTS_DIR,
    METADATA_DIR,
    PROJECT_ROOT,
    RENDERS_DIR,
    STATE_DB_PATH,
    load_config,
)


def test_load_config_uses_env_for_clipr_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIPR_API_KEY", "test-key")

    config = load_config()

    assert config.clipr_api_key == "test-key"


def test_load_config_uses_env_for_twitch_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWITCH_CLIENT_ID", "client-id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "client-secret")

    config = load_config()

    assert config.twitch_client_id == "client-id"
    assert config.twitch_client_secret == "client-secret"
    assert config.require_twitch_credentials() == ("client-id", "client-secret")


def test_config_requires_twitch_credentials() -> None:
    config = ClipforgeConfig(twitch_client_id="client-id")

    with pytest.raises(ConfigError, match="TWITCH_CLIENT_SECRET"):
        config.require_twitch_credentials()


def test_load_config_defaults_to_ytdlp_downloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_DOWNLOADER", raising=False)

    config = load_config()

    assert config.downloader_backend == DEFAULT_DOWNLOADER_BACKEND
    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_defaults_to_ytdlp_even_with_clipr_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_DOWNLOADER", raising=False)
    monkeypatch.setenv("CLIPR_API_KEY", "test-key")

    config = load_config()

    assert config.clipr_api_key == "test-key"
    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_uses_env_for_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "CLIPR")

    config = load_config()

    assert config.require_downloader_backend() == "clipr"


def test_load_config_accepts_ytdlp_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "YTDLP")

    config = load_config()

    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_rejects_invalid_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "missing")

    with pytest.raises(ConfigError, match="Invalid downloader backend"):
        load_config()


def test_config_defines_project_paths_with_pathlib() -> None:
    config = ClipforgeConfig()

    assert config.project_root == PROJECT_ROOT
    assert config.downloads_dir == DOWNLOADS_DIR
    assert config.renders_dir == RENDERS_DIR
    assert config.metadata_dir == METADATA_DIR
    assert config.state_db_path == STATE_DB_PATH
    assert config.example_layouts_dir == EXAMPLE_LAYOUTS_DIR
    assert all(
        isinstance(path, Path)
        for path in (
            config.project_root,
            config.downloads_dir,
            config.renders_dir,
            config.metadata_dir,
            config.state_db_path,
            config.example_layouts_dir,
        )
    )


def test_config_defaults_to_vertical_short_resolution() -> None:
    config = ClipforgeConfig()

    assert config.target_width == 1080
    assert config.target_height == 1920
    assert config.target_resolution == (1080, 1920)
    assert config.output_format == "mp4"


def test_config_clipr_api_key_is_optional() -> None:
    config = ClipforgeConfig()

    assert config.clipr_api_key is None
