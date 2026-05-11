from pathlib import Path

import pytest

from clipforge.core.config import (
    ClipforgeConfig,
    ConfigError,
    DEFAULT_DOWNLOADER_BACKEND,
    DEFAULT_CAPTION_RENDERER_BACKEND,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    ANALYSIS_DIR,
    DOWNLOADS_DIR,
    EXAMPLE_LAYOUTS_DIR,
    METADATA_DIR,
    PROJECT_ROOT,
    RENDERS_DIR,
    STATE_DB_PATH,
    load_config,
)
from clipforge.media.render_settings import (
    DEFAULT_FFMPEG_RENDER_SETTINGS,
    FFmpegRenderSettings,
)


def test_load_config_uses_env_for_clipr_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIPR_API_KEY", "test-key")

    config = load_config(load_dotenv_file=False)

    assert config.clipr_api_key == "test-key"


def test_load_config_uses_env_for_twitch_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWITCH_CLIENT_ID", "client-id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "client-secret")

    config = load_config(load_dotenv_file=False)

    assert config.twitch_client_id == "client-id"
    assert config.twitch_client_secret == "client-secret"
    assert config.require_twitch_credentials() == ("client-id", "client-secret")


def test_config_requires_twitch_credentials() -> None:
    config = ClipforgeConfig(twitch_client_id="client-id")

    with pytest.raises(ConfigError, match="TWITCH_CLIENT_SECRET"):
        config.require_twitch_credentials()


def test_load_config_uses_env_for_openai_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-test")
    monkeypatch.setenv("CLIPFORGE_GENERATE_CAPTIONS", "true")

    config = load_config(load_dotenv_file=False)

    assert config.openai_api_key == "openai-key"
    assert config.openai_transcription_model == "whisper-test"
    assert config.generate_captions is True
    assert config.require_openai_api_key() == "openai-key"
    assert config.require_openai_transcription_model() == "whisper-test"


def test_load_config_defaults_openai_transcription_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)
    monkeypatch.delenv("CLIPFORGE_GENERATE_CAPTIONS", raising=False)

    config = load_config(load_dotenv_file=False)

    assert config.openai_transcription_model == DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    assert config.generate_captions is False


def test_load_config_uses_env_for_caption_font_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_CAPTION_FONT_FILE", "C:/Windows/Fonts/arial.ttf")

    config = load_config(load_dotenv_file=False)

    assert config.caption_font_file == Path("C:/Windows/Fonts/arial.ttf")


def test_load_config_uses_env_for_caption_renderer_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_CAPTION_RENDERER", "ASS")
    monkeypatch.setenv("CLIPFORGE_ASS_TEMP_DIR", "data/tmp/ass")
    monkeypatch.setenv("CLIPFORGE_CAPTION_FONT_FALLBACKS", "Inter, Segoe UI Emoji")

    config = load_config(load_dotenv_file=False)

    assert config.require_caption_renderer_backend() == "ass"
    assert config.ass_temp_dir == Path("data/tmp/ass")
    assert config.caption_font_fallbacks == ("Inter", "Segoe UI Emoji")


def test_load_config_defaults_caption_renderer_to_drawtext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_CAPTION_RENDERER", raising=False)
    monkeypatch.delenv("CLIPFORGE_CAPTION_FONT_FALLBACKS", raising=False)

    config = load_config(load_dotenv_file=False)

    assert config.caption_renderer_backend == DEFAULT_CAPTION_RENDERER_BACKEND
    assert config.require_caption_renderer_backend() == "drawtext"
    assert config.caption_font_fallbacks == ("Arial",)


def test_load_config_defaults_ffmpeg_render_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_FFMPEG_ENCODER", raising=False)
    monkeypatch.delenv("CLIPFORGE_REVIEW_FAST_RENDER", raising=False)
    monkeypatch.delenv("CLIPFORGE_REVIEW_FFMPEG_ENCODER", raising=False)

    config = load_config(load_dotenv_file=False)

    assert config.ffmpeg_render_settings == DEFAULT_FFMPEG_RENDER_SETTINGS
    assert config.render_settings_for(review=False) == DEFAULT_FFMPEG_RENDER_SETTINGS
    assert config.render_settings_for(review=True) == DEFAULT_FFMPEG_RENDER_SETTINGS


def test_load_config_uses_env_for_ffmpeg_render_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_FFMPEG_ENCODER", "libx264")
    monkeypatch.setenv("CLIPFORGE_FFMPEG_PRESET", "slow")
    monkeypatch.setenv("CLIPFORGE_FFMPEG_CRF", "20")
    monkeypatch.setenv("CLIPFORGE_FFMPEG_THREADS", "4")

    config = load_config(load_dotenv_file=False)

    assert config.render_settings_for(review=False) == FFmpegRenderSettings(
        encoder="libx264",
        preset="slow",
        crf=20,
        threads=4,
    )


def test_load_config_can_select_nvenc_for_review_renders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_REVIEW_FFMPEG_ENCODER", "h264_nvenc")
    monkeypatch.setenv("CLIPFORGE_REVIEW_FFMPEG_PRESET", "p4")
    monkeypatch.setenv("CLIPFORGE_REVIEW_FFMPEG_QUALITY", "24")

    config = load_config(load_dotenv_file=False)

    assert config.render_settings_for(review=False).encoder == "libx264"
    assert config.render_settings_for(review=True) == FFmpegRenderSettings(
        encoder="h264_nvenc",
        preset="p4",
        crf=23,
        quality=24,
        threads=0,
    )


def test_load_config_fast_review_mode_only_changes_review_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_REVIEW_FAST_RENDER", "true")

    config = load_config(load_dotenv_file=False)

    assert config.render_settings_for(review=False) == DEFAULT_FFMPEG_RENDER_SETTINGS
    assert config.render_settings_for(review=True) == FFmpegRenderSettings(
        encoder="libx264",
        preset="veryfast",
        crf=23,
        threads=0,
    )


def test_load_config_rejects_invalid_caption_renderer_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_CAPTION_RENDERER", "missing")

    with pytest.raises(ConfigError, match="Invalid caption renderer backend"):
        load_config(load_dotenv_file=False)


def test_config_requires_openai_api_key() -> None:
    config = ClipforgeConfig()

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        config.require_openai_api_key()


def test_load_config_rejects_invalid_generate_captions_bool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_GENERATE_CAPTIONS", "sometimes")

    with pytest.raises(ConfigError, match="CLIPFORGE_GENERATE_CAPTIONS"):
        load_config(load_dotenv_file=False)


def test_load_config_defaults_to_ytdlp_downloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_DOWNLOADER", raising=False)

    config = load_config(load_dotenv_file=False)

    assert config.downloader_backend == DEFAULT_DOWNLOADER_BACKEND
    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_defaults_to_ytdlp_even_with_clipr_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPFORGE_DOWNLOADER", raising=False)
    monkeypatch.setenv("CLIPR_API_KEY", "test-key")

    config = load_config(load_dotenv_file=False)

    assert config.clipr_api_key == "test-key"
    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_uses_env_for_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "CLIPR")

    config = load_config(load_dotenv_file=False)

    assert config.require_downloader_backend() == "clipr"


def test_load_config_accepts_ytdlp_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "YTDLP")

    config = load_config(load_dotenv_file=False)

    assert config.require_downloader_backend() == "ytdlp"


def test_load_config_rejects_invalid_downloader_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFORGE_DOWNLOADER", "missing")

    with pytest.raises(ConfigError, match="Invalid downloader backend"):
        load_config(load_dotenv_file=False)


def test_config_defines_project_paths_with_pathlib() -> None:
    config = ClipforgeConfig()

    assert config.project_root == PROJECT_ROOT
    assert config.downloads_dir == DOWNLOADS_DIR
    assert config.renders_dir == RENDERS_DIR
    assert config.metadata_dir == METADATA_DIR
    assert config.analysis_dir == ANALYSIS_DIR
    assert config.state_db_path == STATE_DB_PATH
    assert config.example_layouts_dir == EXAMPLE_LAYOUTS_DIR
    assert all(
        isinstance(path, Path)
        for path in (
            config.project_root,
            config.downloads_dir,
            config.renders_dir,
            config.metadata_dir,
            config.analysis_dir,
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
