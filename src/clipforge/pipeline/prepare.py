"""Non-interactive preparation workflow for filling the review queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.integrations.twitch import (
    list_channel_clips,
    twitch_channel_login_from_input,
)
from clipforge.pipeline.discovery import discover_channel_clips
from clipforge.media.layouts import OutputSize
from clipforge.pipeline.metadata import render_candidates_from_metadata
from clipforge.pipeline.state_sync import record_discovered_clips, rerank_persisted_clips
from clipforge.pipeline.workflows import process_clip
from clipforge.storage.state import (
    ClipState,
    get_clip,
    get_prepare_candidate_clips,
    mark_clip_failed,
    mark_clip_mobile_review,
)


class ClipPrepareError(RuntimeError):
    """Raised when clip preparation cannot be started."""


@dataclass(frozen=True)
class PreparedClip:
    clip_id: str
    metadata_path: Path


@dataclass(frozen=True)
class FailedPreparedClip:
    clip_id: str
    error_message: str


@dataclass(frozen=True)
class PrepareResult:
    discovered_count: int
    reranked_count: int
    selected_count: int
    prepared: tuple[PreparedClip, ...]
    failed: tuple[FailedPreparedClip, ...]
    requested_count: int = 0
    exhausted: bool = False
    max_failures_reached: bool = False

    @property
    def rendered_count(self) -> int:
        return len(self.prepared)

    @property
    def attempted_count(self) -> int:
        return len(self.prepared) + len(self.failed)


ProcessClipFn = Callable[..., Path]
MOBILE_REVIEW_OUTPUT_SIZE = OutputSize(width=1080, height=1920)
DEFAULT_MAX_PREPARE_FAILURES = 10
DEFAULT_FAILED_CLIP_RETRY_COOLDOWN_MINUTES = 60


def prepare_streamer_clips(
    *,
    streamer: str,
    count: int = 3,
    generate_captions: bool | None = None,
    force_captions: bool = False,
    clip_ids: Sequence[str] = (),
    started_at: str | None = None,
    ended_at: str | None = None,
    discovery_limit: int | None = None,
    use_generated_layouts: bool = True,
    max_failures: int = DEFAULT_MAX_PREPARE_FAILURES,
    failed_retry_cooldown_minutes: int = DEFAULT_FAILED_CLIP_RETRY_COOLDOWN_MINUTES,
    config: ClipforgeConfig | None = None,
    process_clip_fn: ProcessClipFn | None = None,
) -> PrepareResult:
    """Discover, rank, and render candidates without prompting or exporting."""

    if count < 1:
        raise ClipPrepareError("--count must be at least 1.")
    if max_failures < 1:
        raise ClipPrepareError("--max-failures must be at least 1.")
    if failed_retry_cooldown_minutes < 0:
        raise ClipPrepareError("--failed-retry-cooldown-minutes cannot be negative.")

    config = config or load_config()
    process_clip_fn = process_clip_fn or process_clip
    streamer_login = twitch_channel_login_from_input(streamer)
    discovery = discover_channel_clips(
        streamer,
        config=config,
        limit=discovery_limit,
        started_at=started_at,
        ended_at=ended_at,
        list_clips_fn=list_channel_clips,
    )
    discovered = discovery.clips
    record_discovered_clips(clips=discovered, channel=streamer, config=config)
    reranked_count = rerank_persisted_clips(config=config, channel=streamer)
    requested_count = len(clip_ids) if clip_ids else count

    candidate_clips = _selected_prepare_clips(
        clip_ids=clip_ids,
        streamer_login=streamer_login,
        failed_before=_failed_retry_cutoff(
            cooldown_minutes=failed_retry_cooldown_minutes,
        ),
        config=config,
    )

    prepared, failed = prepare_until_count(
        candidate_clips,
        count=requested_count,
        max_failures=max_failures,
        streamer_login=streamer_login,
        generate_captions=generate_captions,
        force_captions=force_captions,
        use_generated_layouts=use_generated_layouts,
        config=config,
        process_clip_fn=process_clip_fn,
    )
    exhausted = len(prepared) < requested_count and len(failed) < max_failures

    return PrepareResult(
        discovered_count=len(discovered),
        reranked_count=reranked_count,
        selected_count=len(prepared) + len(failed),
        prepared=tuple(prepared),
        failed=tuple(failed),
        requested_count=requested_count,
        exhausted=exhausted,
        max_failures_reached=len(prepared) < requested_count
        and len(failed) >= max_failures,
    )


def prepare_until_count(
    candidates: Iterable[ClipState],
    *,
    count: int,
    max_failures: int,
    streamer_login: str,
    generate_captions: bool | None,
    force_captions: bool,
    use_generated_layouts: bool,
    config: ClipforgeConfig,
    process_clip_fn: ProcessClipFn,
) -> tuple[list[PreparedClip], list[FailedPreparedClip]]:
    prepared: list[PreparedClip] = []
    failed: list[FailedPreparedClip] = []
    attempted_clip_ids: set[str] = set()
    for clip in iter_ranked_candidates(candidates, attempted_clip_ids=attempted_clip_ids):
        if len(prepared) >= count or len(failed) >= max_failures:
            break
        process_kwargs = {
            "candidate_output_size": MOBILE_REVIEW_OUTPUT_SIZE,
            "channel": streamer_login,
            "config": config,
            "print_summary": False,
            "use_generated_layouts": use_generated_layouts,
        }
        if generate_captions is not None:
            process_kwargs["generate_captions"] = generate_captions
        if force_captions:
            process_kwargs["force_captions"] = True

        try:
            metadata_path = process_clip_fn(clip.url, **process_kwargs)
            state = _ensure_rendered_state(clip, metadata_path, config=config)
        except Exception as exc:
            error_message = str(exc)
            mark_clip_failed(
                clip.clip_id,
                error_message=error_message,
                db_path=config.state_db_path,
            )
            failed.append(
                FailedPreparedClip(
                    clip_id=clip.clip_id,
                    error_message=error_message,
                )
            )
            continue

        prepared.append(
            PreparedClip(
                clip_id=clip.clip_id,
                metadata_path=Path(state.metadata_path or metadata_path),
            )
        )
    return prepared, failed


def iter_ranked_candidates(
    candidates: Iterable[ClipState],
    *,
    attempted_clip_ids: set[str],
) -> Iterable[ClipState]:
    for clip in candidates:
        selected = select_next_candidate(clip, attempted_clip_ids=attempted_clip_ids)
        if selected is not None:
            yield selected


def select_next_candidate(
    clip: ClipState,
    *,
    attempted_clip_ids: set[str],
) -> ClipState | None:
    if clip.clip_id in attempted_clip_ids:
        return None
    attempted_clip_ids.add(clip.clip_id)
    return clip


def _selected_prepare_clips(
    *,
    clip_ids: Sequence[str],
    streamer_login: str,
    failed_before: str,
    config: ClipforgeConfig,
) -> tuple[ClipState, ...]:
    if not clip_ids:
        return get_prepare_candidate_clips(
            db_path=config.state_db_path,
            streamer_login=streamer_login,
            failed_before=failed_before,
        )

    clips: list[ClipState] = []
    for clip_id in clip_ids:
        clip = get_clip(clip_id, db_path=config.state_db_path)
        if clip is None:
            raise ClipPrepareError(f"Clip not found after discovery: {clip_id}.")
        _ensure_manual_prepare_clip_is_eligible(
            clip,
            streamer_login=streamer_login,
        )
        clips.append(clip)
    return tuple(clips)


def _ensure_rendered_state(
    clip: ClipState,
    metadata_path: Path,
    *,
    config: ClipforgeConfig,
) -> ClipState:
    state = get_clip(clip.clip_id, db_path=config.state_db_path)
    if (
        state is not None
        and state.status == "mobile_review"
        and state.render_dir is not None
        and state.metadata_path is not None
        and state.selected_render_layout is None
        and state.selected_render_path is None
        and state.export_path is None
        and state.exported_at is None
    ):
        return state

    render_options = render_candidates_from_metadata(metadata_path)
    if not render_options:
        raise ClipPrepareError(f"No render candidates found for clip: {clip.clip_id}.")
    return mark_clip_mobile_review(
        clip.clip_id,
        render_dir=render_options[0].path.parent,
        metadata_path=metadata_path,
        db_path=config.state_db_path,
    )


def _ensure_manual_prepare_clip_is_eligible(
    clip: ClipState,
    *,
    streamer_login: str,
) -> None:
    if (
        clip.streamer_login is not None
        and clip.streamer_login.lower() != streamer_login.lower()
    ):
        raise ClipPrepareError(
            f"Clip {clip.clip_id} belongs to streamer {clip.streamer_login}, "
            f"not {streamer_login}."
        )
    if not _is_normal_prepare_candidate(clip):
        raise ClipPrepareError(
            f"Clip is not preparation-eligible: {clip.clip_id} ({clip.status})."
        )


def _is_normal_prepare_candidate(clip: ClipState) -> bool:
    return clip.status in {"discovered", "queued", "failed"}


def _failed_retry_cutoff(*, cooldown_minutes: int) -> str:
    retry_after = datetime.now(UTC) - timedelta(minutes=cooldown_minutes)
    return retry_after.isoformat()
