"""TTL cleanup for local clipforge artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from clipforge.core.config import ClipforgeConfig, load_config
from clipforge.storage.state import get_mobile_review_clips


DOWNLOAD_TTL = timedelta(hours=24)
RENDER_TTL = timedelta(hours=24)
READY_EXPORT_TTL = timedelta(hours=24)
METADATA_TTL = timedelta(days=7)


@dataclass(frozen=True)
class CleanupResult:
    deleted_files: tuple[Path, ...]
    deleted_dirs: tuple[Path, ...]

    @property
    def file_count(self) -> int:
        return len(self.deleted_files)

    @property
    def dir_count(self) -> int:
        return len(self.deleted_dirs)


@dataclass(frozen=True)
class _CleanupTarget:
    root: Path
    ttl: timedelta
    pattern: str = "*"


@dataclass(frozen=True)
class _ProtectedPaths:
    files: frozenset[Path]
    dirs: frozenset[Path]


def cleanup_local_artifacts(
    *,
    apply: bool,
    config: ClipforgeConfig | None = None,
    now: datetime | None = None,
) -> CleanupResult:
    """Delete expired local artifacts while preserving review-needed files."""

    config = config or load_config()
    current_time = now or datetime.now(UTC)
    protected = _mobile_review_protected_paths(config)

    deleted_files: list[Path] = []
    deleted_dirs: list[Path] = []
    for target in _cleanup_targets(config):
        deleted_files.extend(
            _delete_expired_files(
                target,
                apply=apply,
                now=current_time,
                protected=protected,
            )
        )
        deleted_dirs.extend(
            _delete_empty_dirs(target.root, apply=apply, protected=protected)
        )

    return CleanupResult(
        deleted_files=tuple(deleted_files),
        deleted_dirs=tuple(deleted_dirs),
    )


def _cleanup_targets(config: ClipforgeConfig) -> tuple[_CleanupTarget, ...]:
    return (
        _CleanupTarget(root=config.downloads_dir, ttl=DOWNLOAD_TTL),
        _CleanupTarget(root=config.renders_dir, ttl=RENDER_TTL),
        _CleanupTarget(
            root=config.exports_dir / "ready",
            ttl=READY_EXPORT_TTL,
            pattern="*.mp4",
        ),
        _CleanupTarget(root=config.metadata_dir, ttl=METADATA_TTL, pattern="*.json"),
    )


def _delete_expired_files(
    target: _CleanupTarget,
    *,
    apply: bool,
    now: datetime,
    protected: _ProtectedPaths,
) -> tuple[Path, ...]:
    if not target.root.exists():
        return ()

    cutoff = now - target.ttl
    deleted: list[Path] = []
    for path in sorted(target.root.rglob(target.pattern)):
        if not path.is_file():
            continue
        if _is_protected(path, protected):
            continue
        if _modified_at(path) >= cutoff:
            continue
        if apply:
            path.unlink(missing_ok=True)
        deleted.append(path)
    return tuple(deleted)


def _delete_empty_dirs(
    root: Path,
    *,
    apply: bool,
    protected: _ProtectedPaths,
) -> tuple[Path, ...]:
    if not root.exists():
        return ()

    deleted: list[Path] = []
    dirs = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in dirs:
        if _is_protected(path, protected):
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            if apply:
                path.rmdir()
            deleted.append(path)
        except OSError:
            continue
    return tuple(deleted)


def _mobile_review_protected_paths(config: ClipforgeConfig) -> _ProtectedPaths:
    if not config.state_db_path.exists():
        return _ProtectedPaths(files=frozenset(), dirs=frozenset())

    files: set[Path] = set()
    dirs: set[Path] = set()
    for clip in get_mobile_review_clips(db_path=config.state_db_path):
        if clip.metadata_path is not None:
            files.add(Path(clip.metadata_path).resolve())
        if clip.render_dir is not None:
            dirs.add(Path(clip.render_dir).resolve())
    return _ProtectedPaths(files=frozenset(files), dirs=frozenset(dirs))


def _is_protected(path: Path, protected: _ProtectedPaths) -> bool:
    resolved = path.resolve()
    if resolved in protected.files:
        return True
    return any(
        _is_relative_to(resolved, protected_dir)
        for protected_dir in protected.dirs
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
