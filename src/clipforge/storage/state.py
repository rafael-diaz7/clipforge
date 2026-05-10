"""Local SQLite-backed clip processing state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.core.config import STATE_DB_PATH
from clipforge.utils.paths import ensure_directory, utc_timestamp


DEFAULT_STATE_DB_PATH = STATE_DB_PATH

CLIP_STATUSES = frozenset(
    {
        "discovered",
        "queued",
        "downloaded",
        "rendered",
        "approved",
        "selected",
        "exported",
        "posted",
        "skipped",
        "failed",
    }
)
UNPROCESSED_STATUSES = ("discovered", "queued")
REVIEW_EXCLUDED_STATUSES = (
    "approved",
    "selected",
    "exported",
    "posted",
    "skipped",
    "failed",
)

_CLIPS_COLUMNS = (
    "clip_id",
    "url",
    "streamer_login",
    "title",
    "view_count",
    "created_at",
    "duration_seconds",
    "rank_score",
    "rank_breakdown",
    "discovered_at",
    "last_seen_at",
    "status",
    "download_path",
    "metadata_path",
    "render_dir",
    "skip_reason",
    "error_message",
    "selected_render_layout",
    "selected_render_path",
    "export_path",
    "exported_at",
)

_RANKED_CLIPS_ORDER_SQL = """
        ORDER BY rank_score IS NULL ASC, rank_score DESC, discovered_at ASC, clip_id ASC
    """

_CREATE_CLIPS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clips (
  clip_id TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  streamer_login TEXT,
  title TEXT,
  view_count INTEGER,
  created_at TEXT,
  duration_seconds REAL,
  rank_score REAL,
  rank_breakdown TEXT,
  discovered_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'discovered'
    CHECK (status IN (
      'discovered',
      'queued',
      'downloaded',
      'rendered',
      'approved',
      'selected',
      'exported',
      'posted',
      'skipped',
      'failed'
    )),
  download_path TEXT,
  metadata_path TEXT,
  render_dir TEXT,
  skip_reason TEXT,
  error_message TEXT,
  selected_render_layout TEXT,
  selected_render_path TEXT,
  export_path TEXT,
  exported_at TEXT
);
"""


class ClipStateError(RuntimeError):
    """Raised when clip state cannot be read or updated."""


@dataclass(frozen=True)
class ClipState:
    clip_id: str
    url: str
    streamer_login: str | None
    title: str | None
    view_count: int | None
    created_at: str | None
    duration_seconds: float | None
    rank_score: float | None
    rank_breakdown: dict[str, float] | None
    discovered_at: str
    last_seen_at: str
    status: str
    download_path: str | None
    metadata_path: str | None
    render_dir: str | None
    skip_reason: str | None
    error_message: str | None
    selected_render_layout: str | None
    selected_render_path: str | None
    export_path: str | None
    exported_at: str | None


def init_db(db_path: Path | str = DEFAULT_STATE_DB_PATH) -> Path:
    """Create the clip state database if it does not already exist."""

    resolved_path = Path(db_path)
    ensure_directory(resolved_path.parent)

    with _connect(resolved_path) as connection:
        connection.execute(_CREATE_CLIPS_TABLE_SQL)
        _migrate_clips_table(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_clips_status_last_seen
            ON clips(status, last_seen_at);
            """
        )
        _ensure_column(connection, "clips", "created_at", "TEXT")
        _ensure_column(connection, "clips", "rank_score", "REAL")
        _ensure_column(connection, "clips", "rank_breakdown", "TEXT")
        _ensure_column(connection, "clips", "selected_render_layout", "TEXT")
        _ensure_column(connection, "clips", "selected_render_path", "TEXT")
        _ensure_column(connection, "clips", "export_path", "TEXT")
        _ensure_column(connection, "clips", "exported_at", "TEXT")

    return resolved_path


def upsert_discovered_clip(
    *,
    clip_id: str,
    url: str,
    streamer_login: str | None = None,
    title: str | None = None,
    view_count: int | None = None,
    created_at: str | None = None,
    duration_seconds: float | None = None,
    rank_score: float | None = None,
    rank_breakdown: dict[str, float] | None = None,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
    now: str | None = None,
) -> ClipState:
    """Insert or refresh a discovered clip without changing processing status."""

    timestamp = now or utc_timestamp()
    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        connection.execute(
            """
            INSERT INTO clips (
              clip_id,
              url,
              streamer_login,
              title,
              view_count,
              created_at,
              duration_seconds,
              rank_score,
              rank_breakdown,
              discovered_at,
              last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
              url = excluded.url,
              streamer_login = excluded.streamer_login,
              title = excluded.title,
              view_count = excluded.view_count,
              created_at = COALESCE(excluded.created_at, clips.created_at),
              duration_seconds = excluded.duration_seconds,
              rank_score = excluded.rank_score,
              rank_breakdown = excluded.rank_breakdown,
              last_seen_at = excluded.last_seen_at;
            """,
            (
                clip_id,
                url,
                streamer_login,
                title,
                view_count,
                created_at,
                duration_seconds,
                rank_score,
                _serialize_rank_breakdown(rank_breakdown),
                timestamp,
                timestamp,
            ),
        )

    clip = get_clip(clip_id, db_path=resolved_path)
    if clip is None:
        raise ClipStateError(f"Clip state was not written for clip_id: {clip_id}.")
    return clip


def get_unprocessed_clips(
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
    limit: int | None = None,
    streamer_login: str | None = None,
) -> tuple[ClipState, ...]:
    """Return clips that are eligible for automatic processing."""

    resolved_path = init_db(db_path)
    params: list[Any] = list(UNPROCESSED_STATUSES)
    sql = """
        SELECT *
        FROM clips
        WHERE status IN (?, ?)
    """
    if streamer_login is not None:
        sql += " AND LOWER(streamer_login) = LOWER(?)"
        params.append(streamer_login)
    sql += _RANKED_CLIPS_ORDER_SQL
    sql += _limit_clause(limit, label="Unprocessed clip", params=params)
    return _select_clips(db_path=resolved_path, sql=sql, params=params)


def get_review_eligible_clips(
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
    limit: int | None = None,
    streamer_login: str | None = None,
) -> tuple[ClipState, ...]:
    """Return clips eligible for manual final-render review."""

    resolved_path = init_db(db_path)
    params: list[Any] = list(REVIEW_EXCLUDED_STATUSES)
    placeholders = ", ".join("?" for _ in REVIEW_EXCLUDED_STATUSES)
    sql = f"""
        SELECT *
        FROM clips
        WHERE status NOT IN ({placeholders})
    """
    if streamer_login is not None:
        sql += " AND LOWER(streamer_login) = LOWER(?)"
        params.append(streamer_login)
    sql += _RANKED_CLIPS_ORDER_SQL
    sql += _limit_clause(limit, label="Review clip", params=params)
    return _select_clips(db_path=resolved_path, sql=sql, params=params)


def get_persisted_clips(
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
    streamer_login: str | None = None,
) -> tuple[ClipState, ...]:
    """Return persisted clips, optionally scoped to one streamer."""

    resolved_path = init_db(db_path)
    params: list[Any] = []
    sql = """
        SELECT *
        FROM clips
    """
    if streamer_login is not None:
        sql += " WHERE LOWER(streamer_login) = LOWER(?)"
        params.append(streamer_login)
    sql += " ORDER BY clip_id ASC"

    return _select_clips(db_path=resolved_path, sql=sql, params=params)


def _select_clips(
    *,
    db_path: Path,
    sql: str,
    params: list[Any],
) -> tuple[ClipState, ...]:
    with _connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return tuple(_clip_from_row(row) for row in rows)


def _limit_clause(limit: int | None, *, label: str, params: list[Any]) -> str:
    if limit is None:
        return ""
    if limit < 1:
        raise ClipStateError(f"{label} limit must be at least 1.")
    params.append(limit)
    return " LIMIT ?"


def get_clip(
    clip_id: str,
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState | None:
    """Read one clip by ID."""

    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM clips
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
    if row is None:
        return None
    return _clip_from_row(row)


def update_clip_rank(
    clip_id: str,
    *,
    rank_score: float,
    rank_breakdown: dict[str, float],
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Update ranking fields without changing clip processing status."""

    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        cursor = connection.execute(
            """
            UPDATE clips
            SET
              rank_score = ?,
              rank_breakdown = ?
            WHERE clip_id = ?
            """,
            (
                rank_score,
                _serialize_rank_breakdown(rank_breakdown),
                clip_id,
            ),
        )
        if cursor.rowcount == 0:
            raise ClipStateError(f"Clip not found: {clip_id}.")

    clip = get_clip(clip_id, db_path=resolved_path)
    if clip is None:
        raise ClipStateError(f"Clip not found after rank update: {clip_id}.")
    return clip


def mark_clip_downloaded(
    clip_id: str,
    *,
    download_path: Path | str,
    metadata_path: Path | str | None = None,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Mark a clip as downloaded and persist local artifact paths."""

    return _update_clip_status(
        clip_id,
        "downloaded",
        db_path=db_path,
        download_path=str(download_path),
        metadata_path=_optional_path_string(metadata_path),
        clear_skip_reason=True,
        clear_error_message=True,
    )


def mark_clip_rendered(
    clip_id: str,
    *,
    render_dir: Path | str,
    metadata_path: Path | str | None = None,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Mark a clip as rendered and persist local artifact paths."""

    return _update_clip_status(
        clip_id,
        "rendered",
        db_path=db_path,
        render_dir=str(render_dir),
        metadata_path=_optional_path_string(metadata_path),
        clear_skip_reason=True,
        clear_error_message=True,
    )


def mark_clip_skipped(
    clip_id: str,
    *,
    skip_reason: str,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Mark a clip as intentionally skipped."""

    return _update_clip_status(
        clip_id,
        "skipped",
        db_path=db_path,
        skip_reason=skip_reason,
        clear_error_message=True,
    )


def mark_clip_exported(
    clip_id: str,
    *,
    selected_render_layout: str,
    selected_render_path: Path | str,
    export_path: Path | str,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
    exported_at: str | None = None,
) -> ClipState:
    """Mark a clip as exported and persist the selected render metadata."""

    return _update_clip_status(
        clip_id,
        "exported",
        db_path=db_path,
        selected_render_layout=selected_render_layout,
        selected_render_path=str(selected_render_path),
        export_path=str(export_path),
        exported_at=exported_at or utc_timestamp(),
        clear_skip_reason=True,
        clear_error_message=True,
    )


def mark_clip_failed(
    clip_id: str,
    *,
    error_message: str,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Mark a clip as failed."""

    return _update_clip_status(
        clip_id,
        "failed",
        db_path=db_path,
        error_message=error_message,
    )


def reset_clip_to_discovered(
    clip_id: str,
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> ClipState:
    """Reset one clip to discovered and clear processing artifact fields."""

    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        cursor = connection.execute(
            """
            UPDATE clips
            SET
              status = 'discovered',
              download_path = NULL,
              metadata_path = NULL,
              render_dir = NULL,
              skip_reason = NULL,
              error_message = NULL,
              selected_render_layout = NULL,
              selected_render_path = NULL,
              export_path = NULL,
              exported_at = NULL
            WHERE clip_id = ?
            """,
            (clip_id,),
        )
        if cursor.rowcount == 0:
            raise ClipStateError(f"Clip not found: {clip_id}.")

    clip = get_clip(clip_id, db_path=resolved_path)
    if clip is None:
        raise ClipStateError(f"Clip not found after reset: {clip_id}.")
    return clip


def reset_all_clips_to_discovered(
    *,
    db_path: Path | str = DEFAULT_STATE_DB_PATH,
) -> int:
    """Reset every persisted clip to discovered and clear processing artifacts."""

    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        cursor = connection.execute(
            """
            UPDATE clips
            SET
              status = 'discovered',
              download_path = NULL,
              metadata_path = NULL,
              render_dir = NULL,
              skip_reason = NULL,
              error_message = NULL,
              selected_render_layout = NULL,
              selected_render_path = NULL,
              export_path = NULL,
              exported_at = NULL
            """
        )
        return cursor.rowcount


def _update_clip_status(
    clip_id: str,
    status: str,
    *,
    db_path: Path | str,
    download_path: str | None = None,
    metadata_path: str | None = None,
    render_dir: str | None = None,
    skip_reason: str | None = None,
    error_message: str | None = None,
    selected_render_layout: str | None = None,
    selected_render_path: str | None = None,
    export_path: str | None = None,
    exported_at: str | None = None,
    clear_skip_reason: bool = False,
    clear_error_message: bool = False,
) -> ClipState:
    if status not in CLIP_STATUSES:
        raise ClipStateError(f"Unsupported clip status: {status}.")

    resolved_path = init_db(db_path)
    with _connect(resolved_path) as connection:
        cursor = connection.execute(
            """
            UPDATE clips
            SET
              status = ?,
              download_path = COALESCE(?, download_path),
              metadata_path = COALESCE(?, metadata_path),
              render_dir = COALESCE(?, render_dir),
              skip_reason = CASE WHEN ? THEN NULL ELSE COALESCE(?, skip_reason) END,
              error_message = CASE WHEN ? THEN NULL ELSE COALESCE(?, error_message) END,
              selected_render_layout = COALESCE(?, selected_render_layout),
              selected_render_path = COALESCE(?, selected_render_path),
              export_path = COALESCE(?, export_path),
              exported_at = COALESCE(?, exported_at)
            WHERE clip_id = ?
            """,
            (
                status,
                download_path,
                metadata_path,
                render_dir,
                clear_skip_reason,
                skip_reason,
                clear_error_message,
                error_message,
                selected_render_layout,
                selected_render_path,
                export_path,
                exported_at,
                clip_id,
            ),
        )
        if cursor.rowcount == 0:
            raise ClipStateError(f"Clip not found: {clip_id}.")

    clip = get_clip(clip_id, db_path=resolved_path)
    if clip is None:
        raise ClipStateError(f"Clip not found after status update: {clip_id}.")
    return clip


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _migrate_clips_table(connection: sqlite3.Connection) -> None:
    table_sql = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'clips'
        """
    ).fetchone()["sql"]
    if "'exported'" in table_sql and "'selected'" in table_sql:
        return

    existing_columns = _table_columns(connection, "clips")
    connection.execute("ALTER TABLE clips RENAME TO clips_old")
    connection.execute(_CREATE_CLIPS_TABLE_SQL)
    select_values = [
        column if column in existing_columns else "NULL"
        for column in _CLIPS_COLUMNS
    ]
    connection.execute(
        f"""
        INSERT INTO clips ({", ".join(_CLIPS_COLUMNS)})
        SELECT {", ".join(select_values)}
        FROM clips_old
        """
    )
    connection.execute("DROP TABLE clips_old")


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing_columns = _table_columns(connection, table)
    if column not in existing_columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _optional_path_string(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _serialize_rank_breakdown(value: dict[str, float] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _deserialize_rank_breakdown(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        return None
    return {str(key): float(score) for key, score in payload.items()}


def _clip_from_row(row: sqlite3.Row) -> ClipState:
    data = dict(row)
    return ClipState(
        clip_id=data["clip_id"],
        url=data["url"],
        streamer_login=data["streamer_login"],
        title=data["title"],
        view_count=data["view_count"],
        created_at=data["created_at"],
        duration_seconds=data["duration_seconds"],
        rank_score=data["rank_score"],
        rank_breakdown=_deserialize_rank_breakdown(data["rank_breakdown"]),
        discovered_at=data["discovered_at"],
        last_seen_at=data["last_seen_at"],
        status=data["status"],
        download_path=data["download_path"],
        metadata_path=data["metadata_path"],
        render_dir=data["render_dir"],
        skip_reason=data["skip_reason"],
        error_message=data["error_message"],
        selected_render_layout=data["selected_render_layout"],
        selected_render_path=data["selected_render_path"],
        export_path=data["export_path"],
        exported_at=data["exported_at"],
    )
