"""Small stdlib HTTP adapter for phone-friendly clip review."""

from __future__ import annotations

import html
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from clipforge.core.config import ClipforgeConfig
from clipforge.server.review import (
    ApprovedExport,
    ReviewItem,
    ReviewItemNotFound,
    ReviewQueueService,
    ReviewServerError,
    UnsafeExportPath,
    UnsafeRenderPath,
)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes = b""


class ReviewApplication:
    """Route local HTTP requests to review queue operations."""

    def __init__(self, service: ReviewQueueService) -> None:
        self.service = service

    def handle(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> HttpResponse:
        headers = headers or {}
        parsed_target = urlsplit(target)
        path = parsed_target.path
        if method == "GET" and path == "/":
            return self._review_page()
        if method == "GET":
            video_request = _parse_video_path(path)
            if video_request is not None:
                clip_id, layout = video_request
                return self._video_response(
                    clip_id=clip_id,
                    layout=layout,
                    range_header=headers.get("Range"),
                )
            export_request = _parse_export_path(path)
            if export_request is not None:
                return self._export_response(
                    relative_parts=export_request,
                    range_header=headers.get("Range"),
                    disposition=_export_disposition(parsed_target.query),
                )
        if method == "POST" and path in {"/approve", "/skip", "/rerender"}:
            return self._handle_action(path, body)
        return _html_response(HTTPStatus.NOT_FOUND, _error_page("Not Found"))

    def _review_page(self, *, error: str | None = None) -> HttpResponse:
        try:
            item = self.service.next_item()
        except ReviewServerError as exc:
            return _html_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error_page(str(exc)),
            )
        if item is None:
            return _html_response(HTTPStatus.OK, _page(_empty_state(), title="Clipforge Review"))
        return _html_response(
            HTTPStatus.OK,
            _page(_review_item_html(item, error=error), title="Clipforge Review"),
        )

    def _video_response(
        self,
        *,
        clip_id: str,
        layout: str,
        range_header: str | None,
    ) -> HttpResponse:
        try:
            candidate_path = self.service.candidate_path(clip_id=clip_id, layout=layout)
        except UnsafeRenderPath:
            return _html_response(HTTPStatus.FORBIDDEN, _error_page("Forbidden"))
        except ReviewItemNotFound:
            return _html_response(HTTPStatus.NOT_FOUND, _error_page("Not Found"))
        except ReviewServerError as exc:
            return _html_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error_page(str(exc)),
            )
        return _file_response(candidate_path, range_header=range_header)

    def _export_response(
        self,
        *,
        relative_parts: tuple[str, ...],
        range_header: str | None,
        disposition: str,
    ) -> HttpResponse:
        try:
            export_path = self.service.export_file_path(relative_parts=relative_parts)
        except UnsafeExportPath:
            return _html_response(HTTPStatus.FORBIDDEN, _error_page("Forbidden"))
        except ReviewItemNotFound:
            return _html_response(HTTPStatus.NOT_FOUND, _error_page("Not Found"))
        except ReviewServerError as exc:
            return _html_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                _error_page(str(exc)),
            )
        return _file_response(
            export_path,
            range_header=range_header,
            disposition=disposition,
            disposition_filename=export_path.name,
        )

    def _handle_action(self, path: str, body: bytes) -> HttpResponse:
        form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        clip_id = _one(form, "clip_id")
        try:
            if not clip_id:
                raise ReviewServerError("Missing clip_id.")
            if path == "/approve":
                layout = _one(form, "layout")
                if not layout:
                    raise ReviewServerError("Missing layout.")
                approved = self.service.approve(clip_id=clip_id, layout=layout)
                return _html_response(
                    HTTPStatus.OK,
                    _page(_approved_html(approved), title="Clipforge Review"),
                )
            elif path == "/skip":
                self.service.skip(clip_id=clip_id)
            else:
                self.service.mark_needs_rerender(clip_id=clip_id)
        except ReviewServerError as exc:
            return self._review_page(error=str(exc))
        return _redirect("/")


def serve_review_app(
    *,
    host: str,
    port: int,
    config: ClipforgeConfig | None = None,
) -> None:
    """Start the blocking local review HTTP server."""

    app = ReviewApplication(ReviewQueueService(config=config))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._send(app.handle("GET", self.path, headers=dict(self.headers)))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            self._send(
                app.handle("POST", self.path, headers=dict(self.headers), body=body)
            )

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, response: HttpResponse) -> None:
            self.send_response(response.status)
            for name, value in response.headers:
                self.send_header(name, value)
            self.end_headers()
            if response.body:
                self.wfile.write(response.body)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _parse_video_path(path: str) -> tuple[str, str] | None:
    parts = path.split("/")
    if len(parts) != 5 or parts[0] or parts[1] != "clips" or parts[3] != "renders":
        return None
    filename = unquote(parts[4])
    if not filename.endswith(".mp4"):
        return None
    clip_id = unquote(parts[2])
    layout = filename[:-4]
    if "/" in clip_id or "\\" in clip_id or "/" in layout or "\\" in layout:
        return None
    return clip_id, layout


def _parse_export_path(path: str) -> tuple[str, ...] | None:
    parts = path.split("/")
    if len(parts) < 4 or parts[0] or parts[1] != "exports":
        return None
    filename = unquote(parts[-1])
    if not filename.endswith(".mp4"):
        return None
    return tuple(unquote(part) for part in parts[2:])


def _export_disposition(query: str) -> str:
    values = parse_qs(query, keep_blank_values=True)
    if _one(values, "disposition") == "inline" or _one(values, "inline") == "1":
        return "inline"
    return "attachment"


def _review_item_html(item: ReviewItem, *, error: str | None) -> str:
    clip = item.clip
    facts = [
        ("streamer", clip.streamer_login),
        ("score", "" if clip.rank_score is None else f"{clip.rank_score:g}"),
        ("views", "" if clip.view_count is None else str(clip.view_count)),
        (
            "duration",
            "" if clip.duration_seconds is None else f"{clip.duration_seconds:g}s",
        ),
        ("created", clip.created_at),
        ("clip id", clip.clip_id),
    ]
    metadata = "\n".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in facts
        if value
    )
    candidates = "\n".join(_candidate_html(item, candidate.layout) for candidate in item.candidates)
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""
<main>
  <header>
    <p class="eyebrow">Review Queue</p>
    <h1>{html.escape(clip.title or "Untitled clip")}</h1>
    <dl>{metadata}</dl>
  </header>
  {error_html}
  <section class="actions">
    <form method="post" action="/skip">
      <input type="hidden" name="clip_id" value="{html.escape(clip.clip_id)}">
      <button class="secondary" type="submit">Skip</button>
    </form>
    <form method="post" action="/rerender">
      <input type="hidden" name="clip_id" value="{html.escape(clip.clip_id)}">
      <button class="secondary" type="submit">Needs Rerender</button>
    </form>
  </section>
  <section class="candidates">{candidates}</section>
</main>
"""


def _candidate_html(item: ReviewItem, layout: str) -> str:
    clip_id = item.clip.clip_id
    candidate = next(candidate for candidate in item.candidates if candidate.layout == layout)
    resolution = (
        ""
        if candidate.resolution is None
        else f"{candidate.resolution[0]}x{candidate.resolution[1]}"
    )
    source = f"/clips/{quote(clip_id, safe='')}/renders/{quote(layout, safe='')}.mp4"
    details = f"<p>{html.escape(resolution)}</p>" if resolution else ""
    return f"""
<article class="candidate">
  <h2>{html.escape(layout)}</h2>
  {details}
  <video controls preload="metadata" src="{source}"></video>
  <form method="post" action="/approve">
    <input type="hidden" name="clip_id" value="{html.escape(clip_id)}">
    <input type="hidden" name="layout" value="{html.escape(layout)}">
    <button type="submit">Approve This Layout</button>
  </form>
</article>
"""


def _approved_html(approved: ApprovedExport) -> str:
    filename = approved.export_path.name
    download_url = html.escape(approved.download_url)
    view_url = html.escape(f"{approved.download_url}?disposition=inline")
    return f"""
<main class="empty">
  <h1>Export ready</h1>
  <p>{html.escape(filename)}</p>
  <a class="button" href="{download_url}" download>Download MP4</a>
  <a class="button secondary" href="{view_url}">View MP4</a>
  <p><a href="/">Review next clip</a></p>
</main>
"""


def _empty_state() -> str:
    return """
<main class="empty">
  <h1>Review queue is empty</h1>
  <p>Run <code>clipforge clips prepare</code> to add rendered clips for review.</p>
</main>
"""


def _error_page(message: str) -> str:
    return _page(
        f"""
<main class="empty">
  <h1>{html.escape(message)}</h1>
  <p><a href="/">Back to review queue</a></p>
</main>
""",
        title="Clipforge Review",
    )


def _page(content: str, *, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f5f5f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    main {{ width: min(100%, 920px); margin: 0 auto; padding: 16px; }}
    header {{ padding: 10px 0 14px; }}
    h1 {{ font-size: 1.6rem; line-height: 1.2; margin: 0 0 12px; }}
    h2 {{ font-size: 1.05rem; margin: 0 0 6px; }}
    .eyebrow {{ color: #9bd7c9; font-weight: 700; margin: 0 0 6px; }}
    dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0; }}
    dt {{ color: #a8a8a8; font-size: 0.78rem; text-transform: uppercase; }}
    dd {{ margin: 2px 0 0; overflow-wrap: anywhere; }}
    .actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0 18px; }}
    .candidates {{ display: grid; gap: 18px; }}
    .candidate {{ border-top: 1px solid #32343a; padding-top: 14px; }}
    video {{ display: block; width: 100%; max-height: 72vh; background: #050506; }}
    button, .button {{
      display: inline-grid;
      place-items: center;
      width: 100%;
      min-height: 52px;
      margin-top: 10px;
      border: 0;
      border-radius: 8px;
      background: #2fbf8f;
      color: #07110d;
      font: inherit;
      font-weight: 800;
      text-decoration: none;
    }}
    button.secondary {{ background: #2a2d34; color: #f5f5f2; }}
    .button.secondary {{ background: #2a2d34; color: #f5f5f2; }}
    .empty {{ min-height: 100vh; display: grid; align-content: center; }}
    .error {{ padding: 12px; border: 1px solid #b64848; color: #ffd7d7; }}
    a, code {{ color: #9bd7c9; }}
    @media (max-width: 520px) {{
      main {{ padding: 12px; }}
      dl, .actions {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def _file_response(
    path: Path,
    *,
    range_header: str | None,
    disposition: str | None = None,
    disposition_filename: str | None = None,
) -> HttpResponse:
    file_size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    start, end = _parse_range(range_header, file_size=file_size)
    status = HTTPStatus.OK
    headers = [
        ("Content-Type", content_type),
        ("Accept-Ranges", "bytes"),
    ]
    if disposition is not None and disposition_filename is not None:
        headers.append(
            (
                "Content-Disposition",
                f'{disposition}; filename="{_disposition_filename(disposition_filename)}"',
            )
        )
    if start is None or end is None:
        body = path.read_bytes()
        headers.append(("Content-Length", str(file_size)))
    else:
        status = HTTPStatus.PARTIAL_CONTENT
        with path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        headers.extend(
            (
                ("Content-Length", str(len(body))),
                ("Content-Range", f"bytes {start}-{end}/{file_size}"),
            )
        )
    return HttpResponse(status=status, headers=tuple(headers), body=body)


def _disposition_filename(filename: str) -> str:
    return filename.replace("\\", "_").replace("/", "_").replace('"', "_")


def _parse_range(range_header: str | None, *, file_size: int) -> tuple[int | None, int | None]:
    if not range_header or not range_header.startswith("bytes="):
        return None, None
    value = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in value:
        return None, None
    start_text, end_text = value.split("-", 1)
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_length = int(end_text)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
    except ValueError:
        return None, None
    if start < 0 or end < start or start >= file_size:
        return None, None
    return start, min(end, file_size - 1)


def _html_response(status: HTTPStatus, body: str) -> HttpResponse:
    encoded = body.encode("utf-8")
    return HttpResponse(
        status=int(status),
        headers=(
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(encoded))),
        ),
        body=encoded,
    )


def _redirect(location: str) -> HttpResponse:
    return HttpResponse(
        status=int(HTTPStatus.SEE_OTHER),
        headers=(("Location", location), ("Content-Length", "0")),
    )


def _one(form: dict[str, list[str]], key: str) -> str | None:
    values = form.get(key)
    if not values:
        return None
    return values[0]
