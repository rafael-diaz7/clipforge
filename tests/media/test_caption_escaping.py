from clipforge.media.caption_escaping import (
    escape_ass_text,
    escape_drawtext_option,
    escape_drawtext_text,
)


def test_escape_drawtext_text_handles_special_characters() -> None:
    text = "Bob's \"win\", path C:\\clips\\a:b 100% - wow 😀"

    escaped = escape_drawtext_text(text)

    assert "Bob\\\\\\'s" in escaped
    assert '\\"' not in escaped
    assert "\\," in escaped
    assert "\\:" in escaped
    assert "\\%" in escaped
    assert "C\\:\\\\clips\\\\a\\:b" in escaped
    assert "wow\\ 😀" in escaped


def test_escape_drawtext_option_handles_windows_font_paths() -> None:
    path = "C:\\Windows\\Fonts\\O'Hara, Condensed.ttf"

    escaped = escape_drawtext_option(path)

    assert escaped == "C\\:\\\\Windows\\\\Fonts\\\\O\\'Hara\\, Condensed.ttf"


def test_escape_ass_text_handles_special_characters() -> None:
    text = "Bob's \"win\", path C:\\clips\\a:b 100% {tag}\nemoji 😀 - wait…"

    escaped = escape_ass_text(text)

    assert "Bob's \"win\", path" in escaped
    assert "C:\\\\clips\\\\a:b" in escaped
    assert "100%" in escaped
    assert "\\{tag\\}" in escaped
    assert "\\Nemoji 😀 - wait…" in escaped
