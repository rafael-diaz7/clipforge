"""Backend-specific caption escaping helpers."""

from __future__ import annotations


def escape_drawtext_option(value: str) -> str:
    """Escape one FFmpeg filter option value."""

    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def escape_drawtext_text(value: str) -> str:
    """Escape literal text for FFmpeg drawtext."""

    return (
        value.replace("\\", "\\\\")
        .replace(" ", "\\ ")
        .replace("'", "\\\\\\'")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("%", "\\%")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
    )


def escape_ass_text(value: str) -> str:
    """Escape literal text for ASS subtitle dialogue events."""

    return (
        value.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\N")
    )
