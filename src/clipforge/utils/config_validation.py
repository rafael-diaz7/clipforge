"""Small helpers for environment-backed configuration validation."""

from __future__ import annotations

from typing import TypeVar


ErrorT = TypeVar("ErrorT", bound=Exception)


def require_config_value(
    value: str | None,
    name: str,
    *,
    context: str,
    error_cls: type[ErrorT],
) -> str:
    """Return a stripped config value or raise a consistent missing-config error."""

    normalized = (value or "").strip()
    if normalized:
        return normalized

    raise error_cls(
        f"Missing required {context} configuration: {name}. "
        "Set it in your environment or .env file."
    )


def require_config_values(
    values: tuple[tuple[str, str | None], ...],
    *,
    context: str,
    error_cls: type[ErrorT],
) -> tuple[str, ...]:
    """Return stripped config values or raise a grouped missing-config error."""

    normalized_values = tuple((name, (value or "").strip()) for name, value in values)
    missing = tuple(name for name, value in normalized_values if not value)
    if missing:
        raise error_cls(
            f"Missing required {context} configuration: {', '.join(missing)}. "
            "Set them in your environment or .env file."
        )
    return tuple(value for _, value in normalized_values)
