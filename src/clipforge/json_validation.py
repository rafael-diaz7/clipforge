"""Small JSON payload validation helpers."""

from __future__ import annotations

from typing import Any


def required_object(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> dict[str, Any]:
    value = required_value(payload, key, context=context, error_cls=error_cls)
    if not isinstance(value, dict):
        raise error_cls(f"{context}.{key} must be an object.")
    return value


def required_list(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> list[Any]:
    value = required_value(payload, key, context=context, error_cls=error_cls)
    if not isinstance(value, list):
        raise error_cls(f"{context}.{key} must be a list.")
    return value


def required_string(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> str:
    value = required_value(payload, key, context=context, error_cls=error_cls)
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{context}.{key} must be a non-empty string.")
    return value


def required_int(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> int:
    value = required_value(payload, key, context=context, error_cls=error_cls)
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_cls(f"{context}.{key} must be an integer.")
    return value


def required_number(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> float:
    value = required_value(payload, key, context=context, error_cls=error_cls)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_cls(f"{context}.{key} must be a number.")
    return float(value)


def required_value(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    error_cls: type[Exception],
) -> Any:
    try:
        return payload[key]
    except KeyError as exc:
        raise error_cls(f"{context}.{key} is required.") from exc
