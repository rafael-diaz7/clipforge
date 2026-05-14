"""Generic retry helpers for external API integrations."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    jitter_seconds: float


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    reason: str
    delay_override_seconds: float | None = None


def retry_call(
    *,
    operation_name: str,
    provider: str,
    operation: Callable[[], T],
    policy: RetryPolicy,
    classify_error: Callable[[BaseException], RetryDecision],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an integration operation with provider-specific retry classification."""

    _validate_policy(policy)

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            decision = classify_error(exc)
            if not decision.retryable or attempt >= policy.max_attempts:
                raise

            delay_seconds = _retry_delay_seconds(policy, attempt, decision)
            LOGGER.warning(
                "Retrying %s %s after attempt %s/%s failed: %s "
                "(%s). Sleeping %.2fs before next attempt.",
                provider,
                operation_name,
                attempt,
                policy.max_attempts,
                decision.reason,
                type(exc).__name__,
                delay_seconds,
            )
            sleep(delay_seconds)

    raise RuntimeError("retry_call exhausted attempts without returning or raising.")


def _validate_policy(policy: RetryPolicy) -> None:
    if policy.max_attempts < 1:
        raise ValueError("RetryPolicy.max_attempts must be at least 1.")
    if policy.base_delay_seconds < 0:
        raise ValueError("RetryPolicy.base_delay_seconds must be non-negative.")
    if policy.max_delay_seconds < 0:
        raise ValueError("RetryPolicy.max_delay_seconds must be non-negative.")
    if policy.jitter_seconds < 0:
        raise ValueError("RetryPolicy.jitter_seconds must be non-negative.")


def _retry_delay_seconds(
    policy: RetryPolicy,
    failed_attempt: int,
    decision: RetryDecision,
) -> float:
    if decision.delay_override_seconds is not None:
        return max(0.0, decision.delay_override_seconds)

    exponential_delay = policy.base_delay_seconds * (2 ** (failed_attempt - 1))
    jitter = random.uniform(0.0, policy.jitter_seconds) if policy.jitter_seconds else 0.0
    return min(exponential_delay + jitter, policy.max_delay_seconds)
