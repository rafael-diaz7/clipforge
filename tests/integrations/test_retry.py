from __future__ import annotations

import pytest

from clipforge.integrations.retry import RetryDecision, RetryPolicy, retry_call


class RetryableError(RuntimeError):
    pass


class FatalError(RuntimeError):
    pass


def test_retry_call_succeeds_on_first_attempt() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = retry_call(
        operation_name="test operation",
        provider="TestProvider",
        operation=operation,
        policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            jitter_seconds=0.0,
        ),
        classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert calls == 1
    assert sleeps == []


def test_retry_call_retries_once_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("try again")
        return "ok"

    result = retry_call(
        operation_name="test operation",
        provider="TestProvider",
        operation=operation,
        policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            jitter_seconds=0.0,
        ),
        classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert calls == 2
    assert sleeps == [1.0]


def test_retry_call_raises_final_error_after_max_attempts() -> None:
    calls = 0
    sleeps: list[float] = []
    final_error = RetryableError("still failing")

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise final_error

    with pytest.raises(RetryableError) as exc_info:
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=operation,
            policy=RetryPolicy(
                max_attempts=3,
                base_delay_seconds=1.0,
                max_delay_seconds=5.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
            sleep=sleeps.append,
        )

    assert exc_info.value is final_error
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_retry_call_raises_non_retryable_error_immediately() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise FatalError("do not retry")

    with pytest.raises(FatalError):
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=operation,
            policy=RetryPolicy(
                max_attempts=3,
                base_delay_seconds=1.0,
                max_delay_seconds=5.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(
                retryable=isinstance(exc, RetryableError),
                reason="test",
            ),
            sleep=sleeps.append,
        )

    assert calls == 1
    assert sleeps == []


def test_retry_call_exponential_delay_increases() -> None:
    sleeps: list[float] = []

    with pytest.raises(RetryableError):
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=lambda: (_ for _ in ()).throw(RetryableError("again")),
            policy=RetryPolicy(
                max_attempts=4,
                base_delay_seconds=1.0,
                max_delay_seconds=20.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
            sleep=sleeps.append,
        )

    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_call_delay_is_capped() -> None:
    sleeps: list[float] = []

    with pytest.raises(RetryableError):
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=lambda: (_ for _ in ()).throw(RetryableError("again")),
            policy=RetryPolicy(
                max_attempts=4,
                base_delay_seconds=5.0,
                max_delay_seconds=6.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
            sleep=sleeps.append,
        )

    assert sleeps == [5.0, 6.0, 6.0]


def test_retry_call_respects_delay_override() -> None:
    sleeps: list[float] = []

    with pytest.raises(RetryableError):
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=lambda: (_ for _ in ()).throw(RetryableError("again")),
            policy=RetryPolicy(
                max_attempts=2,
                base_delay_seconds=1.0,
                max_delay_seconds=2.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(
                retryable=True,
                reason="test",
                delay_override_seconds=7.5,
            ),
            sleep=sleeps.append,
        )

    assert sleeps == [7.5]


def test_retry_call_uses_injected_sleep() -> None:
    sleeps: list[float] = []
    sleep_calls = 0

    def fake_sleep(delay_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        sleeps.append(delay_seconds)

    with pytest.raises(RetryableError):
        retry_call(
            operation_name="test operation",
            provider="TestProvider",
            operation=lambda: (_ for _ in ()).throw(RetryableError("again")),
            policy=RetryPolicy(
                max_attempts=2,
                base_delay_seconds=0.5,
                max_delay_seconds=2.0,
                jitter_seconds=0.0,
            ),
            classify_error=lambda exc: RetryDecision(retryable=True, reason="test"),
            sleep=fake_sleep,
        )

    assert sleep_calls == 1
    assert sleeps == [0.5]
