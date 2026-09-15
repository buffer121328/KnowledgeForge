"""Bounded shared security-state backend with memory and Redis adapters."""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from shared.utils.metrics import security_state_operations_total

_SAFE_KEY = re.compile(r"^[a-z0-9][a-z0-9:_.-]{0,255}$")


class SecurityStateConfigurationError(RuntimeError):
    """Safe startup error for an invalid production security-state backend."""


class SecurityStateUnavailableError(RuntimeError):
    """Safe runtime error; backend details must never be attached."""

    def __init__(self) -> None:
        """Initialize the security state unavailable error."""
        super().__init__("shared security state is unavailable")


class CorruptSecurityStateError(SecurityStateUnavailableError):
    """A record cannot be trusted because its bounded envelope is invalid."""


class SecurityStateBackend(Protocol):
    """Define the contract for security state backend operations."""
    max_record_bytes: int

    def ping(self) -> bool:
        """Check whether the security state backend is available."""
        ...

    def read(self, key: str) -> dict[str, Any] | None:
        """Read the requested value from the security state backend."""
        ...

    def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        add_indexes: tuple[str, ...] = (),
        remove_indexes: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
    ) -> None:
        """Handle write for the security state backend."""
        ...

    def delete(
        self,
        key: str,
        *,
        remove_indexes: tuple[str, ...] = (),
    ) -> bool:
        """Delete a record through the security state backend."""
        ...

    def members(self, index: str) -> list[str]:
        """List members stored by the security state backend."""
        ...

    def keys(self, prefix: str) -> list[str]:
        """List keys stored by the security state backend."""
        ...

    def locked(self, name: str) -> Iterator[None]:
        """Hold the named security state backend lock for the duration of the context."""
        ...


def _validate_key(key: str) -> None:
    """Validate the key."""
    if not _SAFE_KEY.fullmatch(key):
        raise CorruptSecurityStateError()


def _encode(value: dict[str, Any], max_record_bytes: int) -> str:
    """Return the encode."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CorruptSecurityStateError() from error
    if len(payload.encode("utf-8")) > max_record_bytes:
        raise CorruptSecurityStateError()
    return payload


def _decode(payload: str | bytes, max_record_bytes: int) -> dict[str, Any]:
    """Return the decode."""
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raise CorruptSecurityStateError()
    if len(raw) > max_record_bytes:
        raise CorruptSecurityStateError()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptSecurityStateError() from error
    if not isinstance(value, dict):
        raise CorruptSecurityStateError()
    return value


class MemorySecurityStateBackend:
    """Shared in-process backend for explicit development/test and ATDD."""

    def __init__(self, *, max_record_bytes: int = 65_536) -> None:
        """Initialize the memory security state backend."""
        self.max_record_bytes = max_record_bytes
        self._records: dict[str, str] = {}
        self._indexes: dict[str, set[str]] = {}
        self._expires: dict[str, float] = {}
        self._lock = threading.RLock()

    def ping(self) -> bool:
        """Check whether the memory security state backend is available."""
        return True

    def set_raw(self, key: str, payload: str) -> None:
        """Store the raw."""
        _validate_key(key)
        with self._lock:
            self._records[key] = payload
            self._expires.pop(key, None)

    def _purge_expired(self) -> None:
        """Remove expired records and their index memberships."""
        now = time.monotonic()
        expired = [key for key, deadline in self._expires.items() if deadline <= now]
        for key in expired:
            self._records.pop(key, None)
            self._expires.pop(key, None)
            for members in self._indexes.values():
                members.discard(key)

    def read(self, key: str) -> dict[str, Any] | None:
        """Read the requested value from the memory security state backend."""
        _validate_key(key)
        with self._lock:
            self._purge_expired()
            payload = self._records.get(key)
        return None if payload is None else _decode(payload, self.max_record_bytes)

    def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        add_indexes: tuple[str, ...] = (),
        remove_indexes: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
    ) -> None:
        """Handle write for the memory security state backend."""
        _validate_key(key)
        for index in (*add_indexes, *remove_indexes):
            _validate_key(index)
        payload = _encode(value, self.max_record_bytes)
        with self._lock:
            self._purge_expired()
            self._records[key] = payload
            if ttl_seconds is not None:
                self._expires[key] = time.monotonic() + max(1, ttl_seconds)
            else:
                self._expires.pop(key, None)
            for index in remove_indexes:
                self._indexes.setdefault(index, set()).discard(key)
            for index in add_indexes:
                self._indexes.setdefault(index, set()).add(key)

    def delete(
        self,
        key: str,
        *,
        remove_indexes: tuple[str, ...] = (),
    ) -> bool:
        """Delete a record through the memory security state backend."""
        _validate_key(key)
        with self._lock:
            self._purge_expired()
            existed = self._records.pop(key, None) is not None
            self._expires.pop(key, None)
            for index in remove_indexes:
                self._indexes.setdefault(index, set()).discard(key)
        return existed

    def members(self, index: str) -> list[str]:
        """List members stored by the memory security state backend."""
        _validate_key(index)
        with self._lock:
            self._purge_expired()
            return sorted(self._indexes.get(index, set()))

    def keys(self, prefix: str) -> list[str]:
        """List keys stored by the memory security state backend."""
        _validate_key(prefix.rstrip(":"))
        with self._lock:
            self._purge_expired()
            return sorted(key for key in self._records if key.startswith(prefix))

    @contextmanager
    def locked(self, name: str) -> Iterator[None]:
        """Hold the named memory security state backend lock for the duration of the context."""
        _validate_key(name)
        with self._lock:
            yield


class RedisSecurityStateBackend:
    """Small synchronous Redis boundary used by synchronous auth services."""

    def __init__(
        self,
        url: str,
        *,
        client: Any | None = None,
        max_record_bytes: int = 65_536,
        socket_timeout_seconds: float = 1.0,
    ) -> None:
        """Initialize the Redis security state backend."""
        if not url and client is None:
            raise SecurityStateConfigurationError(
                "shared security-state URL is required"
            )
        self.max_record_bytes = max_record_bytes
        if client is None:
            from redis import Redis

            client = Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
            )
        self._client = client

    def _call(self, callback, *, operation: str = "call"):
        """Call the redis security state backend."""
        try:
            value = callback()
            security_state_operations_total.labels("redis", operation, "success").inc()
            return value
        except (CorruptSecurityStateError, SecurityStateConfigurationError):
            security_state_operations_total.labels("redis", operation, "rejected").inc()
            raise
        except Exception as error:
            security_state_operations_total.labels("redis", operation, "unavailable").inc()
            raise SecurityStateUnavailableError() from error

    def ping(self) -> bool:
        """Check whether the Redis security state backend is available."""
        return bool(self._call(self._client.ping, operation="ping"))

    def read(self, key: str) -> dict[str, Any] | None:
        """Read the requested value from the Redis security state backend."""
        _validate_key(key)
        payload = self._call(lambda: self._client.get(key), operation="read")
        return None if payload is None else _decode(payload, self.max_record_bytes)

    def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        add_indexes: tuple[str, ...] = (),
        remove_indexes: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
    ) -> None:
        """Handle write for the Redis security state backend."""
        _validate_key(key)
        payload = _encode(value, self.max_record_bytes)

        def operation() -> None:
            """Handle operation for the Redis security state backend."""
            pipe = self._client.pipeline(transaction=True)
            pipe.set(key, payload, ex=ttl_seconds)
            for index in remove_indexes:
                _validate_key(index)
                pipe.srem(index, key)
            for index in add_indexes:
                _validate_key(index)
                pipe.sadd(index, key)
                if ttl_seconds:
                    pipe.expire(index, ttl_seconds)
            pipe.execute()

        self._call(operation, operation="write")

    def delete(
        self,
        key: str,
        *,
        remove_indexes: tuple[str, ...] = (),
    ) -> bool:
        """Delete a record through the Redis security state backend."""
        _validate_key(key)

        def operation() -> bool:
            """Handle operation for the Redis security state backend."""
            pipe = self._client.pipeline(transaction=True)
            pipe.delete(key)
            for index in remove_indexes:
                _validate_key(index)
                pipe.srem(index, key)
            result = pipe.execute()
            return bool(result[0])

        return bool(self._call(operation, operation="delete"))

    def members(self, index: str) -> list[str]:
        """List members stored by the Redis security state backend."""
        _validate_key(index)
        values = self._call(lambda: self._client.smembers(index), operation="members")
        return sorted(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        )

    def keys(self, prefix: str) -> list[str]:
        """List keys stored by the Redis security state backend."""
        _validate_key(prefix.rstrip(":"))
        values = self._call(
            lambda: list(self._client.scan_iter(match=f"{prefix}*")),
            operation="keys",
        )
        return sorted(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in values
        )

    @contextmanager
    def locked(self, name: str) -> Iterator[None]:
        """Hold the named Redis security state backend lock for the duration of the context."""
        _validate_key(name)
        try:
            lock = self._client.lock(
                f"security:v1:lock:{name}",
                timeout=10,
                blocking_timeout=2,
            )
            acquired = lock.acquire(blocking=True)
            if not acquired:
                raise SecurityStateUnavailableError()
            try:
                yield
            finally:
                lock.release()
        except SecurityStateUnavailableError:
            raise
        except Exception as error:
            raise SecurityStateUnavailableError() from error


__all__ = [
    "CorruptSecurityStateError",
    "MemorySecurityStateBackend",
    "RedisSecurityStateBackend",
    "SecurityStateBackend",
    "SecurityStateConfigurationError",
    "SecurityStateUnavailableError",
]
