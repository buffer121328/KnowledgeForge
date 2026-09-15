"""Best-effort Redis token-blacklist collaboration used by HTTP auth boundaries."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from auth.config import AuthSettings
import logging

logger = logging.getLogger(__name__)


def _blacklist_key(token: str) -> str:
    """Return the blacklist key."""
    return f"token:blacklist:v2:{hashlib.sha256(token.encode('utf-8')).hexdigest()}"


class TokenBlacklist:
    """Keep Redis failures non-fatal while preserving current logout semantics."""

    def __init__(self, settings: AuthSettings | None = None, redis_client: Any = None) -> None:
        """Initialize the token blacklist."""
        self.settings = settings or AuthSettings()
        self.redis = redis_client

    async def contains(self, token: str) -> bool:
        """Report whether the token blacklist contains the requested value."""
        if not self.redis:
            return False
        try:
            return bool(await self.redis.get(_blacklist_key(token)))
        except Exception as error:
            logger.warning(
                "token_blacklist_check_failed error_type=%s",
                type(error).__name__,
            )
            return False

    async def add(self, token: str, expires_at: int) -> None:
        """Add a value to the token blacklist."""
        client = self.redis
        should_close = False
        if client is None:
            try:
                import redis.asyncio as aioredis

                client = aioredis.from_url(self.settings.redis_url, decode_responses=True)
                should_close = True
            except Exception:
                return
        try:
            ttl = max(int(expires_at - datetime.now(timezone.utc).timestamp()), 1)
            await client.set(_blacklist_key(token), "1", ex=ttl)
        except Exception:
            # Logout has always succeeded even if Redis is unavailable.
            pass
        finally:
            if should_close:
                try:
                    await client.aclose()
                except Exception:
                    pass
