"""
FastAPI Caching Layer
=====================

Built-in response caching with multiple backend support.

Provides:
- CacheBackend protocol with pluggable implementations
- MemoryCache for development/simple deployments
- RedisCache for production/distributed deployments
- @cache decorator for easy endpoint caching
- Cache-aside pattern with TTL and vary-by headers
- Full async support
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from arc import Depends, Request, Response
from arc.responses import JSONResponse

if TYPE_CHECKING:
    import redis.asyncio as redis

T = TypeVar("T")


class CacheType(str, Enum):
    """Supported cache backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


@dataclass(frozen=True)
class CacheOptions:
    """Configuration options for caching behavior.

    Attributes:
        ttl: Time-to-live in seconds for cached entries
        vary_on: List of header names to vary cache key by
        key_prefix: Prefix for all cache keys (namespace isolation)
        cache_type: Backend cache type to use
        enabled: Whether caching is active (can be toggled)
        max_size: Maximum entries (for memory cache)
    """

    ttl: int = 60
    vary_on: tuple[str, ...] = ("Accept", "Accept-Language")
    key_prefix: str = "arc:cache"
    cache_type: CacheType = CacheType.MEMORY
    enabled: bool = True
    max_size: int | None = 1000

    def __post_init__(self) -> None:
        if self.ttl < 0:
            raise ValueError("TTL must be non-negative")
        if self.max_size is not None and self.max_size < 1:
            raise ValueError("max_size must be positive if specified")


@dataclass
class CacheEntry(Generic[T]):
    """Represents a cached value with metadata.

    Attributes:
        value: The cached data
        created_at: Timestamp when entry was created
        expires_at: Timestamp when entry expires
        access_count: Number of times entry was accessed
        last_accessed: Timestamp of last access
        metadata: Additional cache metadata
    """

    value: T
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol defining the cache backend interface.

    Implementations must provide async get/set/delete operations
    and handle serialization/deserialization of cached values.
    """

    async def get(self, key: str) -> Any | None:
        """Retrieve a value from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found/expired
        """
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successfully cached
        """
        ...

    async def delete(self, key: str) -> bool:
        """Remove a value from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was deleted, False if not found
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired
        """
        ...

    async def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of keys cleared
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, etc.)
        """
        ...


class AbstractCacheBackend(ABC):
    """Abstract base class for cache backends.

    Provides common functionality and defines the interface
    that all backend implementations must follow.
    """

    def __init__(self, options: CacheOptions | None = None) -> None:
        self.options = options or CacheOptions()

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve a value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove a value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all cached values."""
        pass

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics. Override in subclasses."""
        return {
            "type": self.__class__.__name__,
            "enabled": self.options.enabled,
        }

    def _build_key(self, *parts: str) -> str:
        """Build a cache key from parts.

        Args:
            *parts: Key components to join

        Returns:
            Formatted cache key
        """
        separator = ":"
        return separator.join(
            filter(None, [self.options.key_prefix] + list(parts))
        )

    def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        return json.dumps(
            value,
            default=self._json_serializer,
        ).encode("utf-8")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize a value from storage.

        Args:
            data: Bytes to deserialize

        Returns:
            Deserialized value
        """
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class MemoryCache(AbstractCacheBackend):
    """In-memory cache backend using thread-safe dictionary.

    Suitable for development, single-instance deployments,
    or as a local cache in front of distributed backends.

    Features:
    - Thread-safe operations
    - LRU eviction when max_size is reached
    - Automatic expiration checking
    - Per-entry TTL support
    """

    def __init__(
        self,
        options: CacheOptions | None = None,
    ) -> None:
        super().__init__(options)
        self._cache: dict[str, CacheEntry[Any]] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }
        self._access_order: list[str] = []

    async def get(self, key: str) -> Any | None:
        """Retrieve a value from memory cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._access_order.remove(key)
                self._stats["misses"] += 1
                return None

            entry.touch()
            self._update_access_order(key)
            self._stats["hits"] += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store a value in memory cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful
        """
        ttl = ttl if ttl is not None else self.options.ttl
        now = time.time()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + ttl if ttl > 0 else float("inf"),
        )

        async with self._lock:
            if self.options.max_size and len(self._cache) >= self.options.max_size:
                self._evict_lru()

            self._cache[key] = entry
            self._update_access_order(key)
            self._stats["sets"] += 1

        return True

    async def delete(self, key: str) -> bool:
        """Remove a value from memory cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["deletes"] += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return False
            return True

    async def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, etc.
        """
        async with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total if total > 0 else 0.0
            )

            return {
                **self._stats,
                "type": "memory",
                "size": len(self._cache),
                "max_size": self.options.max_size,
                "hit_rate": round(hit_rate, 4),
                "enabled": self.options.enabled,
            }

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Glob-style pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        import fnmatch

        async with self._lock:
            keys_to_delete = [
                key
                for key in self._cache
                if fnmatch.fnmatch(key, pattern)
            ]
            for key in keys_to_delete:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            return len(keys_to_delete)

    def _update_access_order(self, key: str) -> None:
        """Update LRU access order list."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats["evictions"] += 1


class RedisCache(AbstractCacheBackend):
    """Redis cache backend for distributed caching.

    Provides production-grade caching with:
    - Atomic operations
    - Distributed storage
    - Key expiration
    - Pub/sub for cache invalidation
    - Connection pooling
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        url: str | None = None,
        options: CacheOptions | None = None,
        **redis_kwargs: Any,
    ) -> None:
        super().__init__(options)
        self._redis: redis.Redis | None = redis_client
        self._url = url
        self._redis_kwargs = redis_kwargs
        self._pool: Any | None = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis_async

            if self._url:
                self._redis = redis_async.from_url(
                    self._url,
                    decode_responses=False,
                    **self._redis_kwargs,
                )
            else:
                self._redis = redis_async.Redis(
                    decode_responses=False,
                    **self._redis_kwargs,
                )
        return self._redis

    async def get(self, key: str) -> Any | None:
        """Retrieve a value from Redis.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            r = await self._get_redis()
            data = await r.get(key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return self._deserialize(data)

        except Exception:
            self._stats["errors"] += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Store a value in Redis.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        try:
            r = await self._get_redis()
            ttl = ttl if ttl is not None else self.options.ttl
            data = self._serialize(value)

            if ttl > 0:
                await r.setex(key, ttl, data)
            else:
                await r.set(key, data)

            self._stats["sets"] += 1
            return True

        except Exception:
            self._stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """Remove a value from Redis.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        try:
            r = await self._get_redis()
            result = await r.delete(key)
            self._stats["deletes"] += 1
            return result > 0

        except Exception:
            self._stats["errors"] += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            r = await self._get_redis()
            return await r.exists(key) > 0

        except Exception:
            self._stats["errors"] += 1
            return False

    async def clear(self) -> int:
        """Clear all cached values with the configured prefix.

        Returns:
            Number of keys cleared
        """
        try:
            r = await self._get_redis()
            pattern = f"{self.options.key_prefix}:*"

            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await r.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await r.delete(*keys)
                if cursor == 0:
                    break

            return deleted

        except Exception:
            self._stats["errors"] += 1
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics from Redis.

        Returns:
            Dictionary with Redis info
        """
        try:
            r = await self._get_redis()
            info = await r.info("stats")
            total = self._stats["hits"] + self._stats["misses"]

            return {
                **self._