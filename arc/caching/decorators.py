from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from arc import Depends, FastAPI, Request, Response
from arc.responses import JSONResponse
from starlette.datastructures import Headers

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

try:
    import memcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    memcache = None


logger = logging.getLogger(__name__)


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


@dataclass
class CacheEntry:
    """Represents a cached value with metadata."""
    value: Any
    created_at: float
    ttl: int
    content_type: str = "application/json"
    vary_headers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    ttl: int = 60
    vary_on: List[str] = field(default_factory=list)
    key_prefix: str = ""
    cache_when: Optional[Callable[..., bool]] = None
    skip_on_status_codes: List[int] = field(default_factory=list)
    force_refresh: bool = False


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol defining the cache backend interface."""
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a value from cache."""
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int,
        content_type: str = "application/json",
        vary_headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Store a value in cache."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Remove a value from cache."""
        ...
    
    async def clear(self) -> None:
        """Clear all cached values."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        ...


class MemoryCache(CacheBackend):
    """
    In-memory cache backend using a dictionary with TTL support.
    Suitable for single-instance deployments or testing.
    """
    
    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}
        self._lock = False
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        entry = self._store.get(key)
        if entry is None:
            return None
        
        if entry.is_expired:
            await self.delete(key)
            return None
        
        return entry
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int,
        content_type: str = "application/json",
        vary_headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl,
            content_type=content_type,
            vary_headers=vary_headers or {},
        )
        self._store[key] = entry
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    async def clear(self) -> None:
        self._store.clear()
    
    async def exists(self, key: str) -> bool:
        entry = await self.get(key)
        return entry is not None
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = len(self._store)
        expired = sum(1 for e in self._store.values() if e.is_expired)
        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
        }


class RedisCache(CacheBackend):
    """
    Redis-based cache backend with async support.
    Provides distributed caching suitable for multi-instance deployments.
    """
    
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "arc:cache:",
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        decode_responses: bool = True,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for RedisCache. Install it with: pip install redis"
            )
        
        self._url = url
        self._key_prefix = key_prefix
        self._client: Optional[Redis] = None
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout
        self._retry_on_timeout = retry_on_timeout
        self._decode_responses = decode_responses
    
    async def connect(self) -> Redis:
        """Establish connection to Redis."""
        if self._client is None:
            self._client = redis.Redis(
                from_url=self._url,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                retry_on_timeout=self._retry_on_timeout,
                decode_responses=self._decode_responses,
            )
        return self._client
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    def _make_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        try:
            client = await self.connect()
            data = await client.get(self._make_key(key))
            
            if data is None:
                return None
            
            parsed = json.loads(data)
            return CacheEntry(
                value=parsed["value"],
                created_at=parsed["created_at"],
                ttl=parsed["ttl"],
                content_type=parsed.get("content_type", "application/json"),
                vary_headers=parsed.get("vary_headers", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse cache entry for key {key}: {e}")
            await self.delete(key)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int,
        content_type: str = "application/json",
        vary_headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        try:
            client = await self.connect()
            data = json.dumps({
                "value": value,
                "created_at": time.time(),
                "ttl": ttl,
                "content_type": content_type,
                "vary_headers": vary_headers or {},
            })
            
            await client.set(
                self._make_key(key),
                data,
                ex=ttl,
            )
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            client = await self.connect()
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> None:
        try:
            client = await self.connect()
            cursor = 0
            pattern = f"{self._key_prefix}*"
            
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        try:
            client = await self.connect()
            return await client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        try:
            client = await self.connect()
            full_pattern = self._make_key(pattern)
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=full_pattern, count=100)
                if keys:
                    deleted = await client.delete(*keys)
                    deleted_count += deleted
                if cursor == 0:
                    break
            
            return deleted_count
        except Exception as e:
            logger.error(f"Redis invalidate_pattern error: {e}")
            return 0


class MemcachedCache(CacheBackend):
    """
    Memcached-based cache backend.
    Provides distributed caching with minimal memory footprint.
    """
    
    def __init__(
        self,
        servers: Union[str, List[str]] = ["localhost:11211"],
        key_prefix: str = "arc_cache_",
    ) -> None:
        if not MEMCACHED_AVAILABLE:
            raise ImportError(
                "pymemcache is required for MemcachedCache. Install it with: pip install pymemcache"
            )
        
        from pymemcache.client.base import Client
        
        if isinstance(servers, str):
            servers = [servers]
        
        self._servers = servers
        self._key_prefix = key_prefix
        self._client: Optional[Client] = None
    
    def _get_client(self) -> Client:
        if self._client is None:
            from pymemcache.client.base import Client
            self._client = Client(
                self._servers[0],
                connect_timeout=5,
                timeout=5,
                no_delay=True,
            )
        return self._client
    
    def _make_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        try:
            client = self._get_client()
            data = client.get(self._make_key(key))
            
            if data is None:
                return None
            
            parsed = json.loads(data)
            entry = CacheEntry(
                value=parsed["value"],
                created_at=parsed["created_at"],
                ttl=parsed["ttl"],
                content_type=parsed.get("content_type", "application/json"),
                vary_headers=parsed.get("vary_headers", {}),
            )
            
            if entry.is_expired:
                await self.delete(key)
                return None
            
            return entry
        except Exception as e:
            logger.error(f"Memcached get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int,
        content_type: str = "application/json",
        vary_headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        try:
            client = self._get_client()
            data = json.dumps({
                "value": value,
                "created_at": time.time(),
                "ttl": ttl,
                "content_type": content_type,
                "vary_headers": vary_headers or {},
            })
            
            client.set(
                self._make_key(key),
                data,
                expire=ttl,
            )
            return True
        except Exception as e:
            logger.error(f"Memcached set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            client = self._get_client()
            result = client.delete(self._make_key(key))
            return result is True
        except Exception as e:
            logger.error(f"Memcached delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> None:
        try:
            client = self._get_client()
            client.flush_all()
        except Exception as e:
            logger.error(f"Memcached clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        try:
            client = self._get_client()
            return client.get(self._make_key(key)) is not None
        except Exception as e:
            logger.error(f"Memcached exists error for key {key}: {e}")
            return False


def generate_cache_key(
    request: Request,
    vary_on: List[str],
    prefix: str = "",
) -> str:
    """
    Generate a unique cache key from request information.
    
    Includes:
    - HTTP method
    - Path
    - Query parameters (sorted)
    - Specified vary headers
    """
    parts = [request.method, request.url.path]
    
    if request.query_params:
        sorted_params = sorted(request.query_params.multi_items())
        params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        parts.append(params_str)
    
    for header in vary_on:
        header_value = request.headers.get(header.lower()) or request.headers.get(header)
        if header_value:
            parts.append(f"{header}:{header_value}")
    
    if prefix:
        parts.insert(0, prefix)
    
    key_string = "|".join(parts)
    
    if len(key_string) > 200:
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return key_hash
    
    return key_string


def cache(
    ttl: int = 60,
    vary_on: Optional[List[str]] = None,
    key_prefix: str = "",
    cache_when: Optional[Callable[..., bool]] = None,
    skip_on_status_codes: Optional[List[int]] = None,
    force_refresh: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to cache FastAPI endpoint responses.
    
    Args:
        ttl: Time-to-live in seconds for cached responses.
        vary_on: List of HTTP header names to vary the cache key on.
                 Common values: 'Accept', 'Accept-Language', 'Accept-Encoding'.
        key_prefix: Optional prefix for cache keys.
        cache_when: Optional callable that takes the response and returns bool.
                    Only caches if returns True.
        skip_on_status_codes: List of HTTP status codes that should NOT be cached.
        force_refresh: If True, bypass cache and always fetch fresh data.
    
    Example:
        @app.get("/items/{item_id}")
        @cache(ttl=300, vary_on=["Accept"])
        async def get_item(item_id: int):
            return {"item_id": item_id}
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(
            request: Request,
            response: Response,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            cache_backend: CacheBackend = request.state.cache_backend
            
            vary_headers = vary_on or []
            
            should_refresh = (
                force_refresh or
                request.query_params.get("_refresh") == "true" or
                request.query_params.get("refresh") == "1"
            )
            
            if not should_refresh and hasattr(request.state, "skip_cache") and request.state.skip_cache:
                should_refresh = True
            
            cache_key = generate_cache_key(
                request=request,
                vary_on=vary_headers,
                prefix=key_prefix,
            )
            
            if not should_refresh:
                cached = await cache_backend.get(cache_key)
                if cached is not None:
                    content_type = cached.content_type
                    response.headers["X-Cache"] = "HIT"
                    response.headers["X-Cache-Key"] = cache_key
                    
                    if cached.vary_headers:
                        response.headers["Vary"] = ",".join(cached.vary_headers.keys())
                    
                    if "json" in content_type or "application/json" in content_type:
                        return JSONResponse(
                            content=cached.value,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                        )
                    else:
                        from arc.responses import Response as StarletteResponse
                        return StarletteResponse(
                            content=json.dumps(cached.value).encode(),
                            status_code=response.status_code,
                            media_type=content_type,
                            headers=dict(response.headers),
                        )
            
            response.headers["X-Cache"] = "MISS"
            response.headers["X-Cache-Key"] = cache_key
            
            result = await func(request, response, *args, **kwargs)
            
            status_code = getattr(response, "status_code", 200)
            
            skip_statuses = skip_on_status_codes or []
            if status_code in skip_statuses:
                return result
            
            if cache_when is not None and not cache_when(result):
                return result
            
            if isinstance(result, JSONResponse):
                content = result.body.decode() if hasattr(result, "body") else result.json()
                content_value = json.loads(content) if isinstance(content, str) else content
                content_type = result.media_type or "application/json"
            elif hasattr(result, "model_dump"):
                content_value = result.model_dump()
                content_type = "application/json"
            elif isinstance(result, dict):
                content_value = result
                content_type = "application/json"
            elif isinstance(result, str):