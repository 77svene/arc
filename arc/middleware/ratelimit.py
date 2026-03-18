from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: int = 0,
        reset: Optional[int] = None,
    ):
        self.message = message
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset = reset
        super().__init__(message)

    def to_response(self) -> JSONResponse:
        headers = {
            "X-RateLimit-Limit": str(self.limit) if self.limit else "0",
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset) if self.reset else "",
            "Retry-After": str(self.retry_after) if self.retry_after else "",
        }
        return JSONResponse(
            status_code=429,
            content={"detail": self.message},
            headers=headers,
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests: int
    period: Union[int, float]  # seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    
    # Optional scope constraints
    key_prefix: str = "ratelimit"
    scope: Optional[str] = None  # for grouping limits
    exempt_when: Optional[Callable[[Request], bool]] = None
    
    # For token bucket algorithm
    tokens_per_second: Optional[float] = None
    bucket_size: Optional[int] = None


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    
    allowed: bool
    limit: int
    remaining: int
    reset: int  # Unix timestamp when the limit resets
    retry_after: Optional[int] = None  # seconds until next allowed request
    
    def to_headers(self) -> Dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset),
        }
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimitBackend(ABC):
    """Abstract base class for rate limit backends."""
    
    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: float,
        algorithm: RateLimitAlgorithm,
        tokens_per_second: Optional[float] = None,
        bucket_size: Optional[int] = None,
    ) -> RateLimitResult:
        """Check and update rate limit for a given key."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the backend connection."""
        pass
    
    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a given key."""
        pass


class InMemoryBackend(RateLimitBackend):
    """In-memory rate limit backend for single-instance deployments."""
    
    def __init__(self):
        self._windows: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self._token_buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True
    
    async def start(self) -> None:
        """Start the cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while self._running:
            await asyncio.sleep(60)  # Run cleanup every minute
            await self._cleanup()
    
    async def _cleanup(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        async with self._lock:
            for key in list(self._windows.keys()):
                self._windows[key] = [
                    (ts, count)
                    for ts, count in self._windows[key]
                    if ts > current_time - 3600
                ]
                if not self._windows[key]:
                    del self._windows[key]
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: float,
        algorithm: RateLimitAlgorithm,
        tokens_per_second: Optional[float] = None,
        bucket_size: Optional[int] = None,
    ) -> RateLimitResult:
        """Check rate limit using in-memory storage."""
        current_time = time.time()
        
        async with self._lock:
            if algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return await self._sliding_window(key, limit, window, current_time)
            elif algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return await self._token_bucket(
                    key, limit, window, tokens_per_second or 1.0,
                    bucket_size or limit, current_time
                )
            else:  # FIXED_WINDOW
                return await self._fixed_window(key, limit, window, current_time)
    
    async def _sliding_window(
        self, key: str, limit: int, window: float, current_time: float
    ) -> RateLimitResult:
        """Implement sliding window rate limiting."""
        window_start = current_time - window
        
        # Remove expired entries
        self._windows[key] = [
            (ts, count) for ts, count in self._windows[key]
            if ts > window_start
        ]
        
        # Calculate total requests in window
        total_requests = sum(count for ts, count in self._windows[key])
        
        if total_requests < limit:
            # Add current request
            self._windows[key].append((current_time, 1))
            remaining = limit - total_requests - 1
            reset_time = int(current_time + window)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=max(0, remaining),
                reset=reset_time,
            )
        
        # Calculate retry after based on oldest entry
        oldest = min(self._windows[key], key=lambda x: x[0])
        retry_after = int(oldest[0] + window - current_time) + 1
        
        return RateLimitResult(
            allowed=False,
            limit=limit,
            remaining=0,
            reset=int(current_time + retry_after),
            retry_after=retry_after,
        )
    
    async def _token_bucket(
        self,
        key: str,
        limit: int,
        window: float,
        tokens_per_second: float,
        bucket_size: int,
        current_time: float,
    ) -> RateLimitResult:
        """Implement token bucket rate limiting."""
        bucket = self._token_buckets.get(key)
        
        if bucket is None:
            bucket = {
                "tokens": float(bucket_size),
                "last_update": current_time,
            }
            self._token_buckets[key] = bucket
        
        # Add tokens based on elapsed time
        elapsed = current_time - bucket["last_update"]
        bucket["tokens"] = min(bucket_size, bucket["tokens"] + elapsed * tokens_per_second)
        bucket["last_update"] = current_time
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            remaining = int(bucket["tokens"])
            reset_time = int(current_time + (bucket_size - bucket["tokens"]) / tokens_per_second)
            return RateLimitResult(
                allowed=True,
                limit=bucket_size,
                remaining=remaining,
                reset=reset_time,
            )
        
        # Calculate time until next token
        retry_after = int((1 - bucket["tokens"]) / tokens_per_second) + 1
        reset_time = int(current_time + retry_after)
        
        return RateLimitResult(
            allowed=False,
            limit=bucket_size,
            remaining=0,
            reset=reset_time,
            retry_after=retry_after,
        )
    
    async def _fixed_window(
        self, key: str, limit: int, window: float, current_time: float
    ) -> RateLimitResult:
        """Implement fixed window rate limiting."""
        window_key = int(current_time / window)
        full_key = f"{key}:{window_key}"
        
        current_count = self._windows[full_key][0][1] if full_key in self._windows else 0
        
        if current_count < limit:
            self._windows[full_key] = [(current_time, current_count + 1)]
            remaining = limit - current_count - 1
            reset_time = int((window_key + 1) * window)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=max(0, remaining),
                reset=reset_time,
            )
        
        reset_time = int((window_key + 1) * window)
        retry_after = int(reset_time - current_time) + 1
        
        return RateLimitResult(
            allowed=False,
            limit=limit,
            remaining=0,
            reset=reset_time,
            retry_after=retry_after,
        )
    
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            if key in self._windows:
                del self._windows[key]
            if key in self._token_buckets:
                del self._token_buckets[key]
    
    async def close(self) -> None:
        """Close the backend."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._windows.clear()
        self._token_buckets.clear()


class RedisBackend(RateLimitBackend):
    """Redis-based rate limit backend for distributed deployments."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: Optional[Dict[str, Any]] = None,
        prefix: str = "ratelimit:",
    ):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is required for RedisBackend. "
                "Install it with: pip install redis"
            )
        
        self.redis_url = redis_url
        self.connection_kwargs = connection_kwargs or {}
        self.prefix = prefix
        self._redis: Optional[redis.Redis] = None
        self._pipeline: Optional[redis.client.Pipeline] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._redis is None:
            self._redis = await redis.from_url(
                self.redis_url,
                decode_responses=True,
                **self.connection_kwargs,
            )
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            await self.connect()
        return self._redis
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: float,
        algorithm: RateLimitAlgorithm,
        tokens_per_second: Optional[float] = None,
        bucket_size: Optional[int] = None,
    ) -> RateLimitResult:
        """Check rate limit using Redis."""
        r = await self._get_redis()
        full_key = f"{self.prefix}{key}"
        current_time = time.time()
        
        if algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._sliding_window(r, full_key, limit, window, current_time)
        elif algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._token_bucket(
                r, full_key, limit, tokens_per_second or 1.0,
                bucket_size or limit, current_time
            )
        else:  # FIXED_WINDOW
            return await self._fixed_window(r, full_key, limit, window, current_time)
    
    async def _sliding_window(
        self,
        r: redis.Redis,
        key: str,
        limit: int,
        window: float,
        current_time: float,
    ) -> RateLimitResult:
        """Implement sliding window using Redis sorted sets."""
        window_start = current_time - window
        
        # Use Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        
        -- Count current entries
        local count = redis.call('ZCARD', key)
        
        if count < limit then
            -- Add new entry
            redis.call('ZADD', key, now, now .. ':' .. redis.call('INCR', key .. ':counter'))
            redis.call('EXPIRE', key, math.ceil(window))
            return {1, limit - count - 1, 0}
        else
            -- Get oldest entry for retry-after calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if #oldest > 0 then
                retry_after = math.ceil(oldest[2] + window - now)
            end
            return {0, 0, retry_after}
        end
        """
        
        result = await r.eval(
            lua_script, 1, key, current_time, window_start, limit, window
        )
        
        allowed, remaining, retry_after = result
        
        if allowed:
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset=int(current_time + window),
            )
        
        return RateLimitResult(
            allowed=False,
            limit=limit,
            remaining=0,
            reset=int(current_time + retry_after),
            retry_after=retry_after,
        )
    
    async def _token_bucket(
        self,
        r: redis.Redis,
        key: str,
        limit: int,
        tokens_per_second: float,
        bucket_size: int,
        current_time: float,
    ) -> RateLimitResult:
        """Implement token bucket using Redis."""
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local bucket_size = tonumber(ARGV[2])
        local tokens_per_second = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1])
        local last_update = tonumber(bucket[2])
        
        if tokens == nil then
            tokens = bucket_size
            last_update = now
        end
        
        -- Add tokens based on elapsed time
        local elapsed = now - last_update
        tokens = math.min(bucket_size, tokens + elapsed * tokens_per_second)
        
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return {1, math.floor(tokens), 0}
        else
            local retry_after = math.ceil((1 - tokens) / tokens_per_second)
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return {0, 0, retry_after}
        end
        """
        
        result = await r.eval(
            lua_script, 1, key, current_time, bucket_size, tokens_per_second
        )
        
        allowed, remaining, retry_after = result
        
        if allowed:
            return RateLimitResult(
                allowed=True,
                limit=bucket_size,
                remaining=remaining,
                reset=int(current_time + (bucket_size - remaining) / tokens_per_second),
            )
        
        return RateLimitResult(
            allowed=False,
            limit=bucket_size,
            remaining=0,
            reset=int(current_time + retry_after),
            retry_after=retry_after,
        )
    
    async def _fixed_window(
        self,
        r: redis.Redis,
        key: str,
        limit: int,
        window: float,
        current_time: float,
    ) -> RateLimitResult:
        """Implement fixed window using Redis counters."""
        window_start = int(current_time / window)
        window_key = f"{key}:{window_start}"
        
        # Use pipeline for atomic increment
        pipe = r.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, math.ceil(window))
        results = await pipe.execute()
        
        current_count = results[0]
        
        if current_count <= limit:
            remaining = limit - current_count
            reset_time = int((window_start + 1) * window)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=max(0, remaining),
                reset=reset_time,
            )