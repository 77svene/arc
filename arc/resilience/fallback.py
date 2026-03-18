"""Resilience module for graceful degradation and fallback responses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar, Generic, Awaitable, Protocol
from collections.abc import Awaitable
from collections import OrderedDict

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class FallbackStrategy(Enum):
    """Enumeration of available fallback strategies."""
    
    CACHE = "cache"
    STATIC = "static"
    SECONDARY_SERVICE = "secondary_service"
    DEGRADED_MODE = "degraded_mode"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    
    strategies: list[FallbackStrategy] = field(
        default_factory=lambda: [FallbackStrategy.CACHE, FallbackStrategy.STATIC]
    )
    cache_ttl: int = 300  # seconds
    cache_max_size: int = 1000
    timeout: float = 5.0  # seconds
    retry_count: int = 2
    retry_delay: float = 0.5  # seconds
    enable_degraded_mode: bool = True
    degraded_response: dict[str, Any] | None = None


@dataclass
class FallbackResult:
    """Result of a fallback operation."""
    
    success: bool
    data: Any = None
    source: FallbackStrategy | None = None
    error: str | None = None
    from_cache: bool = False
    latency_ms: float = 0.0


class FallbackStrategyHandler(ABC, Generic[T]):
    """Abstract base class for fallback strategy handlers."""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
    
    @abstractmethod
    async def execute(
        self,
        request: Request,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute the fallback strategy."""
        pass
    
    @abstractmethod
    async def can_handle(self, request: Request) -> bool:
        """Check if this strategy can handle the request."""
        pass


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation for fallback responses."""
    
    def __init__(self, max_size: int, ttl: int):
        self._cache: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
    
    def _generate_key(self, request: Request) -> str:
        """Generate cache key from request."""
        key_data = f"{request.url.path}:{request.url.query}:{request.method}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, request: Request) -> T | None:
        """Get cached response if available and not expired."""
        key = self._generate_key(request)
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                return data
            else:
                del self._cache[key]
        return None
    
    def set(self, request: Request, data: T) -> None:
        """Store response in cache."""
        key = self._generate_key(request)
        self._cache[key] = (data, time.time())
        self._cache.move_to_end(key)
        
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
    
    def invalidate(self, request: Request) -> None:
        """Invalidate specific cache entry."""
        key = self._generate_key(request)
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()


class CacheFallbackHandler(FallbackStrategyHandler[Any]):
    """Handler for cache-based fallback strategy."""
    
    def __init__(self, config: FallbackConfig):
        super().__init__(config)
        self._cache = LRUCache[dict[str, Any]](config.cache_max_size, config.cache_ttl)
    
    async def can_handle(self, request: Request) -> bool:
        """Check if we have a cached response."""
        return self._cache.get(request) is not None
    
    async def execute(
        self,
        request: Request,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute cache fallback strategy."""
        start_time = time.perf_counter()
        
        cached_data = self._cache.get(request)
        if cached_data is not None:
            latency = (time.perf_counter() - start_time) * 1000
            return FallbackResult(
                success=True,
                data=cached_data,
                source=FallbackStrategy.CACHE,
                from_cache=True,
                latency_ms=latency
            )
        
        return FallbackResult(
            success=False,
            error="No cached response available",
            source=FallbackStrategy.CACHE
        )
    
    def store_response(self, request: Request, response: dict[str, Any]) -> None:
        """Store response in cache (call after successful primary request)."""
        self._cache.set(request, response)


class StaticFallbackHandler(FallbackStrategyHandler[Any]):
    """Handler for static/default response fallback strategy."""
    
    def __init__(self, config: FallbackConfig, static_responses: dict[str, dict[str, Any]] | None = None):
        super().__init__(config)
        self._static_responses = static_responses or {
            "default": {
                "status": "degraded",
                "message": "Service is operating in degraded mode",
                "data": []
            }
        }
        self._path_fallbacks: dict[str, dict[str, Any]] = {}
    
    async def can_handle(self, request: Request) -> bool:
        """Static responses are always available."""
        return True
    
    def register_fallback(self, path_pattern: str, response: dict[str, Any]) -> None:
        """Register a static fallback for a specific path pattern."""
        self._path_fallbacks[path_pattern] = response
    
    async def execute(
        self,
        request: Request,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute static fallback strategy."""
        start_time = time.perf_counter()
        
        # Check for path-specific fallback
        path = request.url.path
        static_response = None
        
        for pattern, response in self._path_fallbacks.items():
            if path.startswith(pattern):
                static_response = response
                break
        
        if static_response is None:
            static_response = self._static_responses.get(
                path,
                self._static_responses["default"]
            )
        
        latency = (time.perf_counter() - start_time) * 1000
        return FallbackResult(
            success=True,
            data=static_response,
            source=FallbackStrategy.STATIC,
            from_cache=False,
            latency_ms=latency
        )


class SecondaryServiceFallbackHandler(FallbackStrategyHandler[Any]):
    """Handler for secondary service fallback strategy."""
    
    def __init__(self, config: FallbackConfig, secondary_clients: dict[str, Callable[..., Awaitable[Any]]] | None = None):
        super().__init__(config)
        self._secondary_clients = secondary_clients or {}
        self._health_checks: dict[str, bool] = {}
    
    async def can_handle(self, request: Request) -> bool:
        """Check if any secondary service is available."""
        for name in self._secondary_clients:
            if self._health_checks.get(name, False):
                return True
        return len(self._secondary_clients) > 0
    
    def register_secondary(self, name: str, client: Callable[..., Awaitable[Any]]) -> None:
        """Register a secondary service client."""
        self._secondary_clients[name] = client
        self._health_checks[name] = True
    
    async def _health_check(self, name: str) -> bool:
        """Perform health check on secondary service."""
        try:
            client = self._secondary_clients[name]
            if asyncio.iscoroutinefunction(client):
                await asyncio.wait_for(client(health_check=True), timeout=2.0)
            else:
                client(health_check=True)
            self._health_checks[name] = True
            return True
        except Exception as e:
            logger.warning(f"Secondary service {name} health check failed: {e}")
            self._health_checks[name] = False
            return False
    
    async def execute(
        self,
        request: Request,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute secondary service fallback strategy."""
        start_time = time.perf_counter()
        
        for name, client in self._secondary_clients.items():
            if not self._health_checks.get(name, True):
                continue
            
            for attempt in range(self.config.retry_count):
                try:
                    if asyncio.iscoroutinefunction(client):
                        result = await asyncio.wait_for(
                            client(*args, **kwargs),
                            timeout=self.config.timeout
                        )
                    else:
                        result = client(*args, **kwargs)
                    
                    latency = (time.perf_counter() - start_time) * 1000
                    return FallbackResult(
                        success=True,
                        data=result,
                        source=FallbackStrategy.SECONDARY_SERVICE,
                        from_cache=False,
                        latency_ms=latency
                    )
                except Exception as e:
                    logger.warning(
                        f"Secondary {name} attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self.config.retry_count - 1:
                        await asyncio.sleep(self.config.retry_delay)
        
        latency = (time.perf_counter() - start_time) * 1000
        return FallbackResult(
            success=False,
            error="All secondary services failed",
            source=FallbackStrategy.SECONDARY_SERVICE,
            latency_ms=latency
        )


class DegradedModeHandler(FallbackStrategyHandler[Any]):
    """Handler for degraded mode fallback strategy."""
    
    def __init__(self, config: FallbackConfig):
        super().__init__(config)
        self._enabled = True
        self._degraded_response = config.degraded_response or {
            "status": "degraded",
            "message": "Service is experiencing reduced functionality",
            "timestamp": "",
            "available_endpoints": []
        }
    
    async def can_handle(self, request: Request) -> bool:
        """Degraded mode always available when enabled."""
        return self._enabled and self.config.enable_degraded_mode
    
    def set_degraded(self, enabled: bool) -> None:
        """Enable or disable degraded mode."""
        self._enabled = enabled
    
    def set_response(self, response: dict[str, Any]) -> None:
        """Set the degraded response template."""
        self._degraded_response = response
    
    async def execute(
        self,
        request: Request,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute degraded mode fallback strategy."""
        start_time = time.perf_counter()
        
        response = self._degraded_response.copy()
        response["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        response["path"] = request.url.path
        response["method"] = request.method
        
        latency = (time.perf_counter() - start_time) * 1000
        return FallbackResult(
            success=True,
            data=response,
            source=FallbackStrategy.DEGRADED_MODE,
            from_cache=False,
            latency_ms=latency
        )


class FallbackManager:
    """Manager for coordinating multiple fallback strategies."""
    
    def __init__(self, config: FallbackConfig | None = None):
        self.config = config or FallbackConfig()
        self._handlers: dict[FallbackStrategy, FallbackStrategyHandler[Any]] = {}
        self._cache_handler: CacheFallbackHandler | None = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize default handlers based on config."""
        if self._initialized:
            return
        
        self._cache_handler = CacheFallbackHandler(self.config)
        self._handlers[FallbackStrategy.CACHE] = self._cache_handler
        
        self._handlers[FallbackStrategy.STATIC] = StaticFallbackHandler(self.config)
        self._handlers[FallbackStrategy.SECONDARY_SERVICE] = SecondaryServiceFallbackHandler(self.config)
        self._handlers[FallbackStrategy.DEGRADED_MODE] = DegradedModeHandler(self.config)
        
        self._initialized = True
    
    def get_handler(self, strategy: FallbackStrategy) -> FallbackStrategyHandler[Any] | None:
        """Get handler for specific strategy."""
        if not self._initialized:
            self.initialize()
        return self._handlers.get(strategy)
    
    def register_handler(self, strategy: FallbackStrategy, handler: FallbackStrategyHandler[Any]) -> None:
        """Register a custom fallback handler."""
        if not self._initialized:
            self.initialize()
        self._handlers[strategy] = handler
    
    async def execute_chain(
        self,
        request: Request,
        primary_func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> FallbackResult:
        """Execute fallback chain starting with primary function."""
        if not self._initialized:
            self.initialize()
        
        # Try primary function first
        for attempt in range(self.config.retry_count):
            try:
                start_time = time.perf_counter()
                result = await asyncio.wait_for(
                    primary_func(*args, **kwargs),
                    timeout=self.config.timeout
                )
                
                # Cache successful response
                if self._cache_handler and isinstance(result, dict):
                    self._cache_handler.store_response(request, result)
                
                latency = (time.perf_counter() - start_time) * 1000
                return FallbackResult(
                    success=True,
                    data=result,
                    source=None,
                    latency_ms=latency
                )
            except asyncio.TimeoutError:
                logger.warning(f"Primary function timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Primary function failed on attempt {attempt + 1}: {e}")
            
            if attempt < self.config.retry_count - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        # Try fallback strategies in order
        for strategy in self.config.strategies:
            handler = self._handlers.get(strategy)
            if handler is None:
                continue
            
            if not await handler.can_handle(request):
                continue
            
            result = await handler.execute(request, primary_func, *args, **kwargs)
            if result.success:
                logger.info(f"Fallback succeeded using {strategy.value}")
                return result
        
        # Last resort: degraded mode
        degraded_handler = self._handlers.get(FallbackStrategy.DEGRADED_MODE)
        if degraded_handler and await degraded_handler.can_handle(request):
            return await degraded_handler.execute(request, primary_func, *args, **kwargs)
        
        return FallbackResult(
            success=False,
            error="All fallback strategies exhausted"
        )


# Global fallback manager instance
_fallback_manager: FallbackManager | None = None


def get_fallback_manager() -> FallbackManager:
    """Get the global fallback manager instance."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager()
        _fallback_manager.initialize()
    return _fallback_manager


def set_fallback_manager(manager: FallbackManager) -> None:
    """Set a custom fallback manager instance."""
    global _fallback_manager
    _fallback_manager = manager


def fallback(
    strategies: list[FallbackStrategy] | None = None,
    cache_ttl: int = 300,
    cache_max_size: int = 1000,
    timeout: float = 5.0,
    retry_count: int = 2,
    retry_delay: float = 0.5,
    degraded_response: dict[str, Any] | None = None,
    status_code: int = 200,
    fallback_status_code: int = 503
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[Response]]]:
    """
    Decorator for adding fallback behavior to async endpoint functions.
    
    Args:
        strategies: List of fallback strategies to try in order
        cache_ttl: Time-to-live for cached responses in seconds
        cache_max_size: Maximum number of cached responses
        timeout: Timeout for primary function execution
        retry_count: Number of retries for primary function
        retry_delay: Delay between retries in seconds
        degraded_response: Custom response for degraded mode
        status_code: Status code for successful responses
        fallback_status_code: Status code when using fallback
    
    Returns:
        Decorated function with fallback support
    
    Example:
        @fallback(strategies=[FallbackStrategy.CACHE, FallbackStrategy.STATIC])
        async def get_user(request: Request, user_id: str) -> Response:
            # Primary function that might fail
            return await user_service.get(user_id)
    """
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[Response]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            manager = get_fallback_manager()
            
            # Update config with decorator parameters
            config = FallbackConfig(
                strategies=strategies or [FallbackStrategy.CACHE, FallbackStrategy.STATIC],
                cache_ttl=cache_ttl,
                cache_max_size=cache_max_size,
                timeout=timeout,
                retry_count=retry_count,
                retry_delay=retry_delay,
                degraded_response=degraded_response