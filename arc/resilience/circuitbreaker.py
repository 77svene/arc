"""Circuit Breaker Pattern Implementation for FastAPI.

Provides resilience for external service calls by failing fast after N failures,
testing recovery with half-open state, and providing metrics for monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import defaultdict

from arc import HTTPException, Request
from starlette.responses import Response

T = TypeVar("T")

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and requests are rejected."""
    
    def __init__(
        self,
        service_name: str,
        retry_after: float,
        message: Optional[str] = None
    ):
        self.service_name = service_name
        self.retry_after = retry_after
        self.message = message or f"Circuit breaker is open for '{service_name}'. Retry after {retry_after:.2f}s"
        super().__init__(self.message)


@dataclass
class CircuitBreakerMetrics:
    """Metrics for monitoring circuit breaker state."""
    
    service_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_at: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_half_open_attempts: int = 0
    half_open_successes: int = 0
    half_open_failures: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate over all calls."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls
    
    @property
    def half_open_success_rate(self) -> float:
        """Calculate success rate during half-open state."""
        total = self.half_open_successes + self.half_open_failures
        if total == 0:
            return 0.0
        return self.half_open_successes / total
    
    @property
    def time_in_current_state(self) -> float:
        """Time spent in current state in seconds."""
        return time.time() - self.state_changed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": round(self.failure_rate, 4),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_in_current_state": round(self.time_in_current_state, 2),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "half_open_attempts": self.total_half_open_attempts,
            "half_open_success_rate": round(self.half_open_success_rate, 4),
        }


class CircuitBreakerCallback:
    """Base class for circuit breaker callbacks."""
    
    def on_state_change(
        self,
        service_name: str,
        old_state: CircuitState,
        new_state: CircuitState,
        metrics: CircuitBreakerMetrics
    ) -> None:
        """Called when circuit state changes."""
        pass
    
    def on_call_success(
        self,
        service_name: str,
        metrics: CircuitBreakerMetrics
    ) -> None:
        """Called after successful call."""
        pass
    
    def on_call_failure(
        self,
        service_name: str,
        metrics: CircuitBreakerMetrics,
        error: Optional[Exception] = None
    ) -> None:
        """Called after failed call."""
        pass


class LoggingCallback(CircuitBreakerCallback):
    """Logs circuit breaker events."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def on_state_change(
        self,
        service_name: str,
        old_state: CircuitState,
        new_state: CircuitState,
        metrics: CircuitBreakerMetrics
    ) -> None:
        level = logging.WARNING if new_state == CircuitState.OPEN else logging.INFO
        self.logger.log(
            level,
            f"Circuit breaker state change: {service_name} | "
            f"{old_state.value} -> {new_state.value} | "
            f"failures: {metrics.failure_count}, successes: {metrics.success_count}"
        )
    
    def on_call_failure(
        self,
        service_name: str,
        metrics: CircuitBreakerMetrics,
        error: Optional[Exception] = None
    ) -> None:
        if metrics.consecutive_failures >= 3:
            self.logger.warning(
                f"Circuit breaker: {service_name} | "
                f"consecutive failures: {metrics.consecutive_failures}"
            )


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    _breakers: Dict[str, "CircuitBreaker"] = {}
    _metrics: Dict[str, CircuitBreakerMetrics] = {}
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_or_create(
        cls,
        name: str,
        **kwargs
    ) -> "CircuitBreaker":
        """Get existing or create new circuit breaker."""
        async with cls._lock:
            if name not in cls._breakers:
                breaker = CircuitBreaker(name, **kwargs)
                cls._breakers[name] = breaker
                cls._metrics[name] = breaker.metrics
            return cls._breakers[name]
    
    @classmethod
    def get_metrics(cls, name: str) -> Optional[CircuitBreakerMetrics]:
        """Get metrics for a circuit breaker."""
        return cls._metrics.get(name)
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, CircuitBreakerMetrics]:
        """Get all circuit breaker metrics."""
        return dict(cls._metrics)
    
    @classmethod
    async def reset(cls, name: str) -> None:
        """Reset a circuit breaker."""
        async with cls._lock:
            if name in cls._breakers:
                cls._breakers[name].reset()
    
    @classmethod
    async def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        async with cls._lock:
            for breaker in cls._breakers.values():
                breaker.reset()


class CircuitBreaker:
    """Circuit Breaker implementation with state management.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Circuit is tripped, requests fail fast
        - HALF_OPEN: Testing if service recovered
    
    Args:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close
        timeout: Seconds before attempting recovery (half-open)
        excluded_exceptions: Exception types that don't count as failures
        callbacks: List of callbacks for state change notifications
        half_open_max_calls: Max concurrent calls in half-open state
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        excluded_exceptions: tuple = (),
        callbacks: Optional[List[CircuitBreakerCallback]] = None,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.excluded_exceptions = excluded_exceptions
        self.callbacks = callbacks or [LoggingCallback()]
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        self.metrics = CircuitBreakerMetrics(service_name=name)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                return CircuitState.HALF_OPEN
        return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._opened_at is None:
            return False
        return (time.time() - self._opened_at) >= self.timeout
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state with callbacks."""
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        self.metrics.state = new_state
        self.metrics.state_changed_at = time.time()
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self.metrics.total_half_open_attempts += 1
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
        
        for callback in self.callbacks:
            try:
                callback.on_state_change(
                    self.name, old_state, new_state, self.metrics
                )
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        current_time = time.time()
        self._last_success_time = current_time
        self.metrics.last_success_time = current_time
        self.metrics.success_count += 1
        self.metrics.total_calls += 1
        self.metrics.consecutive_failures = 0
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self.metrics.consecutive_successes += 1
            self.metrics.half_open_successes += 1
            
            if self._success_count >= self.success_threshold:
                await self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self.metrics.consecutive_successes += 1
        
        for callback in self.callbacks:
            try:
                callback.on_call_success(self.name, self.metrics)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _on_failure(self, error: Optional[Exception] = None) -> None:
        """Handle failed call."""
        current_time = time.time()
        self._last_failure_time = current_time
        self.metrics.last_failure_time = current_time
        self.metrics.failure_count += 1
        self.metrics.total_calls += 1
        self.metrics.consecutive_successes = 0
        self.metrics.consecutive_failures += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self.metrics.half_open_failures += 1
            await self._transition_to(CircuitState.OPEN)
            self._opened_at = current_time
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                self._opened_at = current_time
        
        for callback in self.callbacks:
            try:
                callback.on_call_failure(self.name, self.metrics, error)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _on_rejected(self) -> None:
        """Handle rejected call when circuit is open."""
        self.metrics.rejected_calls += 1
    
    def _is_excluded_exception(self, error: Exception) -> bool:
        """Check if exception type is excluded from failure counting."""
        return isinstance(error, self.excluded_exceptions)
    
    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        current_state = self.state
        
        if current_state == CircuitState.CLOSED:
            return True
        
        if current_state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        
        return False
    
    def get_retry_after(self) -> float:
        """Get seconds until circuit might close."""
        if self._opened_at is None:
            return self.timeout
        elapsed = time.time() - self._opened_at
        return max(0, self.timeout - elapsed)
    
    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpen: When circuit is open
        """
        if not await self.can_execute():
            await self._on_rejected()
            raise CircuitBreakerOpen(
                self.name,
                self.get_retry_after()
            )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self._on_success()
            return result
        except self.excluded_exceptions:
            await self._on_success()
            raise
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def reset(self) -> None:
        """Reset circuit breaker to initial closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._opened_at = None
        self._half_open_calls = 0
        self.metrics = CircuitBreakerMetrics(service_name=self.name)
    
    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"state={self.state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    excluded_exceptions: tuple = (),
) -> Callable[[Callable[T], Callable[T]]]:
    """Decorator for protecting functions with circuit breaker.
    
    Usage:
        @circuit_breaker("external-api", failure_threshold=3)
        async def call_external_service():
            ...
    
    Args:
        name: Circuit breaker name/identifier
        failure_threshold: Failures before opening
        success_threshold: Successes in half-open to close
        timeout: Seconds before half-open attempt
        excluded_exceptions: Exceptions that don't count as failures
    """
    def decorator(func: Callable[T]) -> Callable[T]:
        _breaker: Optional[CircuitBreaker] = None
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _breaker
            if _breaker is None:
                _breaker = await CircuitBreakerRegistry.get_or_create(
                    name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    excluded_exceptions=excluded_exceptions,
                )
            return await _breaker.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal _breaker
            if _breaker is None:
                raise RuntimeError(
                    f"Circuit breaker decorator requires async function: {func.__name__}"
                )
            raise RuntimeError(
                f"Circuit breaker decorated function must be awaited: {func.__name__}"
            )
        
        wrapper.__wrapped__ = func
        wrapper._circuit_breaker_name = name
        wrapper._circuit_breaker = None
        
        return wrapper  # type: ignore
    
    return decorator


async def get_circuit_breaker(
    request: Request,
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    excluded_exceptions: tuple = (),
) -> CircuitBreaker:
    """FastAPI dependency for circuit breaker injection.
    
    Usage:
        @app.get("/external")
        async def call_service(cb: CircuitBreaker = Depends(get_circuit_breaker)):
            return await cb.execute(external_call)
    
    Args:
        request: FastAPI request object
        name: Circuit breaker identifier
        failure_threshold: Failures before opening
        success_threshold: Successes in half-open to close
        timeout: Seconds before half-open attempt
        excluded_exceptions: Exceptions that don't count as failures
    """
    return await CircuitBreakerRegistry.get_or_create(
        name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        excluded_exceptions=excluded_exceptions,
    )


class CircuitBreakerMiddleware:
    """Middleware for automatic circuit breaker on HTTP responses.
    
    Usage:
        app.add_middleware(
            CircuitBreakerMiddleware,
            on_force_close=lambda name: logger.critical(f"Circuit {name} force closed")