"""
FastAPI Database Connection Pool Management

Provides built-in database connection pool management with health checks,
automatic reconnection, configurable limits, and metrics monitoring.

Example usage:
    from arc import FastAPI, Depends
    from arc.databases import DatabasePool, PostgresPool, PoolDependency
    
    app = FastAPI()
    
    # Configure pool at startup
    pool = PostgresPool(
        dsn="postgresql://user:pass@localhost/db",
        min_size=5,
        max_size=20,
        health_check_interval=30.0
    )
    
    @app lifespan
    async def lifespan_manager(app: FastAPI):
        async with pool:
            yield
    
    # Use in routes
    @app.get("/users/{user_id}")
    async def get_user(
        user_id: int,
        db: DatabasePool = Depends(PoolDependency(pool))
    ):
        async with db.connection() as conn:
            result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return dict(result) if result else None
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from arc import Depends, FastAPI
from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)


class PoolStatus(Enum):
    """Connection pool status states."""
    INITIALIZING = auto()
    READY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    SHUTTING_DOWN = auto()
    SHUTDOWN = auto()


class ConnectionStatus(Enum):
    """Individual connection status."""
    IDLE = auto()
    IN_USE = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    RECONNECTING = auto()


P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class PoolConfig:
    """Configuration for database connection pool."""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: float = 3600.0
    pool_pre_ping: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    max_reconnect_attempts: int = 3
    reconnect_backoff_base: float = 1.0
    connection_timeout: float = 10.0
    
    def __post_init__(self):
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if self.max_size < self.min_size:
            raise ValueError("max_size must be >= min_size")
        if self.max_overflow < 0:
            raise ValueError("max_overflow must be non-negative")
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""
    total_connections: int = 0
    idle_connections: int = 0
    active_connections: int = 0
    overflow_connections: int = 0
    waiting_tasks: int = 0
    health_check_failures: int = 0
    connection_errors: int = 0
    queries_executed: int = 0
    queries_failed: int = 0
    total_query_time: float = 0.0
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    
    @property
    def avg_query_time(self) -> float:
        """Average query execution time."""
        if self.queries_executed == 0:
            return 0.0
        return self.total_query_time / self.queries_executed
    
    @property
    def success_rate(self) -> float:
        """Query success rate as percentage."""
        total = self.queries_executed + self.queries_failed
        if total == 0:
            return 100.0
        return (self.queries_executed / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_connections": self.total_connections,
            "idle_connections": self.idle_connections,
            "active_connections": self.active_connections,
            "overflow_connections": self.overflow_connections,
            "waiting_tasks": self.waiting_tasks,
            "health_check_failures": self.health_check_failures,
            "connection_errors": self.connection_errors,
            "queries_executed": self.queries_executed,
            "queries_failed": self.queries_failed,
            "avg_query_time_ms": round(self.avg_query_time * 1000, 2),
            "success_rate": round(self.success_rate, 2),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_error": self.last_error,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


@dataclass
class HealthCheckResult:
    """Result of a pool health check."""
    healthy: bool
    latency_ms: float
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatabaseConnection(Protocol):
    """Protocol for database connections."""
    
    async def execute(self, query: str, *args: Any) -> Any:
        """Execute a query without returning results."""
        ...
    
    async def fetch(self, query: str, *args: Any) -> List[Any]:
        """Execute query and return all results."""
        ...
    
    async def fetchrow(self, query: str, *args: Any) -> Optional[Any]:
        """Execute query and return single row."""
        ...
    
    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute query and return single value."""
        ...
    
    async def close(self) -> None:
        """Close the connection."""
        ...
    
    async def __aenter__(self) -> DatabaseConnection:
        """Enter async context."""
        ...
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        ...


@runtime_checkable
class BaseDatabasePool(Protocol):
    """Protocol for database connection pools."""
    
    @property
    def config(self) -> PoolConfig:
        """Pool configuration."""
        ...
    
    @property
    def status(self) -> PoolStatus:
        """Current pool status."""
        ...
    
    @property
    def metrics(self) -> PoolMetrics:
        """Pool metrics."""
        ...
    
    @asynccontextmanager
    async def connection(
        self, timeout: Optional[float] = None
    ) -> AsyncGenerator[DatabaseConnection, None]:
        """Acquire a connection from the pool."""
        ...
    
    async def initialize(self) -> None:
        """Initialize the pool."""
        ...
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        ...
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check on the pool."""
        ...
    
    async def reconnect(self) -> None:
        """Attempt to reconnect the pool."""
        ...


class ConnectionWrapper:
    """Wraps a database connection with metrics tracking."""
    
    def __init__(
        self,
        connection: Any,
        pool_metrics: PoolMetrics,
        on_release: Callable[[], None],
    ):
        self._connection = connection
        self._metrics = pool_metrics
        self._on_release = on_release
        self._acquired_at: Optional[float] = None
    
    async def execute(self, query: str, *args: Any) -> Any:
        """Execute query with timing."""
        return await self._execute_with_metrics(
            self._connection.execute, query, *args
        )
    
    async def fetch(self, query: str, *args: Any) -> List[Any]:
        """Fetch all results with timing."""
        return await self._execute_with_metrics(
            self._connection.fetch, query, *args
        )
    
    async def fetchrow(self, query: str, *args: Any) -> Optional[Any]:
        """Fetch single row with timing."""
        return await self._execute_with_metrics(
            self._connection.fetchrow, query, *args
        )
    
    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch single value with timing."""
        return await self._execute_with_metrics(
            self._connection.fetchval, query, *args
        )
    
    async def _execute_with_metrics(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function with metrics tracking."""
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            self._metrics.queries_executed += 1
            return result
        except Exception as e:
            self._metrics.queries_failed += 1
            self._metrics.last_error = str(e)
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            self._metrics.total_query_time += elapsed
    
    async def close(self) -> None:
        """Close the wrapped connection."""
        await self._connection.close()
    
    async def __aenter__(self) -> ConnectionWrapper:
        """Enter context manager."""
        self._acquired_at = time.perf_counter()
        await self._connection.__aenter__()
        return self
    
    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exit context manager."""
        await self._connection.__aexit__(exc_type, exc_val, exc_tb)
        if self._acquired_at:
            connection_time = time.perf_counter() - self._acquired_at
            logger.debug(
                f"Connection held for {connection_time:.3f}s"
            )
        self._on_release()


class PoolState:
    """Manages internal pool state with thread-safety."""
    
    def __init__(self, config: PoolConfig):
        self.config = config
        self.status = PoolStatus.INITIALIZING
        self.metrics = PoolMetrics()
        self._connections: asyncio.Queue = asyncio.Queue()
        self._total_connections = 0
        self._overflow_connections = 0
        self._waiting_tasks = 0
        self._lock = asyncio.Lock()
        self._start_time = time.perf_counter()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    def update_metrics(self) -> None:
        """Update metrics from current state."""
        self.metrics.total_connections = self._total_connections
        self.metrics.idle_connections = self._connections.qsize()
        self.metrics.active_connections = (
            self._total_connections - self._connections.qsize()
        )
        self.metrics.overflow_connections = self._overflow_connections
        self.metrics.waiting_tasks = self._waiting_tasks
        self.metrics.uptime_seconds = time.perf_counter() - self._start_time


class PostgresPool(ABC):
    """
    PostgreSQL connection pool using asyncpg.
    
    Features:
    - Configurable connection limits
    - Automatic health checks
    - Connection metrics tracking
    - Automatic reconnection on failure
    - Context manager support for lifespan integration
    
    Example:
        pool = PostgresPool(
            dsn="postgresql://user:pass@localhost/db",
            min_size=5,
            max_size=20
        )
        
        async with pool:
            # pool is ready
            async with pool.connection() as conn:
                result = await conn.fetch("SELECT * FROM users")
    """
    
    def __init__(
        self,
        dsn: str,
        config: Optional[PoolConfig] = None,
        **connect_kwargs: Any,
    ):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            dsn: Database connection string.
            config: Pool configuration. Uses defaults if not provided.
            **connect_kwargs: Additional arguments passed to asyncpg.connect.
        """
        self._dsn = dsn
        self._config = config or PoolConfig()
        self._connect_kwargs = connect_kwargs
        self._state = PoolState(self._config)
        self._pool: Optional[Any] = None
        self._initialized = False
        self._closing = False
    
    @property
    def config(self) -> PoolConfig:
        """Pool configuration."""
        return self._config
    
    @property
    def status(self) -> PoolStatus:
        """Current pool status."""
        return self._state.status
    
    @property
    def metrics(self) -> PoolMetrics:
        """Pool metrics."""
        return self._state.metrics
    
    @property
    def dsn(self) -> str:
        """Database connection string (masked)."""
        return self._mask_dsn(self._dsn)
    
    def _mask_dsn(self, dsn: str) -> str:
        """Mask sensitive information in DSN."""
        import re
        # Mask password
        masked = re.sub(
            r'(://[^:]+:)[^@]+(@)',
            r'\1****\2',
            dsn
        )
        return masked
    
    async def initialize(self) -> None:
        """
        Initialize the connection pool.
        
        Creates initial connections up to min_size.
        Starts background health check task.
        """
        if self._initialized:
            return
        
        try:
            import asyncpg
            
            self._state.status = PoolStatus.INITIALIZING
            logger.info(f"Initializing PostgreSQL pool: {self.dsn}")
            
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._config.min_size,
                max_size=self._config.max_size,
                max_queries=self._config.max_overflow * 1000,
                timeout=self._config.connection_timeout,
                command_timeout=self._config.pool_timeout,
                **self._connect_kwargs,
            )
            
            # Warm up connections
            warmup_tasks = []
            for _ in range(self._config.min_size):
                warmup_tasks.append(self._pool.acquire())
            
            connections = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            for conn in connections:
                if isinstance(conn, Exception):
                    logger.warning(f"Failed to warm up connection: {conn}")
                else:
                    await self._pool.release(conn)
                    await self._state._connections.put(conn)
            
            self._state._total_connections = self._config.min_size
            self._state.status = PoolStatus.READY
            self._initialized = True
            
            # Start health check task
            self._state._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
            
            self._state.update_metrics()
            logger.info(
                f"PostgreSQL pool initialized: "
                f"connections={self._state._total_connections}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize pool: {e}")
            self._state.status = PoolStatus.UNHEALTHY
            self._state.metrics.last_error = str(e)
            raise
    
    async def close(self) -> None:
        """Close all connections and stop health checks."""
        if self._closing:
            return
        
        self._closing = True
        self._state.status = PoolStatus.SHUTTING_DOWN
        logger.info("Shutting down PostgreSQL pool...")
        
        # Stop health check task
        if self._state._health_check_task:
            self._state._health_check_task.cancel()
            try:
                await self._state._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close pool
        if self._pool:
            await self._pool.close()
        
        self._state.status = PoolStatus.SHUTDOWN
        self._initialized = False
        logger.info("PostgreSQL pool shut down")
    
    @asynccontextmanager
    async def connection(
        self, timeout: Optional[float] = None
    ) -> AsyncGenerator[ConnectionWrapper, None]:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Optional timeout for acquiring connection.
                     Defaults to pool_timeout from config.
        
        Yields:
            ConnectionWrapper wrapping the database connection.
        
        Raises:
            asyncio.TimeoutError: If connection cannot be acquired in time.
            PoolClosedError: If pool is closed.
        """
        if not self._initialized or not self._pool:
            raise PoolClosedError("Pool is not initialized")
        
        if self._closing:
            raise PoolClosedError("Pool is closing")
        
        timeout = timeout or self._config.pool_timeout
        self._state._waiting_tasks += 1
        
        try:
            conn = await asyncio.wait_for(
                self._pool.acquire(),
                timeout=timeout
            )
            self._state._waiting_tasks -= 1
            
            yield ConnectionWrapper(
                conn,
                self._state.metrics,
                lambda: asyncio.create_task(self._pool.release(conn))
            )
            
        except asyncio.TimeoutError:
            self._state._waiting_tasks -= 1
            self._state.metrics.connection_errors += 1
            logger.warning(
                f"Connection acquire timeout after {timeout}s. "