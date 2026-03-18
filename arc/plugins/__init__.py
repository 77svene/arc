"""
Plugin/Extension Architecture for FastAPI.

This module provides a plugin system that enables extending FastAPI applications
without modifying the core framework. Plugins can hook into the application
lifecycle, modify requests/responses, and register middleware.

Example usage:
    from arc import FastAPI
    from arc.plugins import PluginManager, PluginProtocol, middleware_plugin

    # Define a custom plugin
    class LoggingPlugin(PluginProtocol):
        name = "logging_plugin"
        priority = 100

        async def on_startup(self, app: "FastAPI") -> None:
            print("App starting up")

        async def on_request(self, request: Request) -> Optional[Request]:
            logger.info(f"Incoming request: {request.url}")
            return request

        async def on_response(self, request: Request, response: Response) -> Response:
            logger.info(f"Response status: {response.status_code}")
            return response

    # Or use the decorator
    @middleware_plugin(name="my_plugin", priority=50)
    async def my_plugin_hook(request: Request, call_next):
        # Custom logic
        response = await call_next(request)
        return response

    # Use in app
    app = FastAPI()
    app.include_plugin(LoggingPlugin())
    app.include_plugin(my_plugin_hook)
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from typing_extensions import ParamSpec, Self, TypeAlias

if TYPE_CHECKING:
    from arc import FastAPI
    from starlette.routing import Match

logger = logging.getLogger(__name__)


# Type definitions
P = ParamSpec("P")
T = TypeVar("T")
HookResult: TypeAlias = Optional[Union[Response, "ModifiedContent"]]
MiddlewareType: TypeAlias = Callable[
    [Request, RequestResponseEndpoint],
    Coroutine[Any, Any, Response],
]


class PluginHookType(Enum):
    """Types of hooks a plugin can implement."""
    
    STARTUP = auto()
    SHUTDOWN = auto()
    BEFORE_REQUEST = auto()
    AFTER_REQUEST = auto()
    ON_ERROR = auto()
    MIDDLEWARE = auto()


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Plugin name is required")


@dataclass
class ModifiedContent:
    """
    Container for modified request/response content.
    
    Allows plugins to pass modified data through the chain.
    """
    
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        if not isinstance(self.headers, dict):
            self.headers = {}


@dataclass
class PluginContext:
    """
    Context object passed to plugins containing request/response information.
    
    This provides a clean interface for plugins to access and modify
    request/response data without directly coupling to FastAPI internals.
    """
    
    request: Request
    response: Optional[Response] = None
    app: Optional[FastAPI] = None
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value that persists through the plugin chain."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value from the plugin chain."""
        return self.state.get(key, default)


@runtime_checkable
class PluginProtocol(Protocol):
    """
    Protocol defining the plugin interface.
    
    Plugins can implement any subset of these methods. All methods are
    optional and will only be called if implemented by the plugin.
    
    Example:
        class MyPlugin:
            name = "my_plugin"
            
            async def on_startup(self, app: FastAPI) -> None:
                await init_database()
            
            async def on_request(self, request: Request) -> Optional[Request]:
                # Modify or return None to reject
                return request
            
            async def on_response(self, request: Request, response: Response) -> Response:
                # Modify response
                response.headers["X-Plugin"] = "modified"
                return response
    """
    
    @property
    def name(self) -> str:
        """Unique identifier for the plugin."""
        ...
    
    @property
    def priority(self) -> int:
        """Plugin priority (lower = higher priority, runs first)."""
        return 100
    
    async def on_startup(self, app: FastAPI) -> None:
        """Called when the application starts."""
        ...
    
    async def on_shutdown(self, app: FastAPI) -> None:
        """Called when the application shuts down."""
        ...
    
    async def on_request(self, request: Request) -> Optional[Request]:
        """
        Called before processing a request.
        
        Return the (possibly modified) request, or None to short-circuit
        the request and return a custom response.
        """
        ...
    
    async def on_response(self, request: Request, response: Response) -> Response:
        """
        Called after processing a request but before returning the response.
        
        Return the (possibly modified) response.
        """
        ...
    
    async def on_error(
        self, request: Request, exc: Exception
    ) -> Optional[Response]:
        """
        Called when an error occurs during request processing.
        
        Return a custom error response, or None to use the default error handler.
        """
        ...
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(name=self.name)


class BasePlugin(ABC):
    """Abstract base class for plugins with default implementations."""
    
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    priority: int = 100
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name or self.__class__.__name__,
            version=self.version,
            description=self.description,
            author=self.author,
            priority=self.priority,
        )
    
    async def on_startup(self, app: FastAPI) -> None:
        """Called on application startup. Override for custom behavior."""
        pass
    
    async def on_shutdown(self, app: FastAPI) -> None:
        """Called on application shutdown. Override for custom behavior."""
        pass
    
    async def on_request(self, request: Request) -> Optional[Request]:
        """Called before request processing. Override for custom behavior."""
        return request
    
    async def on_response(self, request: Request, response: Response) -> Response:
        """Called after request processing. Override for custom behavior."""
        return response
    
    async def on_error(self, request: Request, exc: Exception) -> Optional[Response]:
        """Called on error. Override for custom behavior."""
        return None


class MiddlewarePlugin(BaseHTTPMiddleware):
    """
    A plugin that wraps the entire request processing as middleware.
    
    This is useful for plugins that need full control over the request/response
    cycle, including authentication, rate limiting, etc.
    
    Example:
        class RateLimitPlugin(MiddlewarePlugin):
            def __init__(self, app, max_requests: int = 100):
                super().__init__(app)
                self.max_requests = max_requests
            
            async def dispatch(self, request: Request, call_next):
                # Rate limiting logic
                ...
    """
    
    def __init__(self, app: FastAPI, **kwargs: Any) -> None:
        self.plugin_name = kwargs.pop("name", self.__class__.__name__)
        self.plugin_priority = kwargs.pop("priority", 100)
        super().__init__(app, **kwargs)
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request through the middleware.
        
        Override this method for custom middleware behavior.
        """
        return await call_next(request)


class PluginManager:
    """
    Manages plugin registration and lifecycle.
    
    The manager handles:
    - Plugin registration and ordering by priority
    - Executing lifecycle hooks
    - Request/response modification chains
    - Error handling
    
    Example:
        manager = PluginManager()
        manager.register(MyPlugin())
        
        async with manager.lifecycle(app):
            # App is running with plugins active
            pass
    """
    
    def __init__(self) -> None:
        self._plugins: OrderedDict[str, PluginProtocol] = OrderedDict()
        self._middleware_plugins: List[MiddlewarePlugin] = []
        self._hook_handlers: Dict[PluginHookType, List[Callable]] = {
            hook_type: [] for hook_type in PluginHookType
        }
        self._app: Optional[FastAPI] = None
        self._initialized: bool = False
    
    def register(self, plugin: PluginProtocol) -> Self:
        """
        Register a plugin with the manager.
        
        Plugins are sorted by priority (lower = higher priority).
        Duplicate plugin names are not allowed.
        
        Args:
            plugin: The plugin instance to register.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        name = plugin.name
        if name in self._plugins:
            raise ValueError(
                f"Plugin '{name}' is already registered. "
                "Use force_register() to replace."
            )
        
        self._plugins[name] = plugin
        
        # Re-sort by priority
        self._plugins = OrderedDict(
            sorted(
                self._plugins.items(),
                key=lambda x: x[1].priority
            )
        )
        
        # Register middleware plugins
        if isinstance(plugin, MiddlewarePlugin):
            self._middleware_plugins.append(plugin)
        
        logger.debug(f"Registered plugin: {name} (priority={plugin.priority})")
        return self
    
    def force_register(self, plugin: PluginProtocol) -> Self:
        """
        Register a plugin, replacing any existing plugin with the same name.
        
        Args:
            plugin: The plugin instance to register.
            
        Returns:
            Self for method chaining.
        """
        name = plugin.name
        if name in self._plugins:
            old_plugin = self._plugins[name]
            if isinstance(old_plugin, MiddlewarePlugin):
                self._middleware_plugins.remove(old_plugin)
            logger.debug(f"Replacing plugin: {name}")
        
        return self.register(plugin)
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            name: The name of the plugin to unregister.
            
        Returns:
            True if the plugin was found and removed, False otherwise.
        """
        if name in self._plugins:
            plugin = self._plugins.pop(name)
            if isinstance(plugin, MiddlewarePlugin):
                self._middleware_plugins.remove(plugin)
            logger.debug(f"Unregistered plugin: {name}")
            return True
        return False
    
    def get_plugin(self, name: str) -> Optional[PluginProtocol]:
        """Get a registered plugin by name."""
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> List[PluginProtocol]:
        """Get all registered plugins in priority order."""
        return list(self._plugins.values())
    
    def clear(self) -> None:
        """Unregister all plugins."""
        self._plugins.clear()
        self._middleware_plugins.clear()
        self._initialized = False
    
    @property
    def plugin_count(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)
    
    async def execute_startup(self, app: FastAPI) -> None:
        """
        Execute the startup hook for all plugins.
        
        Args:
            app: The FastAPI application instance.
        """
        self._app = app
        logger.info(f"Starting {self.plugin_count} plugins...")
        
        for plugin in self._plugins.values():
            try:
                if hasattr(plugin, "on_startup"):
                    result = plugin.on_startup(app)
                    if hasattr(result, "__await__"):
                        await result
                logger.debug(f"Startup completed for plugin: {plugin.name}")
            except Exception as e:
                logger.error(
                    f"Error in {plugin.name}.on_startup: {e}",
                    exc_info=True
                )
                raise
        
        self._initialized = True
        logger.info("All plugins started successfully")
    
    async def execute_shutdown(self, app: FastAPI) -> None:
        """
        Execute the shutdown hook for all plugins.
        
        Args:
            app: The FastAPI application instance.
        """
        logger.info("Shutting down plugins...")
        
        # Shutdown in reverse priority order
        for plugin in reversed(list(self._plugins.values())):
            try:
                if hasattr(plugin, "on_shutdown"):
                    result = plugin.on_shutdown(app)
                    if hasattr(result, "__await__"):
                        await result
                logger.debug(f"Shutdown completed for plugin: {plugin.name}")
            except Exception as e:
                logger.error(
                    f"Error in {plugin.name}.on_shutdown: {e}",
                    exc_info=True
                )
        
        self._initialized = False
        self._app = None
        logger.info("All plugins shut down")
    
    async def process_request(self, request: Request) -> Optional[Request]:
        """
        Process a request through all registered plugins.
        
        Plugins can modify the request or return None to short-circuit
        the request processing.
        
        Args:
            request: The incoming request.
            
        Returns:
            The (possibly modified) request, or None to short-circuit.
        """
        current_request = request
        
        for plugin in self._plugins.values():
            try:
                if hasattr(plugin, "on_request"):
                    result = plugin.on_request(current_request)
                    if hasattr(result, "__await__"):
                        result = await result
                    if result is None:
                        logger.debug(
                            f"Plugin {plugin.name} short-circuited request"
                        )
                        return None
                    current_request = result
            except Exception as e:
                logger.error(
                    f"Error in {plugin.name}.on_request: {e}",
                    exc_info=True
                )
                error_response = await self._handle_error(request, e)
                if error_response:
                    return error_response
        
        return current_request
    
    async def process_response(
        self, request: Request, response: Response
    ) -> Response:
        """
        Process a response through all registered plugins.
        
        Plugins can modify the response before it's returned to the client.
        
        Args:
            request: The original request.
            response: The response to process.
            
        Returns:
            The (possibly modified) response.
        """
        current_response = response
        
        for plugin in self._plugins.values():
            try:
                if hasattr(plugin, "on_response"):
                    result = plugin.on_response(request, current_response)
                    if hasattr(result, "__await__"):
                        result = await result
                    current_response = result
            except Exception as e:
                logger.error(
                    f"Error in {plugin.name}.on_response: {e}",
                    exc_info=True
                )
        
        return current_response
    
    async def _handle_error(
        self, request: Request, exc: Exception
    ) -> Optional[Response]:
        """
        Handle an error through registered plugins.
        
        Args:
            request: The request that caused the error.
            exc: The exception that occurred.
            
        Returns:
            A custom error response if a plugin provides one, None otherwise.
        """
        for plugin in self._plugins.values():
            try:
                if hasattr(plugin, "on_error"):
                    result = plugin.on_error(request, exc)
                    if hasattr(result, "__await__"):
                        result = await result
                    if result is not None:
                        return result
            except Exception as e:
                logger.error(
                    f"Error in {plugin.name}.on_error: {e}",
                    exc_info=True
                )
        
        return None
    
    @asynccontextmanager
    async def lifecycle(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """
        Context manager for plugin lifecycle.
        
        Automatically executes startup and shutdown hooks.
        
        Args:
            app: The FastAPI application instance.
            
        Example:
            manager = PluginManager()
            manager.register(MyPlugin())
            
            async with manager.lifecycle(app):
                # Plugins are active
                uvicorn.run(app)
        """
        await self.execute_startup(app)
        try:
            yield
        finally:
            await self.execute_shutdown(app)
    
    def get_middleware_stack(self) -> List[MiddlewarePlugin]:
        """Get the middleware plugin stack."""
        return list(self._middleware_plugins)


def middleware_plugin(
    func: Optional[MiddlewareType] = None,
    *,
    name: Optional[str] = None,
    priority: