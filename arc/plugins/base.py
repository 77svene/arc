from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import entry_points
from inspect import Parameter, isasyncgen, isgenerator
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

P = TypeVar("P", bound="BasePlugin")


class HookPhase(Enum):
    """Phases in the request/response lifecycle."""
    EARLY_REQUEST = auto()
    REQUEST = auto()
    LATE_REQUEST = auto()
    EARLY_RESPONSE = auto()
    RESPONSE = auto()
    LATE_RESPONSE = auto()


@dataclass(frozen=True)
class PluginMetadata:
    """Metadata describing a plugin's identity and capabilities."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    website: str = ""
    tags: FrozenSet[str] = field(default_factory=frozen set)
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")


@dataclass
class RequestContext:
    """Mutable context object flowing through the request lifecycle."""
    request: Request
    state: Dict[str, Any] = field(default_factory=dict)
    plugin_data: Dict[str, Any] = field(default_factory=dict)
    app_state: Dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, key: str) -> Any:
        return self.state[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
    
    def has(self, key: str) -> bool:
        return key in self.state


@dataclass
class HookContext:
    """Context for individual hook execution."""
    phase: HookPhase
    plugin_name: str
    success: bool = True
    error: Optional[Exception] = None
    elapsed_ms: float = 0.0


class PluginHook:
    """A single hook callback with metadata."""
    
    __slots__ = ("callback", "phase", "order", "plugin_name", "_is_asyncgen")
    
    def __init__(
        self,
        callback: Callable[..., Any],
        phase: HookPhase,
        order: int,
        plugin_name: str,
    ) -> None:
        self.callback = callback
        self.phase = phase
        self.order = order
        self.plugin_name = plugin_name
        self._is_asyncgen = isasyncgen(callback)
    
    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        result = self.callback(*args, **kwargs)
        
        if asyncio.iscoroutine(result):
            result = await result
        
        return result
    
    def __repr__(self) -> str:
        return f"PluginHook({self.plugin_name}, {self.phase.name}, order={self.order})"


class PluginRegistry:
    """Thread-safe registry for all plugins."""
    
    def __init__(self) -> None:
        self._plugins: Dict[str, "BasePlugin"] = {}
        self._hooks: Dict[HookPhase, List[PluginHook]] = {
            phase: [] for phase in HookPhase
        }
        self._lock = asyncio.Lock()
    
    async def register(self, plugin: "BasePlugin") -> None:
        """Register a plugin and its hooks."""
        async with self._lock:
            name = plugin.metadata.name
            
            if name in self._plugins:
                raise ValueError(
                    f"Plugin '{name}' is already registered. "
                    f"Unregister it first before re-registering."
                )
            
            self._plugins[name] = plugin
            await self._register_hooks(plugin)
            logger.debug(f"Registered plugin: {name} v{plugin.metadata.version}")
    
    async def unregister(self, name: str) -> Optional["BasePlugin"]:
        """Unregister a plugin and remove its hooks."""
        async with self._lock:
            plugin = self._plugins.pop(name, None)
            
            if plugin is None:
                return None
            
            for hooks in self._hooks.values():
                hooks[:] = [h for h in hooks if h.plugin_name != name]
            
            logger.debug(f"Unregistered plugin: {name}")
            return plugin
    
    async def _register_hooks(self, plugin: "BasePlugin") -> None:
        """Extract and register all hooks from a plugin."""
        for phase in HookPhase:
            method_name = f"on_{phase.name.lower()}"
            
            if hasattr(plugin, method_name):
                method = getattr(plugin, method_name)
                
                if asyncio.iscoroutinefunction(method):
                    hook = PluginHook(
                        callback=method,
                        phase=phase,
                        order=getattr(plugin, "hook_order", lambda p: 0)(phase),
                        plugin_name=plugin.metadata.name,
                    )
                    self._hooks[phase].append(hook)
        
        for hooks in self._hooks.values():
            hooks.sort(key=lambda h: (h.order, h.plugin_name))
    
    def get_plugin(self, name: str) -> Optional["BasePlugin"]:
        """Get a registered plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List["BasePlugin"]:
        """List all registered plugins."""
        return list(self._plugins.values())
    
    def get_hooks(self, phase: HookPhase) -> List[PluginHook]:
        """Get all hooks for a specific phase in execution order."""
        return list(self._hooks.get(phase, []))


class LifecycleManager:
    """Manages startup and shutdown lifecycle events."""
    
    def __init__(self) -> None:
        self._startup_hooks: List[Tuple[int, str, Callable[[], Awaitable[None]]]] = []
        self._shutdown_hooks: List[Tuple[int, str, Callable[[], Awaitable[None]]]] = []
        self._lifespan_hooks: List[Tuple[int, str, Callable[[], AsyncGenerator[None, None]]]] = []
    
    def register_startup(self, callback: Callable[[], Awaitable[None]], order: int = 0, name: str = "") -> None:
        """Register a startup hook."""
        self._startup_hooks.append((order, name or f"hook_{len(self._startup_hooks)}", callback))
        self._startup_hooks.sort(key=lambda x: (x[0], x[1]))
    
    def register_shutdown(self, callback: Callable[[], Awaitable[None]], order: int = 0, name: str = "") -> None:
        """Register a shutdown hook."""
        self._shutdown_hooks.append((order, name or f"hook_{len(self._shutdown_hooks)}", callback))
        self._shutdown_hooks.sort(key=lambda x: (-x[0], x[1]))
    
    def register_lifespan(self, callback: Callable[[], AsyncGenerator[None, None]], order: int = 0, name: str = "") -> None:
        """Register a lifespan hook."""
        self._lifespan_hooks.append((order, name or f"lifespan_{len(self._lifespan_hooks)}", callback))
        self._lifespan_hooks.sort(key=lambda x: (x[0], x[1]))
    
    async def execute_startup(self) -> None:
        """Execute all startup hooks in order."""
        for _, name, callback in self._startup_hooks:
            try:
                logger.debug(f"Executing startup hook: {name}")
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Startup hook '{name}' failed: {e}")
                raise
    
    async def execute_shutdown(self) -> None:
        """Execute all shutdown hooks in reverse order."""
        for _, name, callback in reversed(self._shutdown_hooks):
            try:
                logger.debug(f"Executing shutdown hook: {name}")
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except