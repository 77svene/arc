"""
API Versioning System for FastAPI.

Supports path-based, header-based, and content-type versioning strategies
with automatic OpenAPI document separation and version negotiation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Protocol,
    Union,
    Optional,
    Sequence,
    Type,
)
from collections import defaultdict
from copy import deepcopy
from functools import wraps
import inspect

from arc import APIRouter, FastAPI, Request, Response, HTTPException, Depends
from arc.responses import JSONResponse
from arc.types import DecoratorCallable
from arc.routing import APIRoute
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.routing import Route as StarletteRoute
from pydantic import BaseModel


# Type definitions
VersionStr = Union[str, tuple[int, ...]]
T = TypeVar("T")
DecoratedFunc = TypeVar("DecoratedFunc", bound=Callable[..., Any])


class VersioningStrategy(str, Enum):
    """Supported API versioning strategies.
    
    PATH: Version in URL path (/v1/items, /v2/users)
    HEADER: Version in request header (X-API-Version: 1.0)
    CONTENT_TYPE: Version in Accept header (application/vnd.api.v1+json)
    """
    PATH = "path"
    HEADER = "header"
    CONTENT_TYPE = "content_type"


@dataclass(frozen=True, slots=True)
class Version:
    """Immutable semantic version with comparison support.
    
    Supports versions like '1.0.0', 'v2', '2.1', etc.
    Automatically normalizes to major.minor.patch format.
    """
    raw: str
    
    def __post_init__(self) -> None:
        parts = self.raw.lstrip("vV").split(".")
        try:
            self._major = int(parts[0]) if parts and parts[0] else 0
            self._minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            self._patch = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        except ValueError:
            raise ValueError(f"Invalid version format: {self.raw}")
    
    @property
    def major(self) -> int:
        return self._major
    
    @property
    def minor(self) -> int:
        return self._minor
    
    @property
    def patch(self) -> int:
        return self._patch
    
    def __str__(self) -> str:
        return self.raw
    
    def __repr__(self) -> str:
        return f"Version('{self.raw}')"
    
    def __hash__(self) -> int:
        return hash(self.raw)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Version):
            return self.as_tuple == other.as_tuple
        if isinstance(other, str):
            return self.raw == other
        return NotImplemented
    
    def __lt__(self, other: Version | str) -> bool:
        if isinstance(other, str):
            other = Version(other)
        return self.as_tuple < other.as_tuple
    
    def __le__(self, other: Version | str) -> bool:
        if isinstance(other, str):
            other = Version(other)
        return self.as_tuple <= other.as_tuple
    
    def __gt__(self, other: Version | str) -> bool:
        if isinstance(other, str):
            other = Version(other)
        return self.as_tuple > other.as_tuple
    
    def __ge__(self, other: Version | str) -> bool:
        if isinstance(other, str):
            other = Version(other)
        return self.as_tuple >= other.as_tuple
    
    @property
    def as_tuple(self) -> tuple[int, int, int]:
        return (self._major, self._minor, self._patch)
    
    def is_compatible_with(self, other: Version | str, strict: bool = False) -> bool:
        """Check if this version is API-compatible with another."""
        if isinstance(other, str):
            other = Version(other)
        if strict:
            return self == other
        return self._major == other._major
    
    def supports(self, min_version: Version | str | None) -> bool:
        """Check if this version supports a minimum version requirement."""
        if min_version is None:
            return True
        if isinstance(min_version, str):
            min_version = Version(min_version)
        return self >= min_version


class VersioningError(Exception):
    """Raised when version negotiation or extraction fails."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        supported_versions: list[str] | None = None,
        available_versions: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.supported_versions = supported_versions
        self.available_versions = available_versions


@dataclass
class VersionInfo:
    """Metadata and configuration for a specific API version."""
    version: str
    deprecated: bool = False
    sunset_date: str | None = None
    migration_path: str | None = None
    migration_guide: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    
    def __post_init__(self) -> None:
        if not self.tags:
            self.tags = [f"v{self.version.replace('.', '_')}"]


class VersionExtractor(Protocol):
    """Protocol for custom version extraction strategies."""
    
    def extract(self, request: Request) -> str | None:
        """Extract version string from request."""
        ...
    
    def get_header_name(self) -> str:
        """Get the header name for this strategy."""
        ...


class PathVersionExtractor:
    """Extract version from URL path.
    
    Example: /v1/items -> extracts '1'
    """
    
    def __init__(
        self,
        prefix_pattern: str = "/v{version}",
        version_pattern: str = r"v?(\d+(?:\.\d+)?)",
    ):
        self.prefix_pattern = prefix_pattern
        self.version_pattern = version_pattern
        self._compiled = re.compile(version_pattern)
    
    def extract(self, request: Request) -> str | None:
        match = self._compiled.match(request.url.path)
        if match:
            return