"""
FastAPI API Versioning System

Provides comprehensive API versioning with support for:
- Path-based versioning (/v1/, /v2/)
- Header-based versioning (Accept-Version)
- Content-type versioning (application/vnd.api.version)
- Query-based versioning (?version=1.0)
- Automatic OpenAPI document generation per version
- Version negotiation with dependency injection
- Deprecation handling with automatic headers
- Version compatibility resolution

Example usage:
    from arc import FastAPI
    from arc.versioning import VersionedRouter, APIVersioning, Version

    # Create versioned routers
    v1_router = VersionedRouter(version="1.0", strategy=VersioningStrategy.PATH)
    v1_router.get("/items")(get_items_v1)

    v2_router = VersionedRouter(version="2.0", strategy=VersioningStrategy.PATH)
    v2_router.get("/items")(get_items_v2)

    # Mount with versioning
    app = FastAPI()
    versioning = APIVersioning(app, default_version="1.0")
    versioning.add_version("1.0", v1_router, title="Items API v1")
    versioning.add_version("2.0", v2_router, title="Items API v2")
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from arc import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from arc.routing import APIRoute
from pydantic import BaseModel, Field
from starlette import status

if TYPE_CHECKING:
    from arc.responses import JSONResponse

T = TypeVar("T")


class VersioningStrategy(str, Enum):
    """
    Supported API versioning strategies.
    
    PATH: Version in URL path (/v1/items, /v2/items)
    HEADER: Version in Accept-Version header
    CONTENT_TYPE: Version in Accept header (application/vnd.api.v1+json)
    QUERY: Version as query parameter (?version=1.0)
    """
    PATH = "path"
    HEADER = "header"
    CONTENT_TYPE = "content_type"
    QUERY = "query"


class Version(BaseModel):
    """
    Semantic version representation for API versioning.
    
    Supports semantic versioning with major.minor.patch format.
    Examples: 1.0, 2.1, 3.0.0-beta
    """
    major: int = Field(..., ge=0, description="Major version number")
    minor: int = Field(default=0, ge=0, description="Minor version number")
    patch: int = Field(default=0, ge=0, description="Patch version number")
    prerelease: Optional[str] = Field(default=None, description="Prerelease identifier")
    
    def __str__(self) -> str:
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version_str += f"-{self.prerelease}"
        return version_str
    
    def __repr__(self) -> str:
        return f"Version({self})"
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )
    
    def __lt__(self, other: Version) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)
        if self_tuple != other_tuple:
            return self_tuple < other_tuple
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        if self.prerelease:
            return True
        if other.prerelease:
            return False
        return False
    
    def __le__(self, other: Version) -> bool:
        return self == other or self < other
    
    def __gt__(self, other: Version) -> bool:
        return not self <= other
    
    def __ge__(self, other: Version) -> bool:
        return not self < other
    
    @classmethod
    def parse(cls, version_str: str) -> Version:
        """
        Parse a version string into a Version object.
        
        Accepts formats:
        - "1.0" -> major=1, minor=0, patch=0