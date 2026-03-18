import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional
from typing import Callable, Awaitable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    Span = None
    Status = None
    StatusCode = None
    TracerProvider = None
    BatchSpanProcessor = None
    Resource = None
    SERVICE_NAME = None
    TraceContextTextMapPropagator = None
    Context = None


# Core context variables for async-safe request context
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)


@dataclass
class RequestContext:
    """Immutable container for request tracing information."""
    request_id: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_headers(self) -> dict:
        """Convert context to tracing headers for downstream propagation."""
        headers = {"X-Request-ID": self.request_id}
        if self.trace_id:
            headers["X-Trace-ID"] = self.trace_id
        return headers

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "metadata": self.metadata,
        }


def get_request_id() -> str:
    """Get current request ID from context, returns empty string if not set."""
    return request_id_var.get()


def get_trace_id() -> Optional[str]:
    """Get current trace ID from context."""
    return trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Get current span ID from context."""
    return span_id_var.get()


def get_current_context() -> Optional[RequestContext]:
    """Get current request context if available."""
    rid = request_id_var.get()
    if not rid:
        return None
    return RequestContext(
        request_id=rid,
        trace_id=trace_id_var.get(),
        span_id=span_id_var.get(),
        parent_span_id=parent_span_id_var.get(),
    )


def set_request_context(context: RequestContext) -> None:
    """Set all context variables from a RequestContext object."""
    request_id_var.set(context.request_id)
    trace_id_var.set(context.trace_id)
    span_id_var.set(context.span_id)
    parent_span_id_var.set(context.parent_span_id)


def clear_request_context() -> None:
    """Clear all context variables."""
    request_id_var.set("")
    trace_id_var.set(None)
    span_id_var.set(None)
    parent_span_id_var.set(None)


class TracingContext:
    """
    Context manager for manual span creation with automatic cleanup.
    
    Usage:
        async with TracingContext("my-operation") as ctx:
            # ctx contains current tracing info
            await some_async_operation()
    """

    def __init__(
        self,
        operation_name: str,
        attributes: Optional[dict] = None,
        tracer_name: str = "arc.tracing",
    ):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.tracer_name = tracer_name
        self._token = None
        self._span = None
        self._tracer = None
        self._context = None

    async def __aenter__(self) -> "TracingContext":
        if OPENTELEMETRY_AVAILABLE:
            self._tracer = trace.get_tracer(self.tracer_name)
            self._context = Context()
            self._span = self._tracer.start_span(
                self.operation_name,
                context=self._context,
                attributes=self.attributes,
            )
            # Update context vars with span info
            current_trace_id = format(trace_id_var.get() or "", "032x") if trace_id_var.get() else None
            current_span_id = format(self._span.context.span_id, "016x") if self._span.context else None
            
            if current_trace_id:
                trace_id_var.set(current_trace_id)
            span_id_var.set(current_span_id)
            parent_span_id_var.set(span_id_var.get())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._span and OPENTELEMETRY_AVAILABLE:
            if exc_type is not None:
                self._span.set_status(Status(StatusCode.ERROR))
                self._span.record_exception(exc_val)
            self._span.end()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request ID injection and distributed tracing.
    
    Features:
    - Generates UUID for each incoming request (or uses X-Request-ID from client)
    - Injects request ID into contextvars for async-safe access
    - Adds X-Request-ID header to all responses
    - Creates OpenTelemetry spans automatically
    - Propagates trace context from incoming headers
    """

    def __init__(
        self,
        app,
        service_name: str = "arc-service",
        header_name: str = "X-Request-ID",
        propagate_header: str = "X-Trace-ID",
        generate_id: Callable[[], str] = None,
        tracer_provider: Optional[Any] = None,
        propagate_trace_context: bool = True,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.header_name = header_name
        self.propagate_header = propagate_header
        self._generate_id = generate_id or (lambda: str(uuid.uuid4()))
        self._propagator = TraceContextTextMapPropagator() if OPENTELEMETRY_AVAILABLE else None
        self._propagate_trace_context = propagate_trace_context and OPENTELEMETRY_AVAILABLE
        self._tracer = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        
        if OPENTELEMETRY_AVAILABLE:
            try:
                from opentelemetry.sdk.trace import TracerProvider
                provider = trace.get_tracer_provider()
                if not isinstance(provider, TracerProvider):
                    resource = Resource.create({SERVICE_NAME: self.service_name})
                    provider = TracerProvider(resource=resource)
                    trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer(self.service_name)
            except Exception:
                self._tracer = None
                self._propagate_trace_context = False
        
        self._initialized = True

    def _extract_incoming_request_id(self, request: Request) -> Optional[str]:
        """Extract request ID from incoming headers."""
        return request.headers.get(self.header_name)

    def _extract_incoming_trace_context(self, request: Request) -> Optional[dict]:
        """Extract trace context from incoming headers for propagation."""
        if not self._propagate_trace_context:
            return None
        
        headers = {}
        for key in request.headers:
            headers[key.lower()] = request.headers[key]
        
        try:
            ctx = self._propagator.extract(carrier=headers)
            return ctx
        except Exception:
            return None

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        self._ensure_initialized()

        # Generate or extract request ID
        incoming_request_id = self._extract_incoming_request_id(request)
        request_id = incoming_request_id if incoming_request_id else self._generate_id()
        
        # Generate trace ID if not present
        trace_id = request.headers.get(self.propagate_header) or str(uuid.uuid4().hex[:32])
        
        # Set context variables
        token_request_id = request_id_var.set(request_id)
        token_trace_id = trace_id_var.set(trace_id)
        token_span_id = span_id_var.set(None)
        token_parent_span_id = parent_span_id_var.set(None)
        
        # Create span context for the request
        current_span = None
        extracted_context = None
        
        if OPENTELEMETRY_AVAILABLE and self._tracer:
            try:
                extracted_context = self._extract_incoming_trace_context(request)
                current_span = self._tracer.start_span(
                    f"{request.method} {request.url.path}",
                    context=extracted_context,
                    attributes={
                        "http.method": request.method,
                        "http.url": str(request.url),
                        "http.scheme": request.url.scheme,
                        "http.host": request.url.hostname or "",
                        "http.target": request.url.path,
                        "http.user_agent": request.headers.get("user-agent", ""),
                        "http.request_id": request_id,
                    },
                )
                span_id_var.set(format(current_span.context.span_id, "016x"))
            except Exception:
                current_span = None

        try:
            response = await call_next(request)
        except Exception as e:
            if current_span:
                current_span.set_status(Status(StatusCode.ERROR))
                current_span.record_exception(e)
            raise
        finally:
            # Update span with response info
            if current_span:
                current_span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 400:
                    current_span.set_status(Status(StatusCode.ERROR))
                else:
                    current_span.set_status(Status(StatusCode.OK))
                current_span.end()
            
            # Reset context variables
            request_id_var.reset(token_request_id)
            trace_id_var.reset(token_trace_id)
            span_id_var.reset(token_span_id)
            parent_span_id_var.reset(token_parent_span_id)

        # Add tracing headers to response
        response.headers[self.header_name] = request_id
        response.headers["X-Trace-ID"] = trace_id
        
        if current_span:
            span_id = span_id_var.get()
            if span_id:
                response.headers["X-Span-ID"] = span_id

        return response


class TracingDependency:
    """
    FastAPI dependency for injecting request tracing context into endpoints.
    
    Usage:
        @app.get("/items/{item_id}")
        async def get_item(
            item_id: str,
            tracing: TracingContext = Depends(TracingDependency()),
        ):
            return {"item_id": item_id, "request_id": tracing.request_id}
    """

    def __init__(self, required: bool = False):
        self.required = required

    async def __call__(
        self,
        request: Request,
    ) -> RequestContext:
        request_id = request_id_var.get()
        
        if self.required and not request_id:
            raise RuntimeError(
                "Request ID not found in context. "
                "Ensure RequestIDMiddleware is installed."
            )
        
        return RequestContext(
            request_id=request_id or "unknown",
            trace_id=trace_id_var.get(),
            span_id=span_id_var.get(),
            parent_span_id=parent_span_id_var.get(),
        )


# Convenience instance for common usage
tracing = TracingDependency()


async def get_request_context() -> RequestContext:
    """
    Async dependency function for accessing request tracing context.
    
    Usage:
        @app.get("/health")
        async def health(ctx: RequestContext = Depends(get_request_context)):
            return {"status": "ok", "request_id": ctx.request_id}
    """
    return RequestContext(
        request_id=request_id_var.get() or "unknown",
        trace_id=trace_id_var.get(),
        span_id=span_id_var.get(),
        parent_span_id=parent_span_id_var.get(),
    )


def create_trace_headers() -> dict:
    """
    Create headers for propagating trace context to downstream services.
    
    Usage:
        headers = create_trace_headers()
        headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
    """
    headers = {}
    
    request_id = request_id_var.get()
    if request_id:
        headers["X-Request-ID"] = request_id
    
    trace_id = trace_id_var.get()
    if trace_id:
        headers["X-Trace-ID"] = trace_id
    
    if OPENTELEMETRY_AVAILABLE:
        carrier = {}
        try:
            propagator = TraceContextTextMapPropagator()
            ctx = Context()
            propagator.inject(carrier, context=ctx)
            headers.update(carrier)
        except Exception:
            pass
    
    return headers


@dataclass
class TraceLogger:
    """
    Structured logger adapter that automatically includes trace context.
    
    Usage:
        logger = TraceLogger(logging.getLogger(__name__))
        logger.info("Processing request", extra={"operation": "process"})
    """
    logger: any
    
    def _get_extra(self, extra: Optional[dict] = None) -> dict:
        """Merge trace context with existing extra dict."""
        context = get_current_context()
        merged = {
            "request_id": context.request_id if context else "unknown",
            "trace_id": context.trace_id if context else None,
            "span_id": context.span_id if context else None,
        }
        if extra:
            merged.update(extra)
        return merged

    def debug(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        kwargs.setdefault("extra", self._get_extra(kwargs.get("extra")))
        self.logger.critical(msg, *args, **kwargs)


def setup_tracing(
    app,
    service_name: str = "arc-service",
    header_name: str = "X-Request-ID",
    propagate_trace_context: bool = True,
) -> RequestIDMiddleware:
    """
    Convenience function to set up request tracing on a FastAPI application.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service for OpenTelemetry
        header_name: Header name for request ID
        propagate_trace_context: Whether to propagate OpenTelemetry trace context
    
    Returns:
        The installed middleware instance
    """
    middleware = RequestIDMiddleware(
        app=app,
        service_name=service_name,
        header_name=header_name,
        propagate_trace_context=propagate_trace_context,
    )
    app.add_middleware(middleware.__class__, **{
        "service_name": service_name,
        "header_name": header_name,
        "propagate_trace_context": propagate_trace_context,
    })
    return middleware


__all__ = [
    "RequestIDMiddleware",
    "TracingDependency",
    "TracingContext",
    "RequestContext",
    "TraceLogger",
    "get_request_id",
    "get_trace_id",
    "get_span_id",
    "get_current_context",
    "set_request_context",
    "clear_request_context",
    "create_trace_headers",
    "setup_tracing",
    "get_request_context",
    "tracing",
    "request_id_var",
    "trace_id_var",
    "span_id_var",
    "parent_span_id_var",
]