import uuid
import logging
from contextvars import ContextVar
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from functools import wraps
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Span = None


logger = logging.getLogger(__name__)

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
span_id_var: ContextVar[str] = ContextVar("span_id", default="")
parent_span_id_var: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)


@dataclass
class TracingConfig:
    service_name: str = "arc-service"
    request_id_header: str = "X-Request-ID"
    trace_id_header: str = "X-Trace-ID"
    otel_enabled: bool = True
    otel_exporter: Optional[str] = "console"
    propagate_trace_context: bool = True
    log_request_ids: bool = True
    include_hostname_in_span: bool = True


class TracingContext:
    __slots__ = ("request_id", "trace_id", "span_id", "parent_span_id", "start_time")
    
    def __init__(
        self,
        request_id: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.trace_id = trace_id or request_id
        self.span_id = span_id or ""
        self.parent_span_id = parent_span_id
        self.start_time = time.perf_counter()
    
    @property
    def duration_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000


class RequestIDMiddleware(BaseHTTPMiddleware):
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[TracingConfig] = None,
        tracer_provider: Optional[Any] = None,
    ):
        super().__init__(app)
        self.config = config or TracingConfig()
        self._tracer = None
        self._propagator = TraceContextTextMapPropagator() if OTEL_AVAILABLE else None
        
        if OTEL_AVAILABLE and self.config.otel_enabled:
            self._setup_otel(tracer_provider)
    
    def _setup_otel(self, tracer_provider: Optional[Any] = None) -> None:
        if not OTEL_AVAILABLE:
            return
        
        try:
            if tracer_provider is None:
                resource = Resource.create({
                    ResourceAttributes.SERVICE_NAME: self.config.service_name,
                })
                provider = TracerProvider(resource=resource)
                
                if self.config.otel_exporter == "console":
                    provider.add_span_processor(
                        BatchSpanProcessor(ConsoleSpanExporter())
                    )
                
                trace.set_tracer_provider(provider)
            
            self._tracer = trace.get_tracer(self.config.service_name)
            logger.info(f"OpenTelemetry tracing initialized for service: {self.config.service_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
            self._tracer = None
    
    def _generate_request_id(self, request: Request) -> str:
        incoming_id = request.headers.get(self.config.request_id_header)
        if incoming_id:
            return incoming_id
        
        incoming_trace = request.headers.get(self.config.trace_id_header)
        if incoming_trace:
            return incoming_trace
        
        return str(uuid.uuid4())
    
    def _extract_parent_span(self, request: Request) -> Optional[str]:
        if not OTEL_AVAILABLE or not self._propagator:
            return None
        
        try:
            ctx: Context = self._propagator.extract(request.headers)
            span = trace.get_current_span(ctx)
            if span:
                return format(span.get_span_context().span_id, '016x')
        except Exception:
            pass
        return None
    
    def _propagate_trace_context(self, request: Request) -> dict:
        headers = {}
        if OTEL_AVAILABLE and self.config.propagate_trace_context and self._propagator:
            try:
                ctx = trace.get_current_context()
                self._propagator.inject(headers)
            except Exception as e:
                logger.debug(f"Failed to inject trace context: {e}")
        return headers
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = self._generate_request_id(request)
        parent_span_id = self._extract_parent_span(request)
        
        request_id_var.set(request_id)
        trace_id_var.set(request_id)
        parent_span_id_var.set(parent_span_id)
        
        tracing_context = TracingContext(
            request_id=request_id,
            parent_span_id=parent_span_id,
        )
        
        request.state.tracing_context = tracing_context
        request.state.request_id = request_id
        
        span_name = f"{request.method} {request.url.path}"
        
        async def traced_call_next() -> Response:
            if OTEL_AVAILABLE and self._tracer:
                with self._tracer.start_as_current_span(
                    span_name,
                    kind=trace.SpanKind.SERVER,
                    attributes={
                        "http.method": request.method,
                        "http.url": str(request.url),
                        "http.target": request.url.path,
                        "http.host": request.url.hostname or "",
                        "http.scheme": request.url.scheme,
                        "http.user_agent": request.headers.get("user-agent", ""),
                        "http.request_id": request_id,
                        "http.client_ip": self._get_client_ip(request),
                    },
                ) as span:
                    span.set_attribute("http.trace_id", request_id)
                    if parent_span_id:
                        span.set_attribute("http.parent_span_id", parent_span_id)
                    
                    try:
                        response = await call_next(request)
                        
                        span.set_attribute("http.status_code", response.status_code)
                        
                        if response.status_code >= 500:
                            span.set_status(Status(StatusCode.ERROR))
                        elif response.status_code >= 400:
                            span.set_status(Status(StatusCode.ERROR, "Client error"))
                        else:
                            span.set_status(Status(StatusCode.OK))
                        
                        tracing_context.span_id = format(span.get_span_context().span_id, '016x')
                        
                        return response
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
            else:
                return await call_next(request)
        
        response = await traced_call_next()
        
        response.headers[self.config.request_id_header] = request_id
        
        if self.config.log_request_ids:
            duration = tracing_context.duration_ms
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration, 2),
                "client_ip": self._get_client_ip(request),
            }
            
            if response.status_code >= 500:
                logger.error("Request completed", extra=log_data)
            elif response.status_code >= 400:
                logger.warning("Request completed", extra=log_data)
            else:
                logger.info("Request completed", extra=log_data)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"


class TracingMiddleware(RequestIDMiddleware):
    pass


def get_request_id() -> str:
    return request_id_var.get()


def get_trace_id() -> str:
    return trace_id_var.get()


def get_span_id() -> str:
    return span_id_var.get()


def get_current_span() -> Optional["Span"]:
    if OTEL_AVAILABLE:
        return trace.get_current_span()
    return None


def get_tracing_context() -> Optional[TracingContext]:
    try:
        from starlette.requests import Request
        from starlette.middleware.base import _CachedRequest
        return getattr(Request, "__slots__", None)
    except Exception:
        pass
    return None


def create_span(
    name: str,
    attributes: Optional[dict] = None,
    kind: Optional[Any] = None,
) -> Callable:
    if not OTEL_AVAILABLE or trace is None:
        def noop_decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return noop_decorator
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            span_kind = kind or trace.SpanKind.INTERNAL
            
            with tracer.start_as_current_span(name, kind=span_kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("request_id", get_request_id())
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            span_kind = kind or trace.SpanKind.INTERNAL
            
            with tracer.start_as_current_span(name, kind=span_kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("request_id", get_request_id())
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def add_trace_attributes(**attributes) -> None:
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span:
            for key, value in attributes.items():
                span.set_attribute(key, value)


def record_exception(exception: Exception, attributes: Optional[dict] = None) -> None:
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span:
            span.record_exception(exception, attributes=attributes)
            span.set_status(Status(StatusCode.ERROR, str(exception)))


class TracingContextManager:
    
    def __init__(self, name: str, attributes: Optional[dict] = None):
        self.name = name
        self.attributes = attributes or {}
        self.tracer = None
        self.span = None
    
    def __enter__(self):
        if OTEL_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.span = self.tracer.start_span(self.name)
            
            for key, value in self.attributes.items():
                self.span.set_attribute(key, value)
            
            self.span.set_attribute("request_id", get_request_id())
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()
        
        return False
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class TracedHTTPClient:
    
    def __init__(self, client, service_name: str = "external-service"):
        self.client = client
        self.service_name = service_name
    
    async def request(self, method: str, url: str, **kwargs) -> Any:
        if not OTEL_AVAILABLE:
            return await self.client.request(method, url, **kwargs)
        
        tracer = trace.get_tracer(__name__)
        span_name = f"HTTP {method} {url}"
        
        with tracer.start_as_current_span(
            span_name,
            kind=trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)
            span.set_attribute("http.request_id", get_request_id())
            span.set_attribute("service.name", self.service_name)
            
            headers = kwargs.get("headers", {})
            headers["X-Request-ID"] = get_request_id()
            headers["X-Trace-ID"] = get_trace_id()
            kwargs["headers"] = headers
            
            try:
                response = await self.client.request(method, url, **kwargs)
                
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(
                    Status(StatusCode.OK if response.status_code < 400 else StatusCode.ERROR)
                )
                
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


__all__ = [
    "RequestIDMiddleware",
    "TracingMiddleware",
    "TracingConfig",
    "TracingContext",
    "TracingContextManager",
    "TracedHTTPClient",
    "get_request_id",
    "get_trace_id",
    "get_span_id",
    "get_current_span",
    "create_span",
    "add_trace_attributes",
    "record_exception",
    "request_id_var",
    "trace_id_var",
    "span_id_var",
    "parent_span_id_var",
]