"""Module provides prometheus metrics middleware for Starlette."""
import os
from typing import Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SERVICE_NAME = "barcode_recognizer"
REQUESTS = Counter(
    f"{SERVICE_NAME}_starlette_requests_total",
    "Total count of requests by method and path.",
    ["method", "path_template"],
)

RESPONSES = Counter(
    f"{SERVICE_NAME}_starlette_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path_template", "status_code"],
)

REQUESTS_IN_PROGRESS = Gauge(
    f"{SERVICE_NAME}_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method"],
)

EXCEPTIONS = Counter(
    f"{SERVICE_NAME}_starlette_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path_template", "exception_type"],
)

# RAM stats
USED_RAM = Gauge(
    f"{SERVICE_NAME}_used_ram",
    "Gauge of ram currently being used by a worker in bytes",
)

TOTAL_USED_RAM = Gauge(
    f"{SERVICE_NAME}_total_used_ram",
    "Gauge of ram currently being used in bytes",
)


# pylint: disable=import-error,import-outside-toplevel
def register() -> CollectorRegistry:
    """
    Register and return a Prometheus CollectorRegistry.

    This function sets up a CollectorRegistry and registers a MultiProcessCollector to it. It is designed to be called
    at runtime, after process boot and after the environment variable has been set up.

    Returns:
        CollectorRegistry: A Prometheus CollectorRegistry instance with a MultiProcessCollector registered.
    """
    import prometheus_client
    from prometheus_client import multiprocess as prom_mp

    registry = prometheus_client.CollectorRegistry()
    prom_mp.MultiProcessCollector(registry)
    return registry


# pylint: disable=import-error,import-outside-toplevel
def metrics(_: Request) -> Response:
    """
    Generate a Prometheus metrics response.

    This function dynamically imports Prometheus client components and configures a metrics collector.
    If the application is running in a multiprocess environment (indicated by the 'prometheus_multiproc_dir'
    environment variable), it sets up a MultiProcessCollector. Otherwise, it uses the default registry.

    Args:
        _: Request - The incoming request. Not used in the function but required for interface compatibility.

    Returns:
        Response: A response object containing Prometheus metrics data.
    """
    import prometheus_client
    from prometheus_client import multiprocess as prom_mp

    if "prometheus_multiproc_dir" in os.environ:
        registry = prometheus_client.CollectorRegistry()
        prom_mp.MultiProcessCollector(registry)
    else:
        registry = REGISTRY

    return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})


class PrometheusMiddleware:
    """
    Middleware for integrating Prometheus monitoring in an ASGI application.

    This middleware collects and exposes metrics such as HTTP request counts,
    in-progress requests, and exceptions for Prometheus monitoring.

    Attributes:
        app (ASGIApp): The ASGI application instance.
        filter_unhandled_paths (bool): Flag to determine whether to filter out metrics for unhandled paths.
    """

    def __init__(self, app: ASGIApp, filter_unhandled_paths: bool = False) -> None:
        """
        Initialize the PrometheusMiddleware.

        Args:
            app (ASGIApp): The ASGI application to wrap with the middleware.
            filter_unhandled_paths (bool): If set to True, paths that are not explicitly handled by the application will
                be filtered out from metrics collection. Defaults to False.
        """
        self.app = app
        self.filter_unhandled_paths = filter_unhandled_paths

    @staticmethod
    def get_path_template(scope: Scope) -> Tuple[str, bool]:
        """
        Determine the path template for the current request.

        Args:
            scope (Scope): The scope of the request.

        Returns:
            Tuple[str, bool]: A tuple containing the path template and a boolean indicating whether the path is
                explicitly handled by the app.
        """
        conn = HTTPConnection(scope)
        for route in scope["app"].routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return route.path, True
        return conn.url.path, False

    def _is_path_filtered(self, is_handled_path: bool) -> bool:
        """
        Determine if the current path should be filtered based on middleware settings.

        Args:
            is_handled_path (bool): Indicates whether the current path is explicitly handled by the application.

        Returns:
            bool: True if the path should be filtered out, False otherwise.
        """
        return self.filter_unhandled_paths and not is_handled_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Asynchronous call method for the middleware.

        Handles incoming requests and collects metrics for Prometheus.

        Args:
            scope (Scope): The scope of the request, containing request details.
            receive (Receive): An awaitable callable yielding request events.
            send (Send): An awaitable callable used for sending response events.

        Raises:
            BaseException: Propagates exceptions from the application while ensuring metrics are recorded.
        """
        status_code = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path_template, is_handled_path = self.get_path_template(scope)
        if self._is_path_filtered(is_handled_path):
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        REQUESTS_IN_PROGRESS.labels(method=method).inc()
        REQUESTS.labels(method=method, path_template=path_template).inc()
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exp:
            exp_type = type(exp).__name__
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(method=method, path_template=path_template, exception_type=exp_type).inc()
            raise exp from None
        finally:
            RESPONSES.labels(method=method, path_template=path_template, status_code=status_code).inc()
            REQUESTS_IN_PROGRESS.labels(method=method).dec()
