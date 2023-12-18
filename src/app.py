# pylint: disable=wildcard-import,unused-wildcard-import,unused-import
"""App Entrypoint."""

from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes import detector_endpoints, recognizer_endpoints, health_endpoints
from src.routes.routers import detector_router, health_router, recognizer_router
from src.settings import app_settings
from src.utils.metrics import PrometheusMiddleware, metrics


def create_app() -> FastAPI:
    """
    Create a FastAPI instance with configured routes.

    Returns:
        FastAPI: An instance of the FastAPI application.
    """
    container = AppContainer()
    cfg = OmegaConf.load("configs/config.yml")
    container.config.from_dict(cfg)  # type: ignore
    container.wire([recognizer_endpoints, detector_endpoints])
    container.logger()

    app: FastAPI = FastAPI(
        title=app_settings.component_name,
        version=app_settings.service_version,
        description="Inference service for barcode recognition task.",
    )

    app.add_middleware(PrometheusMiddleware, filter_unhandled_paths=True)
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(detector_router, prefix="/detector", tags=["detector"])
    app.include_router(recognizer_router, prefix="/recognizer", tags=["recognizer"])
    app.add_route("/metrics", metrics)
    return app
