# pylint: disable=redefined-outer-name,unused-argument,unused-import
"""Test env preparation module."""
import os
from typing import Any, Generator

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.containers.containers import AppContainer
from src.routes import (  # noqa: F401
    detector_endpoints,
    health_endpoints,
    recognizer_endpoints,
)
from src.routes.routers import detector_router, health_router, recognizer_router

TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session")
def sample_image_bytes():
    """Fixture for loading a sample image as bytes.

    Yields:
        bytes: The loaded image in bytes format.
    """
    image_file = open(os.path.join(TESTS_DIR, "images", "image.jpg"), "rb")  # noqa: WPS515
    try:
        yield image_file.read()
    finally:
        image_file.close()


@pytest.fixture
def sample_image_np() -> NDArray[np.uint8]:
    """Fixture for loading a sample image and converting its color from BGR to RGB.

    Returns:
        NDArray[np.uint8]: The loaded image in RGB format.
    """
    img = cv2.imread(os.path.join(TESTS_DIR, "images", "image.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def app_config() -> DictConfig:
    """Fixture for loading the application configuration from a YAML file.

    Returns:
        DictConfig: The loaded application configuration.
    """
    return OmegaConf.load("configs/config.yml")  # type: ignore


@pytest.fixture
def app_container(app_config: DictConfig):  # noqa: WPS442
    """Fixture for creating an application container and setting the configuration.

    Args:
        app_config (DictConfig): The loaded application configuration.

    Returns:
        AppContainer: The application container with the configuration set.
    """
    container = AppContainer()
    container.config.from_dict(app_config)  # type: ignore
    return container


@pytest.fixture
def wired_app_container(app_config: Any) -> Generator[AppContainer, None, None]:  # noqa: WPS442
    """Fixture for creating and wiring an application container.

    Args:
        app_config (Any): The loaded application configuration.

    Yields:
        AppContainer: The wired application container.
    """
    container = AppContainer()
    container.config.from_dict(app_config)
    container.wire([detector_endpoints, recognizer_endpoints])
    yield container
    container.unwire()


@pytest.fixture
def test_app(wired_app_container: AppContainer):  # noqa: WPS442
    """Fixture for creating the FastAPI app.

    Args:
        wired_app_container (AppContainer): The wired application container.

    Returns:
        FastAPI: The FastAPI app with included routers.
    """
    app = FastAPI()
    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(detector_router, prefix="/detector", tags=["detector"])
    app.include_router(recognizer_router, prefix="/recognizer", tags=["recognizer"])
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:  # noqa: WPS442
    """Fixture for creating a test client for the FastAPI app.

    Args:
        test_app (FastAPI): The FastAPI app.

    Returns:
        TestClient: The test client for the FastAPI app.
    """
    return TestClient(test_app)
