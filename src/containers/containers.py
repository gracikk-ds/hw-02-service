# pylint: disable=c-extension-no-member,no-name-in-module
"""Containers for injection."""

import sys

import loguru
from dependency_injector import containers, providers
from dependency_injector.providers import Singleton
from loguru import logger
from rfc5424logging import Rfc5424SysLogHandler

from src.logger.log import DevelopFormatter
from src.services.detector import SegTorchWrapper
from src.services.recognizer import RecTorchWrapper
from src.settings import app_settings


class LoggerInitializer:
    """Class to handle the initialization and closing of logger."""

    def __init__(self):
        """Initialize the logger initializer."""
        self.develop_fmt = DevelopFormatter("InferenceService")
        self.syslog_handler = Rfc5424SysLogHandler(address=(app_settings.syslog_host, 9000))

    def init_logger(self) -> "loguru.Logger":
        """Initialize and configure the logger.

        Returns:
            loguru.Logger: The configured logger.
        """
        logger.remove()
        logger.add(sys.stderr, format=self.develop_fmt)  # type: ignore
        logger.add(self.syslog_handler, format=self.develop_fmt, serialize=True)  # type: ignore
        return logger

    def close_logger(self, my_logger: "loguru.Logger"):
        """Close and clean up the logger.

        Args:
            my_logger (loguru.Logger): The logger to be closed.
        """
        my_logger.remove()


class AppContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for managing application components.

    Args:
        containers.DeclarativeContainer: The base class for the dependency injection container.
    """

    config = providers.Configuration()

    """
    Singleton provider for the Segmentation Model component.

    Returns:
        checkpoint (str): path to model weights.
        device (str): device type
    """
    seg_model: Singleton[SegTorchWrapper] = Singleton(
        SegTorchWrapper,
        checkpoint=config.segmentation_model.checkpoint,
        device=config.segmentation_model.device,
    )

    """
    Singleton provider for the Recognizer Model component.

    Returns:
        checkpoint (str): path to model weights.
        device (str): device type
    """
    rec_model: Singleton[RecTorchWrapper] = Singleton(
        RecTorchWrapper,
        checkpoint=config.recognizer_model.checkpoint,
        device=config.recognizer_model.device,
    )

    """
    Singleton and Callable provider for the Logger resource.

    Returns:
        Singleton[LoggerInitializer]: An instance of the Logger component.
    """
    logger_initializer = providers.Singleton(LoggerInitializer)
    logger = providers.Callable(logger_initializer().init_logger)
