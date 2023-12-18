"""This module provides the recognizer prediction endpoint for a inference service."""

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import recognizer_router
from src.services.recognizer import RecTorchWrapper


@recognizer_router.post("/recognize_barcode")
@inject
def recognize_barcode(
    image: bytes = File(),
    service: RecTorchWrapper = Depends(Provide[AppContainer.rec_model]),
):
    """
    Make a prediction on the given barcode image using the provided recognizer model.

    This endpoint takes an barcode image file in by tes and uses the `RecTorchWrapper` service to make a prediction.

    Args:
        image (bytes): The image file in bytes to make predictions on.
        service (RecTorchWrapper): The recognizer service to use for making predictions.

    Returns:
        str: Predicted symbols.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict(img)
