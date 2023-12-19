"""This module provides the recognizer prediction endpoint for a inference service."""

from typing import Dict

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import recognizer_router
from src.services.detector import SegTorchWrapper
from src.services.recognizer import RecTorchWrapper
from src.utils.processing import prepare_bbox


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


@recognizer_router.post("/recognize_image")
@inject
def recognize_image(
    image: bytes = File(),
    recognizer_service: RecTorchWrapper = Depends(Provide[AppContainer.rec_model]),
    detector_service: SegTorchWrapper = Depends(Provide[AppContainer.seg_model]),
):
    """
    Make a prediction on the given barcode image using the provided recognizer model.

    This endpoint takes an barcode image file in by tes and uses the `RecTorchWrapper` service to make a prediction.

    Args:
        image (bytes): The image file in bytes to make predictions on.
        recognizer_service (RecTorchWrapper): The recognizer service to use for making predictions.
        detector_service (SegTorchWrapper): The segmentation service to use for making predictions.

    Returns:
        str: Predicted symbols.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    bboxes = detector_service.predict(img)

    preds: Dict[str, list] = {"barcodes": []}
    for bbox in bboxes:
        converted_bbox = prepare_bbox(bbox)
        barcode = img[
            converted_bbox["y_min"] : converted_bbox["y_max"],
            converted_bbox["x_min"] : converted_bbox["x_max"],
            :,
        ]
        rec_value = recognizer_service.predict(barcode)

        preds["barcodes"].append({"bbox": converted_bbox, "value": rec_value})

    return preds
