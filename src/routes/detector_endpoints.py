"""This module provides the segmentation prediction endpoint for a inference service."""

import base64

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import detector_router
from src.services.detector import SegTorchWrapper


@detector_router.post("/predict_mask")
@inject
def predict_mask(
    image: bytes = File(),
    service: SegTorchWrapper = Depends(Provide[AppContainer.seg_model]),
):
    """
    Make a prediction on the given image using the provided segmentation model.

    This endpoint takes an image file in bytes and uses the `SegTorchWrapper` service
    to make a prediction and return the predicted mask.

    Args:
        image (bytes): The image file in bytes to make predictions on.
        service (SegTorchWrapper): The segmentartion service to use for making predictions.

    Returns:
        dict: A dictionary with the key 'objs' containing the sorted bboxes by confidence.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    mask_bytes = service.predict_mask(img).tobytes()
    base64_encoded_mask = base64.b64encode(mask_bytes).decode("utf-8")
    return {"base64_encoded_mask": base64_encoded_mask}


@detector_router.post("/predict_barcodes")
@inject
def predict_barcodes(
    image: bytes = File(),
    service: SegTorchWrapper = Depends(Provide[AppContainer.seg_model]),
):
    """
    Make a prediction on the given image using the provided segmentation model.

    This endpoint takes an image file in bytes and uses the `SegTorchWrapper` service
    to make a prediction and return the predicted barcodes.

    Args:
        image (bytes): The image file in bytes to make predictions on.
        service (SegTorchWrapper): The segmentation service to use for making predictions.

    Returns:
        List[List[int]]: Predicted bounding boxes in COCO format.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return {"barcodes": service.predict(img)}
