"""This module provides the segmentation prediction endpoint for a inference service."""

import base64

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import detector_router
from src.services.detector import SegTorchWrapper
from src.utils.processing import prepare_bbox


@detector_router.post("/predict_mask")  # type: ignore
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
    predicted_mask = service.predict_mask(img)
    mask_bytes = predicted_mask.tobytes()
    base64_encoded_mask = base64.b64encode(mask_bytes).decode("utf-8")
    return {"base64_encoded_mask": base64_encoded_mask}


@detector_router.post("/predict_barcodes")  # type: ignore
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
    bboxes = service.predict(img)
    return {"bboxes": [prepare_bbox(bbox) for bbox in bboxes]}
