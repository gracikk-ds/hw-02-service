"""This module contains tests for predicting bboxes using a FastAPI application.

The tests use FastAPI's TestClient to send HTTP requests to the application,
and then check the responses to ensure that they are correct.
"""
import base64
from http import HTTPStatus

import numpy as np
from fastapi.testclient import TestClient


def test_predict_barcodes(client: TestClient, sample_image_bytes: bytes):
    """Test the predict_barcodes endpoint of the detector in the FastAPI application.

    This test sends a POST request to the "/detector/predict_barcodes" endpoint of the
    FastAPI application, including a sample image in the request data.
    It then checks that the response has a 200 status code, and that the response
    data includes a dict of list of predicted bbox coords.

    Args:
        client (TestClient): The test client used to send requests to the application.
        sample_image_bytes (bytes): The byte representation of a sample image.

    Raises:
        AssertionError: If the response status code is not 200, or if the response
            data does not include a  dict of list of predicted bbox coords.
    """
    files = {
        "image": sample_image_bytes,
    }

    response = client.post("/detector/predict_barcodes", files=files)
    assert response.status_code == HTTPStatus.OK  # noqa: S101

    predicted_bboxes = response.json()
    assert isinstance(predicted_bboxes, dict)  # noqa: S101
    assert isinstance(predicted_bboxes["bboxes"], list)  # noqa: S101
    assert isinstance(predicted_bboxes["bboxes"][0], dict)  # noqa: S101


def test_predict_mask(client: TestClient, sample_image_bytes: bytes, sample_image_np: np.ndarray):
    """Test the predict_mask endpoint of the detector in the FastAPI application.

    This test sends a POST request to the "/detector/predict_mask" endpoint of the
    FastAPI application, including a sample image in the request data.
    It then checks that the response has a 200 status code, and that the response
    data could be decoded to np.array of initial image shape.

    Args:
        client (TestClient): The test client used to send requests to the application.
        sample_image_bytes (bytes): The byte representation of a sample image.
        sample_image_np (np.ndarray): The sample image as a numpy array.

    Raises:
        AssertionError: If the response status code is not 200
    """
    files = {
        "image": sample_image_bytes,
    }
    height, width = sample_image_np.shape[:2]

    response = client.post("/detector/predict_mask", files=files)
    assert response.status_code == HTTPStatus.OK  # noqa: S101

    predicted_mask = response.json()
    # Decode the base64 string back to bytes
    mask_bytes = base64.b64decode(predicted_mask["base64_encoded_mask"])
    np.frombuffer(mask_bytes, dtype=np.float32).reshape((height, width))
