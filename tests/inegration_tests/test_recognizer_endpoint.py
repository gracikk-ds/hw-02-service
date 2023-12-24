"""This module contains tests for predicting barcodes info using a FastAPI application.

The tests use FastAPI's TestClient to send HTTP requests to the application,
and then check the responses to ensure that they are correct.
"""
from http import HTTPStatus

from fastapi.testclient import TestClient


def test_recognize_barcode(client: TestClient, sample_image_bytes: bytes):
    """Test the recognize_barcode endpoint of the recognizer in the FastAPI application.

    This test sends a POST request to the "/recognizer/recognize_barcode" endpoint of the
    FastAPI application, including a sample image in the request data.

    Args:
        client (TestClient): The test client used to send requests to the application.
        sample_image_bytes (bytes): The byte representation of a sample image.

    Raises:
        AssertionError: If the response status code is not 200, or if the response is not a string.
    """
    files = {
        "image": sample_image_bytes,
    }

    response = client.post("/recognizer/recognize_barcode", files=files)
    assert response.status_code == HTTPStatus.OK  # noqa: S101

    predicted_bboxes = response.json()
    assert isinstance(predicted_bboxes, str)  # noqa: S101


def test_recognize_image(client: TestClient, sample_image_bytes: bytes):
    """Test the recognize_image endpoint of the recognizer in the FastAPI application.

    This test sends a POST request to the "/recognizer/recognize_image" endpoint of the
    FastAPI application, including a sample image in the request data.

    Args:
        client (TestClient): The test client used to send requests to the application.
        sample_image_bytes (bytes): The byte representation of a sample image.

    Raises:
        AssertionError: If the response status code is not 200, or if the response
            data does not include mandatory keys.
    """
    files = {
        "image": sample_image_bytes,
    }

    response = client.post("/recognizer/recognize_image", files=files)
    assert response.status_code == HTTPStatus.OK  # noqa: S101

    predicted_image_info = response.json()
    assert "barcodes" in predicted_image_info  # noqa: S101
    assert "bbox" in predicted_image_info["barcodes"][0]  # noqa: S101
    assert "value" in predicted_image_info["barcodes"][0]  # noqa: S101
