"""Health endpoints related tests."""
from http import HTTPStatus

from fastapi.testclient import TestClient


def test_health_checker(client: TestClient):
    """Test the health_checker endpoint of the health route in the FastAPI application.

    This test sends a GET request to the "/health/health_checker" endpoint of the
    FastAPI application. It then checks that the response has a 200 status code.

    Args:
        client (TestClient): The test client used to send requests to the application.

    Raises:
        AssertionError: If the response status code is not 200.
    """
    response = client.get("/health/health_checker")
    assert response.status_code == HTTPStatus.OK  # noqa: S101
