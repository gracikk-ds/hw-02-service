"""Health realted endpoints."""

from fastapi import Response

from src.routes.routers import health_router

AWESOME_RESPONSE: int = 200


@health_router.get("/health_checker")  # type: ignore
async def health_checker():
    """Endpoint is used to check if this service responding.

    Returns:
        Response: An HTTP response indicating the health status.
    """
    return Response(status_code=AWESOME_RESPONSE)
