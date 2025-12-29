"""API route handlers."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/example")
async def example_endpoint():
    """Example API endpoint."""
    return {"message": "Example endpoint"}

