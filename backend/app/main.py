"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router as api_router

app = FastAPI(title="johnmiller-stackai", version="0.1.0")

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["api"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from johnmiller-stackai backend"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

