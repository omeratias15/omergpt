"""
FastAPI Server for omerGPT
Launch with: python src/api/server.py
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.endpoints import router
from src.api.middleware import RateLimitMiddleware, LoggingMiddleware, CORSMiddleware
from src.api.auth import generate_api_token

# -----------------------------------------------------------
# Dynamic cross-platform log path fix
# -----------------------------------------------------------
# Create a "logs" folder that works both locally (Windows) and in Docker (Linux)
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="omerGPT API",
    description="Autonomous Crypto Market Intelligence Platform API",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(CORSMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# Include routers
app.include_router(router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 60)
    logger.info("omerGPT API Server Starting")
    logger.info("=" * 60)

    # Generate a sample token for first-time users
    token = generate_api_token("omergpt_user")
    logger.info(f"Sample API Token: {token}")
    logger.info("Include this token in Authorization header: Bearer <token>")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("omerGPT API Server Shutting Down")

def main():
    """Run the API server"""
    logger.info("Starting omerGPT API Server on http://0.0.0.0:8000")
    logger.info("API Documentation: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
