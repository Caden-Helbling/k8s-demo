# app/main.py
from fastapi import FastAPI
import os

# Create the FastAPI app with a prefix
app = FastAPI(
    title="API Example",
    description="An example API with versioned routes",
    version="1.0.0",
)

# Define a router with the prefix
api_router = FastAPI(prefix="/v1/api")

@api_router.get("/hello")
async def root():
    return {"message": "Hello World"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

@api_router.get("/info")
async def info():
    hostname = os.uname().nodename
    return {
        "hostname": hostname,
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Mount the router to the main app
app.mount("/v1/api", api_router)
