"""FastAPI inference and benchmark reporting service for edge-vision-bench."""

from edge_vision_bench.api.app import AppState, create_app
from edge_vision_bench.api.routes import router

__all__ = ["AppState", "create_app", "router"]
