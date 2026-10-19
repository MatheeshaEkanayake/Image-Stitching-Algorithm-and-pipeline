"""HTTP server: upload photos, stitch in the background, serve a zoomable tile pyramid."""

from stitcher.server.app import ServerSettings, create_app

__all__ = ["ServerSettings", "create_app"]
