"""Fixtures for tests/base/: local HTTP server serving COG fixtures."""

from __future__ import annotations

import functools
import http.server
import shutil
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that serves a given directory and stays quiet."""

    def log_message(self, *args, **kwargs):  # noqa: D401, N802
        """Suppress the default stderr access log."""
        return


@pytest.fixture(scope="module")
def http_cog_dir(tmp_path_factory) -> Path:
    """Materialize a directory with one valid COG and one plain GTiff."""
    root = tmp_path_factory.mktemp("http_cog_dir")

    # Create a small COG
    src = gdal.GetDriverByName("MEM").Create("", 256, 256, 1, gdal.GDT_Float32)
    src.SetGeoTransform((0.0, 0.001, 0.0, 0.0, 0.0, -0.001))
    import osgeo.osr as osr

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    src.SetProjection(sr.ExportToWkt())
    arr = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    src.GetRasterBand(1).WriteArray(arr)
    src.FlushCache()

    cog_path = root / "valid.tif"
    gdal.GetDriverByName("COG").CreateCopy(
        str(cog_path), src, 0, options=["COMPRESS=DEFLATE"]
    )

    plain_path = root / "plain.tif"
    gdal.GetDriverByName("GTiff").CreateCopy(str(plain_path), src, 0)

    src = None
    return root


@pytest.fixture(scope="module")
def http_server(http_cog_dir: Path):
    """Start a local HTTP server serving `http_cog_dir`; yield the base URL."""
    handler = functools.partial(_QuietHandler, directory=str(http_cog_dir))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()
