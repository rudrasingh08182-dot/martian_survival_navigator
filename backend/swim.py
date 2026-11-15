
"""NASA/PSI SWIM 2.0 global subsurface-water-ice consistency layers.

The GeoTIFFs are equirectangular Mars maps (ESRI:103885; Mars sphere radius
3,396,190 m), covering approximately 60 N to 60 S at ~2.964 km/pixel.

Values are ice-consistency scores, not probabilities or ice-volume estimates:
positive values are consistent with ice, negative values are inconsistent,
zero is inconclusive/no indication, and -100 is NoData.

Files expected in data/swim/:
  SWIM2_c0_1.tif  combined consistency, 0-1 m
  SWIM2_c1_5.tif  combined consistency, 1-5 m
  SWIM2_c_5.tif   combined consistency, >5 m
"""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import math
import threading

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "swim"

MARS_RADIUS_M = 3_396_190.0
TILE_SIZE = 256
MAX_NATIVE_ZOOM = 5
NODATA = -100.0

LAYERS = {
    "0-1m": {
        "file": "SWIM2_c0_1.tif",
        "title": "Combined ice consistency (0–1 m)",
        "depth": "0–1 m",
        "description": (
            "Combined neutron, thermal, shallow geomorphology "
            "and radar surface-power evidence."
        ),
        "source": "NASA/PSI SWIM 2.0",
    },
    "1-5m": {
        "file": "SWIM2_c1_5.tif",
        "title": "Combined ice consistency (1–5 m)",
        "depth": "1–5 m",
        "description": (
            "Combined shallow geomorphology, radar surface-power "
            "and radar dielectric evidence."
        ),
        "source": "NASA/PSI SWIM 2.0",
    },
    "over-5m": {
        "file": "SWIM2_c_5.tif",
        "title": "Combined ice consistency (>5 m)",
        "depth": ">5 m",
        "description": (
            "Combined deep geomorphology and radar dielectric evidence."
        ),
        "source": "NASA/PSI SWIM 2.0",
    },
}

_lock = threading.RLock()

_readers: dict[str, rasterio.io.DatasetReader] = {}

_tile_cache: OrderedDict[
    tuple[str, int, int, int], bytes
] = OrderedDict()

_MAX_CACHED_TILES = 192


def _path(layer: str) -> Path:
    """Return the GeoTIFF path for a supported depth layer."""

    if layer not in LAYERS:
        raise KeyError(f"Unknown SWIM depth layer: {layer}")

    return DATA / LAYERS[layer]["file"]


def _reader(layer: str):
    """Open a SWIM raster and reuse the dataset reader."""

    path = _path(layer)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Extract SWIMProducts.zip into data/swim/."
        )

    ds = _readers.get(layer)

    if ds is None or ds.closed:
        ds = rasterio.open(path)
        _readers[layer] = ds

    return ds


def status() -> dict:
    """Summarize local availability and metadata for all SWIM GeoTIFFs."""

    result = {}

    for key, meta in LAYERS.items():
        path = DATA / meta["file"]

        item = {
            "file": meta["file"],
            "title": meta["title"],
            "depth": meta["depth"],
            "description": meta["description"],
            "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "width": None,
            "height": None,
            "resolution_m": None,
            "crs": None,
        }

        if path.is_file():
            try:
                with rasterio.open(path) as ds:
                    item.update({
                        "width": ds.width,
                        "height": ds.height,
                        "resolution_m": round(abs(ds.transform.a), 2),
                        "crs": ds.crs.to_string() if ds.crs else None,
                    })

            except Exception as exc:
                item["metadata_error"] = str(exc)
                item["available"] = False

        result[key] = item

    return {
        "ready": all(v["available"] for v in result.values()),
        "coverage_latitude": [-60, 60],
        "score_range": [-1, 1],
        "score_note": (
            "Consistency with subsurface ice evidence, not probability, "
            "water percentage or measured ice volume."
        ),
        "layers": result,
    }


def _rgba(values: np.ndarray) -> np.ndarray:
    """Render consistency values as a transparent diverging colour overlay."""

    a = np.asarray(values, dtype=np.float32)

    valid = (
        np.isfinite(a)
        & (a != NODATA)
        & (a > -10)
    )

    v = np.clip(a, -1.0, 1.0)

    neutral = np.array([124, 132, 139], dtype=np.float32)
    negative = np.array([226, 105, 58], dtype=np.float32)
    positive = np.array([37, 177, 222], dtype=np.float32)

    rgb = np.empty((*a.shape, 3), dtype=np.uint8)

    neg_t = np.clip(-v, 0, 1)[..., None]
    pos_t = np.clip(v, 0, 1)[..., None]

    neg_rgb = neutral + (negative - neutral) * neg_t
    pos_rgb = neutral + (positive - neutral) * pos_t

    rgb[:] = np.where(
        (v >= 0)[..., None],
        pos_rgb,
        neg_rgb,
    ).astype(np.uint8)

    alpha = (
        88 + 125 * np.abs(v)
    ).clip(0, 220).astype(np.uint8)

    rgba = np.dstack((rgb, alpha))

    # NoData pixels remain completely transparent.
    rgba[~valid, 3] = 0

    return rgba


def tile_png(layer: str, z: int, x: int, y: int) -> bytes:
    """Render one 256px Leaflet/Cesium EPSG:4326 tile from a SWIM raster."""

    if layer not in LAYERS:
        raise KeyError(f"Unknown SWIM depth layer: {layer}")

    # Geographic EPSG:4326 tiling:
    # X has twice as many tiles as Y at the same zoom level.
    #
    # z=1: X = 0..3, Y = 0..1
    # z=2: X = 0..7, Y = 0..3
    # z=3: X = 0..15, Y = 0..7
    if (
        z < 1
        or z > MAX_NATIVE_ZOOM
        or x < 0
        or x >= 2 ** (z + 1)
        or y < 0
        or y >= 2 ** z
    ):
        raise ValueError(
            f"Invalid SWIM tile coordinates: z={z}, x={x}, y={y}"
        )

    key = (layer, z, x, y)

    with _lock:

        # Return an already-rendered tile where available.
        cached = _tile_cache.get(key)

        if cached is not None:
            _tile_cache.move_to_end(key)
            return cached

        ds = _reader(layer)

        # EPSG:4326 has different longitudinal and latitudinal spans.
        lon_span = 360.0 / (2 ** (z + 1))
        lat_span = 180.0 / (2 ** z)

        west = -180.0 + x * lon_span
        east = west + lon_span

        north = 90.0 - y * lat_span
        south = north - lat_span

        # Convert geographic coordinates to Mars equirectangular metres.
        left = MARS_RADIUS_M * math.radians(west)
        right = MARS_RADIUS_M * math.radians(east)

        bottom = MARS_RADIUS_M * math.radians(south)
        top = MARS_RADIUS_M * math.radians(north)

        window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=ds.transform,
        )

        # Read and resample the required geographic window.
        # Areas outside the SWIM raster coverage remain transparent.
        values = ds.read(
            1,
            window=window,
            out_shape=(TILE_SIZE, TILE_SIZE),
            boundless=True,
            fill_value=NODATA,
            resampling=Resampling.bilinear,
        )

        rgba = _rgba(values)

        image = Image.fromarray(rgba, mode="RGBA")

        buf = BytesIO()
        image.save(buf, format="PNG", optimize=True)

        data = buf.getvalue()

        # Cache the rendered PNG.
        _tile_cache[key] = data
        _tile_cache.move_to_end(key)

        while len(_tile_cache) > _MAX_CACHED_TILES:
            _tile_cache.popitem(last=False)

        return data


def sample(layer: str, lat: float, lon: float) -> dict:
    """Sample a SWIM consistency value at a Mars latitude/east longitude."""

    if layer not in LAYERS:
        raise KeyError(f"Unknown SWIM depth layer: {layer}")

    if (
        not math.isfinite(lat)
        or not math.isfinite(lon)
        or not -90 <= lat <= 90
    ):
        raise ValueError(
            "Latitude/longitude are outside valid numeric ranges."
        )

    # Normalize longitude into [-180, 180).
    lon = ((lon + 180) % 360) - 180

    with _lock:

        ds = _reader(layer)

        # Convert Mars geographic coordinates into raster metres.
        x = MARS_RADIUS_M * math.radians(lon)
        y = MARS_RADIUS_M * math.radians(lat)

        row, col = ds.index(x, y)

        if not (0 <= row < ds.height and 0 <= col < ds.width):
            value = None

        else:
            raw = float(
                ds.read(
                    1,
                    window=(
                        (row, row + 1),
                        (col, col + 1),
                    ),
                )[0, 0]
            )

            if (
                not math.isfinite(raw)
                or raw == NODATA
                or raw < -10
            ):
                value = None
            else:
                value = round(raw, 4)

    if value is None:
        meaning = "No SWIM coverage at this coordinate."

    elif value > 0:
        meaning = (
            "Positive consistency: the mapped evidence is "
            "consistent with subsurface ice."
        )

    elif value < 0:
        meaning = (
            "Negative consistency: the mapped evidence is "
            "inconsistent with subsurface ice."
        )

    else:
        meaning = (
            "Neutral: the mapped evidence is inconclusive "
            "or provides no indication either way."
        )

    meta = LAYERS[layer]

    return {
        "layer": layer,
        "title": meta["title"],
        "depth": meta["depth"],
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "consistency": value,
        "meaning": meaning,
        "source": meta["source"],
        "resolution_m": 2964.12,
        "caveat": (
            "This is an orbital evidence-consistency score, not a "
            "probability, ice concentration, or measured water reserve."
        ),
    }
