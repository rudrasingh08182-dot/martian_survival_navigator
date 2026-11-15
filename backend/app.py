"""Martian Map API.  Run:  uvicorn app:app --reload --port 8002  (from backend/)"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import imagery
import pathfind
import places
import rover
import sciencedata
import spaceweather
import terrain
import swim

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Martian Map")
class Gzip(GZipMiddleware):
    """Gzip for text and tiles, but never the tour video: it's already compressed, and the browser's
    range requests (seeking) must reach the file untouched."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/media/"):
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


app.add_middleware(Gzip, minimum_size=800)  # the JS, CSS, JSON and terrain tiles shrink 3-4x
LONG_CACHE = "public, max-age=2592000, immutable"     # tiles never change: browsers keep them 30 days


@app.get("/api/sites")
def sites():
    return json.loads((ROOT / "data" / "sites.json").read_text(encoding="utf-8"))


@app.get("/api/landmarks")
def landmarks():
    """Mars' major landmarks (OrbitStudio's travel list), each with its MOLA elevation."""
    out = json.loads((ROOT / "data" / "landmarks.json").read_text(encoding="utf-8"))
    for lm in out["landmarks"]:
        lm["elev_m"] = round(terrain.mola_elevation(lm["lat"], lm["lon"]))
    return out


@app.get("/api/imagery/arcadia/status")
def arcadia_imagery_status():
    """Local Arcadia Planitia CTX imagery metadata."""
    try:
        return imagery.arcadia_status()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.warning("Arcadia imagery status failed: %r", e)
        raise HTTPException(500, "Arcadia imagery metadata unavailable")


@app.get("/api/imagery/{layer}/{z}/{y}/{x}")
def imagery_tile(layer: str, z: int, y: int, x: int):
    """NASA and local Mars imagery tiles, encoded as WebP."""

    # Local Arcadia Planitia CTX GeoTIFF.
    if layer == "arcadia":
        if not (
            0 <= z <= imagery.ARCADIA_MAX_Z
            and 0 <= x < 2 ** (z + 1)
            and 0 <= y < 2 ** z
        ):
            raise HTTPException(404, "no such Arcadia tile")

        try:
            data = imagery.arcadia_tile_fast(z, x, y)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            log.warning("Arcadia tile %s/%s/%s failed: %r", z, y, x, e)
            raise HTTPException(502, "Arcadia imagery unavailable")

        return Response(
            data,
            media_type="image/webp",
            headers={"Cache-Control": LONG_CACHE}
        )

    # Existing remote CTX and HiRISE imagery.
    coloured = layer.endswith("_c")
    base = layer[:-2] if coloured else layer

    if base not in imagery.LAYERS or not (
        0 <= z <= imagery.MAX_Z[base]
        and 0 <= x < 2 ** (z + 1)
        and 0 <= y < 2 ** z
    ):
        raise HTTPException(404, "no such tile")

    try:
        data = (imagery.colour if coloured else imagery.tile)(
            base, z, x, y
        )
    except imagery.Missing:
        raise HTTPException(404, "outside this layer's coverage")
    except Exception as e:
        log.warning(
            "imagery %s/%s/%s/%s failed: %r",
            layer, z, y, x, e
        )
        raise HTTPException(502, "imagery source unavailable")

    return Response(
        data,
        media_type="image/webp",
        headers={"Cache-Control": LONG_CACHE}
    )


@app.get("/api/elevation")
def elevation(lat: float, lon: float, quick: bool = False):
    """quick=1 never waits on a download (the cursor readout): cached 200 m data, else MOLA."""
    try:
        z, src = terrain.elevation_src(lat, lon, fetch=not quick)
        return {"lat": lat, "lon": lon, "elev_m": round(z, 1), "source": src}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))



@app.get("/api/location-intelligence")
def location_intelligence(lat: float, lon: float):
    """Combined elevation, approximate slope and SWIM depth evidence at a clicked point."""
    import math

    if not (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
    ):
        raise HTTPException(400, "Invalid Mars coordinates")

    lon = ((lon + 180) % 360) - 180

    # 1. Centre elevation, without triggering a network download.
    elevation_data = {
        "value_m": None,
        "source": "Unavailable"
    }

    try:
        elev, elev_source = terrain.elevation_src(
            lat, lon, fetch=False
        )
        elevation_data = {
            "value_m": round(elev, 2),
            "source": elev_source
        }
    except Exception as e:
        log.warning("Location elevation failed: %r", e)

    # 2. Approximate regional slope using four elevation samples.
    radius = getattr(terrain, "MOLA463_RADIUS", 3396190.0)
    step_m = 463.1

    dlat = math.degrees(step_m / radius)
    cos_lat = max(abs(math.cos(math.radians(lat))), 0.01)
    dlon = math.degrees(step_m / (radius * cos_lat))

    slope_data = {
        "value_deg": None,
        "sample_spacing_m": step_m,
        "method": "Four-point elevation difference"
    }

    try:
        north = terrain.elevation_src(
            min(89.999, lat + dlat), lon, fetch=False
        )[0]

        south = terrain.elevation_src(
            max(-89.999, lat - dlat), lon, fetch=False
        )[0]

        east = terrain.elevation_src(
            lat, ((lon + dlon + 180) % 360) - 180,
            fetch=False
        )[0]

        west = terrain.elevation_src(
            lat, ((lon - dlon + 180) % 360) - 180,
            fetch=False
        )[0]

        dz_dx = (east - west) / (2 * step_m)
        dz_dy = (north - south) / (2 * step_m)

        slope = math.degrees(
            math.atan(math.hypot(dz_dx, dz_dy))
        )

        slope_data["value_deg"] = round(slope, 2)

    except Exception as e:
        log.warning("Location slope calculation failed: %r", e)

    # 3. Actual SWIM consistency values at all three depths.
    swim_data = {}

    for depth in ("0-1m", "1-5m", "over-5m"):
        try:
            swim_data[depth] = swim.sample(
                depth, lat, lon
            )
        except Exception as e:
            log.warning("SWIM sample %s failed: %r", depth, e)
            swim_data[depth] = {
                "layer": depth,
                "consistency": None,
                "meaning": "Data unavailable at this coordinate."
            }

    return {
        "coordinates": {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5)
        },
        "elevation": elevation_data,
        "slope": slope_data,
        "swim": swim_data,
        "important_note": (
            "SWIM values represent consistency with subsurface ice "
            "evidence, not water probability or measured ice volume."
        )
    }


@app.get("/api/places")
def places_search(q: str, limit: int = 8):

    """Search IAU-named Mars features (craters, mountains, valleys...)."""
    return places.search(q, min(limit, 20))


@app.get("/api/whereis")
def whereis(lat: float, lon: float):
    """Describe a position: containing/nearest named feature, plus its elevation."""
    out = places.whereis(lat, lon)
    try:
        z, src = terrain.elevation_src(lat, lon)
        out.update(elev_m=round(z, 1), elev_source=src)
    except FileNotFoundError:
        pass
    return out


@app.get("/api/rover/nearest")
def rover_nearest(lat: float, lon: float, max_km: float = 3.0):
    """Nearest Perseverance/Curiosity stop to a point (within max_km), or null."""
    return rover.nearest(lat, lon, max_km)


@app.get("/api/rover/photos")
def rover_photos(mission: str, sol: int, site: int | None = None, drive: int | None = None, kind: str = "nav",
                 sol_end: int | None = None):
    """Rover photos from a stop, preferring frames taken right there: kind "nav" (Navcam look-around),
    "colour" (Mastcam-Z / Mastcam) or "closeup" (WATSON / MAHLI, over every sol the rover stayed)."""
    if mission not in rover.MISSIONS:
        raise HTTPException(400, "mission must be m2020 or msl")
    if kind not in ("nav", "colour", "closeup"):
        raise HTTPException(400, "kind must be nav, colour or closeup")
    try:
        return rover.photos(mission, sol, site, drive, kind, sol_end)
    except Exception as e:
        raise HTTPException(502, f"NASA raw-image service unavailable: {e}")


@app.get("/api/terrain/tile/{z}/{x}/{y}")
def terrain_tile(z: int, x: int, y: int):
    """3D-globe heightmap: 65x65 int16 metres over geographic tile (z, x, y), row 0 = north."""
    if not (0 <= z <= 22 and 0 <= x < 2 ** (z + 1) and 0 <= y < 2 ** z):
        raise HTTPException(404, "no such tile")
    data, src = terrain.tile_bytes(z, x, y)
    # tiles never change, so browsers can keep them; a MOLA stand-in for a missed download can't
    stand_in = z >= terrain.NET_LEVEL and src == terrain.MOLA_LABEL
    return Response(data, media_type="application/octet-stream",
                    headers={"X-Source": src, "Access-Control-Expose-Headers": "X-Source",
                             "Cache-Control": "public, max-age=120" if stand_in else LONG_CACHE})


@app.get("/api/terrain/area")
def terrain_area(s: float, w: float, n: float, e: float, nx: int = 160, ny: int = 160):
    """Elevation grid for a 3D-printed relief model: float32 metres, row 0 = north."""
    nx, ny = max(8, min(nx, 400)), max(8, min(ny, 400))
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        raise HTTPException(400, "bad area")
    try:
        h, src = terrain.area_grid(s, w, n, e, nx, ny)
    except FileNotFoundError as err:
        raise HTTPException(503, str(err))
    return Response(h.tobytes(), media_type="application/octet-stream",
                    headers={"X-Source": src, "X-Nx": str(nx), "X-Ny": str(ny),
                             "Access-Control-Expose-Headers": "X-Source, X-Nx, X-Ny"})


@app.get("/api/features")
def all_features():
    """Every IAU-named feature, compact: [name, kind, lat, lon, diameter_km, MOLA elevation m] — globe labels."""
    return [[f["name"], f["kind"], f["lat"], f["lon"], f["diameter_km"], round(terrain.mola_elevation(f["lat"], f["lon"]))]
            for f in places.features()]


@app.get("/api/rover/track")
def rover_track(mission: str):
    """A rover's whole traverse as [[lat, lon], ...] plus its latest stop."""
    if mission not in rover.MISSIONS:
        raise HTTPException(400, "mission must be m2020 or msl")
    wp = rover.waypoints(mission)
    return {"rover": rover.MISSIONS[mission]["name"], "points": [[p["lat"], p["lon"]] for p in wp], "latest": wp[-1]}


@app.get("/api/dem")
def dem_coverage():
    """Which elevation models are loaded and where they apply."""
    return terrain.dem_coverage()


# --- SWIM 2.0: subsurface water-ice consistency maps --------------------------------------------
@app.get("/api/swim/status")
def swim_status():
    """Availability and georeferencing metadata for the three SWIM 2.0 depth products."""
    return swim.status()


@app.get("/api/swim/sample")
def swim_sample(depth: str, lat: float, lon: float):
    """Sample an ice-consistency score at a Mars coordinate."""
    try:
        return swim.sample(depth, lat, lon)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/swim/{depth}/{z}/{x}/{y}.png")
def swim_tile(depth: str, z: int, x: int, y: int):
    """A transparent PNG overlay tile for a SWIM 2.0 ice-consistency depth layer."""
    try:
        content = swim.tile_png(depth, z, x, y)
        return Response(content, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))




class Route(BaseModel):
    waypoints: list[tuple[float, float]] = Field(..., min_length=2, max_length=200)
    step_m: float = Field(50, ge=20, le=5000)
    avoid: bool = False  # walk around ground steeper than 15° where a reasonable detour exists


@app.post("/api/profile")
def profile(route: Route):
    try:
        detours = None
        if route.avoid:
            legs = list(zip(route.waypoints, route.waypoints[1:]))
            with ThreadPoolExecutor(4) as ex:
                detours = list(ex.map(lambda ab: pathfind.safe_leg(*ab), legs))
        return terrain.profile(route.waypoints, route.step_m, detours)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


# --- science layers: minerals, salts, ice, active slopes, geology -------------------------------
@app.get("/api/science/features")
def science_features():
    """Every mapped detection, compact, for the map layers (changes only when the data is rebuilt)."""
    return Response(json.dumps(sciencedata.features(), separators=(",", ":")), media_type="application/json",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/science/jezero-geology")
def science_jezero_geology():
    """The USGS 1:75,000 geologic map of Jezero and Nili Planum (SIM 3464): unit polygons and descriptions."""
    return Response(json.dumps(sciencedata.jezero_geology(), separators=(",", ":"), ensure_ascii=False),
                    media_type="application/json", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/science/near")
def science_near(lat: float, lon: float, km: float = 25):
    """What orbiters found near a point, plus the geologic unit under it."""
    try:
        geo = sciencedata.geology(lat, lon)
    except Exception:
        geo = None  # the geologic map is a live Mars Trek query; the rest is local
    return {"features": sciencedata.near(lat, lon, min(km, 200)), "geology": geo}


class Points(BaseModel):
    points: list[tuple[float, float]] = Field(..., min_length=1, max_length=5000)
    km: float = Field(3, gt=0, le=50)


@app.post("/api/science/along")
def science_along(body: Points):
    """Science targets close to a route, and the geologic units it crosses, in walking order."""
    return {"features": sciencedata.along(body.points, body.km), "geology": sciencedata.geology_along(body.points)}


@app.get("/api/spaceweather")
def space_weather():
    """Solar flares, particle events and Mars-bound CMEs (NASA DONKI) with NOAA SWPC's GOES data."""
    return spaceweather.status()


@app.get("/", include_in_schema=False)
def index(request: Request):
    """The app page, with its link-preview card pointing at whatever address it is served from
    (social sites need absolute URLs, and the host can be a cloud URL or a custom domain)."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0]
    origin = f"{proto}://{request.headers.get('host', request.url.netloc)}"
    page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(page.replace("https://martian-map.origin", origin), headers={"Cache-Control": "no-cache"})


class Frontend(StaticFiles):
    """Static files that browsers re-validate on every load (cheap 304s via ETag), so a deploy
    never leaves visitors on a stale app.js."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", Frontend(directory=ROOT / "frontend", html=True), name="frontend")
