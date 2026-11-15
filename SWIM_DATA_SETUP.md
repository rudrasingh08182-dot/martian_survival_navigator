# SWIM 2.0 subsurface water-ice layers

This project includes three NASA/Planetary Science Institute SWIM 2.0 GeoTIFF products in
`data/swim/`:

| File | Layer | Intended sensing-depth band |
|---|---|---|
| `SWIM2_c0_1.tif` | Combined ice consistency | 0–1 m |
| `SWIM2_c1_5.tif` | Combined ice consistency | 1–5 m |
| `SWIM2_c_5.tif` | Combined ice consistency | >5 m |

## Run

1. Keep the three `.tif` files in `data/swim/` (the supplied modified project archive already includes them).
2. Install dependencies from the repository root:

   ```powershell
   .venv\\Scripts\\python.exe -m pip install -r requirements.txt
   ```

3. Start the project with `run.bat`.
4. Open `http://127.0.0.1:8002/api/swim/status` to check all three files.
5. In the map, open **Layers → Science data** and enable one of the **Subsurface ice consistency** overlays.

## API

- `GET /api/swim/status` — file availability, raster dimensions, CRS and resolution.
- `GET /api/swim/sample?depth=0-1m&lat=40&lon=-120` — samples one depth-band score.
- `GET /api/swim/0-1m/{z}/{x}/{y}.png` — transparent Leaflet raster tile.
- Other depth keys: `1-5m` and `over-5m`.

## Interpretation and limitations

The three GeoTIFFs are global composite **ice-consistency** maps, not ice-probability maps.
SWIM scores represent how consistent the combined orbital evidence is with subsurface ice:
positive is consistent, negative is inconsistent, and zero is inconclusive or provides no
indication. A score is not a percentage of water, an ice-volume estimate, or a confirmation
that a drill would find extractable ice.

These products are approximately 2.964 km/pixel and cover approximately 60°N to 60°S.
They should be used for regional screening and mission planning context, not for selecting a
precise drill point without higher-resolution ground-truthing.

Official product descriptions: https://swim.psi.edu/SWIM2Products.php
