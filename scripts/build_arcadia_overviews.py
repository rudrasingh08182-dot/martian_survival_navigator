
from pathlib import Path

import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]

IMAGE = (
    ROOT
    / "data"
    / "external"
    / "Arcadia_Planitia_CTX_BlockAdj_dd.tif"
)

FACTORS = [8, 16, 32, 64, 128]


def main():
    if not IMAGE.exists():
        raise FileNotFoundError(IMAGE)

    print("Opening Arcadia CTX...")
    print("File:", IMAGE)

    with rasterio.open(IMAGE, "r+") as ds:

        existing = ds.overviews(1)

        print("Existing overviews:", existing)

        missing = [
            factor for factor in FACTORS
            if factor not in existing
        ]

        if not missing:
            print("All requested overviews already exist.")
            return

        print("Building overview levels:", missing)
        print("Resampling: average")
        print("This is a one-time operation.")

        ds.build_overviews(
            missing,
            Resampling.average
        )

        ds.update_tags(
            ns="rio_overview",
            resampling="average"
        )

        print("New overviews:", ds.overviews(1))

    print("\nArcadia CTX optimization completed!")


if __name__ == "__main__":
    main()
