"""Dünya küresi için kara noktalarını üretir: web/data/land-dots.bin

Küre yüzeyine eşit aralıklı (Fibonacci) noktalar dağıtılır, karaya denk gelenler
tutulur. Her nokta Int16 enlem*100, boylam*100 olarak yazılır (4 bayt).

Kara verisi: Natural Earth 1:50m (kamu malı), npm "world-atlas" paketi üzerinden.

Kullanım:
    npm install world-atlas@2.0.2 topojson-client@3.1.0
    node -e "const t=require('world-atlas/land-50m.json');require('fs').writeFileSync('land.geojson',JSON.stringify(require('topojson-client').feature(t,t.objects.land)))"
    python scripts/build_land_dots.py land.geojson
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "web" / "data" / "land-dots.bin"
N_POINTS = 90_000  # küre üzerindeki toplam nokta (karada kalan ~%29'u)
W, H = 4096, 2048


def main(geojson: Path) -> None:
    geo = json.loads(geojson.read_text())
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    def ring_xy(ring, shift):
        # Tarih değiştirme çizgisini kesen halkalarda boylamı sürekli hale getir
        lon = np.degrees(np.unwrap(np.radians([pt[0] for pt in ring])))
        lat = [pt[1] for pt in ring]
        return [((lo + shift + 180) / 360 * W, (90 - la) / 180 * H) for lo, la in zip(lon, lat)]

    for feature in geo["features"]:
        geom = feature["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for shift in (-360, 0, 360):
                draw.polygon(ring_xy(poly[0], shift), fill=255)
                for hole in poly[1:]:
                    draw.polygon(ring_xy(hole, shift), fill=0)
    m = np.asarray(mask) > 127

    i = np.arange(N_POINTS) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / N_POINTS))
    lon = np.degrees((np.pi * (1 + 5**0.5) * i) % (2 * np.pi)) - 180
    px = np.clip(((lon + 180) / 360 * W).astype(int), 0, W - 1)
    py = np.clip(((90 - lat) / 180 * H).astype(int), 0, H - 1)
    land = m[py, px] & (lat > -60)  # Antarktika'yı dışarıda bırak, küre daha okunaklı olsun
    pts = np.stack([np.round(lat[land] * 100), np.round(lon[land] * 100)], 1).astype("<i2")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(pts.tobytes())
    print(f"{len(pts):,} kara noktası -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
