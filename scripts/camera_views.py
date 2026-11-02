"""Küreden bir mirasa uçarken kameranın dağın içine girmemesi için her mirasa bakış açısı seçer.

MapLibre yakınlaştırma (zoom), eğim (pitch) ve yön (bearing) ile kamerayı merkezden belli
bir uzaklığa koyar. Machu Picchu gibi dağlık yerlerde yüksek eğimle kamera arkadaki dağın
içinde kalıp ekran kararıyor. Bu betik AWS Terrain Tiles (terrarium) yükselti verisiyle her
miras için adayları sırayla dener; kameranın kendisi ve kameradan mirasa giden görüş çizgisi
arazinin üstünde kalan ilk (en yakın, en eğik) bakışı seçer ve web/data/views.json'a yazar:

    {"274": [-13.163, -72.545, 15.3, 62, -18, 2430], ...}   # enlem, boylam, zoom, pitch, bearing, merkezin yüksekliği (m)

Merkez yüksekliği uçuşta kameraya doğrudan verilir: arazi döşemeleri henüz yüklenmemişken
MapLibre merkezi deniz seviyesinde sanıp kamerayı dağın içine indirmesin.

Odak noktası focus.json'dan (mirasın en bilinen yapısı), yoksa UNESCO koordinatından alınır.

    python scripts/camera_views.py
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "data"
DEM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DZ = 11                      # ~76 m/piksel (ekvatorda): dağ silueti için yeterli
H_PX = 1000                  # tasarım ekran yüksekliği (masaüstü)
CAM_PX = 1.5 * H_PX          # MapLibre: kamera-merkez uzaklığı = 0,5 / tan(fov/2) * yükseklik, fov ≈ 36,87°
R_EARTH = 6378137.0
MARGIN = 70.0                # kamera ve görüş çizgisi arazinin en az bu kadar üstünde

CULTURAL = [(15.3, 62), (15.0, 58), (14.6, 54), (14.2, 50), (13.8, 45), (13.3, 40), (12.8, 35), (12.2, 30), (11.6, 25)]
NATURAL = [(10.5, 55), (10.2, 50), (9.8, 45), (9.3, 40)]
BEARINGS = [-18, 70, 160, 250]


@lru_cache(maxsize=4096)
def tile(x: int, y: int):
    n = 2 ** DZ
    x %= n
    if not 0 <= y < n:
        return None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(DEM.format(z=DZ, x=x, y=y), timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:  # noqa: BLE001 - ağ hatası, tekrar dene
            time.sleep(1.5 * (attempt + 1))
    return None


def elevation(lat: float, lon: float) -> float:
    n = 2 ** DZ
    xt = (lon + 180) / 360 * n
    lat = max(-85.0, min(85.0, lat))
    yt = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    im = tile(int(xt), int(yt))
    if im is None:
        return 0.0
    r, g, b = im.getpixel((min(255, int((xt % 1) * 256)), min(255, int((yt % 1) * 256))))
    return max(-50.0, r * 256 + g + b / 256 - 32768)


def offset(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    dlat = dist_m * math.cos(b) / 111320
    dlon = dist_m * math.sin(b) / (111320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def clear(lat: float, lon: float, h0: float, zoom: float, pitch: float, bearing: float) -> bool:
    mpp = 2 * math.pi * R_EARTH * math.cos(math.radians(lat)) / (512 * 2 ** zoom)
    dist = CAM_PX * mpp
    horiz, up = dist * math.sin(math.radians(pitch)), dist * math.cos(math.radians(pitch))
    cam_alt = h0 + up
    # kamera, mirasa bakış yönünün tersinde durur
    back = (bearing + 180) % 360
    for k in range(1, 11):
        f = k / 10
        la, lo = offset(lat, lon, back, horiz * f)
        line = h0 + up * f
        if elevation(la, lo) > line - MARGIN * f:
            return False
    # kameranın çevresi (geniş açı: yanlardaki yamaçlar)
    cla, clo = offset(lat, lon, back, horiz)
    for side in (-90, 90):
        la, lo = offset(cla, clo, back + side, 250)
        if elevation(la, lo) > cam_alt - MARGIN:
            return False
    return True


def view_for(site: dict, focus: list | None):
    lat, lon = (focus[0], focus[1]) if focus else (site["lat"], site["lon"])
    h0 = elevation(lat, lon)
    cands = NATURAL if site.get("cat") == "doğal" else CULTURAL
    for zoom, pitch in cands:
        for bearing in BEARINGS:
            if clear(lat, lon, h0, zoom, pitch, bearing):
                return site["id"], [round(lat, 6), round(lon, 6), zoom, pitch, bearing, round(h0)]
    zoom, pitch = cands[-1]
    return site["id"], [round(lat, 6), round(lon, 6), zoom - 1, 0, 0, round(h0)]


def main() -> None:
    sites = json.loads((DATA / "sites.json").read_text())["sites"]
    focus = json.loads((DATA / "focus.json").read_text()) if (DATA / "focus.json").exists() else {}
    out, moved = {}, 0
    with ThreadPoolExecutor(8) as ex:
        for i, (sid, v) in enumerate(ex.map(lambda s: view_for(s, focus.get(str(s["id"]))), sites), 1):
            out[str(sid)] = v
            if (v[2], v[3], v[4]) != (CULTURAL[0][0], CULTURAL[0][1], BEARINGS[0]) and (v[2], v[3], v[4]) != (NATURAL[0][0], NATURAL[0][1], BEARINGS[0]):
                moved += 1
            if i % 100 == 0:
                print(f"[views] {i}/{len(sites)}", flush=True)
    (DATA / "views.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"[views] bitti: {len(out)} miras, {moved} tanesinde dağ yüzünden bakış değişti -> {DATA / 'views.json'}")


if __name__ == "__main__":
    main()
