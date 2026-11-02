"""Street View bakış noktalarını en yakın yola ya da patikaya kaydırır.

Bir yapının tam koordinatında çoğu zaman ziyaretçilerin yüklediği tek bir 360° fotoğraf
(foto küre) durur; Street View oraya açılınca ilerleme okları çıkmaz. Google'ın kendi
çekimleri (araç ya da sırt çantalı Trekker) ise yollar ve patikalar boyuncadır ve okludur.
Bu betik her bakış noktasını OpenStreetMap'teki en yakın yol/patikaya (en fazla 120 m)
kaydırır ve oradan yapıya bakış yönünü hesaplar. Yol ve patika verisi OpenFreeMap'in
vektör döşemelerinden (OpenMapTiles şeması, "transportation" katmanı) okunur:

    web/data/viewpoints.json   [enlem, boylam, ad, vikipedi]  ->  [..., [yol_enlem, yol_boylam, yön]]

Çalıştırma (anahtar gerekmez):

    python scripts/snap_viewpoints.py && python scripts/build_site_data.py
"""

from __future__ import annotations

import gzip
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mapbox_vector_tile

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "web" / "data" / "viewpoints.json"
TILEJSON = "https://tiles.openfreemap.org/planet"
Z = 14
MAX_M = 120.0
SKIP = {"rail", "transit", "ferry", "aerialway", "motorway", "busway"}


def get(url: str) -> bytes | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MirasGezgini/1.0 (github.com/Oguzhan61-dilmac/360-vid2splat)", "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001 - ağ hatası, tekrar dene
            time.sleep(2 * (attempt + 1))
    return None


def tile_xy(lat: float, lon: float) -> tuple[int, int]:
    n = 2 ** Z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def to_lonlat(x: int, y: int, px: float, py: float, extent: int) -> tuple[float, float]:
    n = 2 ** Z
    lon = (x + px / extent) / n * 360 - 180
    m = math.pi * (1 - 2 * (y + 1 - py / extent) / n)   # MVT y ekseni aşağıdan yukarı
    return lon, math.degrees(math.atan(math.sinh(m)))


_cache: dict[tuple[int, int], list] = {}


def lines(url_tmpl: str, x: int, y: int) -> list[list[tuple[float, float]]]:
    if (x, y) in _cache:
        return _cache[(x, y)]
    raw = get(url_tmpl.format(z=Z, x=x, y=y))
    out = []
    if raw:
        layer = mapbox_vector_tile.decode(raw).get("transportation")
        if layer:
            ext = layer.get("extent", 4096)
            for f in layer["features"]:
                if f["properties"].get("class") in SKIP:
                    continue
                g = f["geometry"]
                parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"] if g["type"] == "MultiLineString" else []
                for part in parts:
                    out.append([to_lonlat(x, y, px, py, ext) for px, py in part])
    _cache[(x, y)] = out
    return out


def meters(lat0: float, a: tuple[float, float]) -> tuple[float, float]:
    return (a[0]) * 111320 * math.cos(math.radians(lat0)), a[1] * 111320


def nearest_on_lines(lat: float, lon: float, segs: list) -> tuple[float, float, float] | None:
    best = None
    for line in segs:
        for (lo1, la1), (lo2, la2) in zip(line, line[1:]):
            ax, ay = meters(lat, (lo1 - lon, la1 - lat))
            bx, by = meters(lat, (lo2 - lon, la2 - lat))
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            t = 0.0 if L == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(px, py)
            if best is None or d < best[2]:
                best = (lat + py / 111320, lon + px / (111320 * math.cos(math.radians(lat))), d)
    return best


def bearing(lat1, lon1, lat2, lon2) -> int:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360)


def snap(url_tmpl: str, pt: list) -> list:
    lat, lon = pt[0], pt[1]
    x, y = tile_xy(lat, lon)
    segs = []
    for dx in (-1, 0, 1):          # nokta döşeme kenarına yakınsa komşu döşemeler de
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0 or _near_edge(lat, lon, x, y, dx, dy):
                segs += lines(url_tmpl, x + dx, y + dy)
    hit = nearest_on_lines(lat, lon, segs)
    base = pt[:4] if len(pt) >= 4 else pt + [""] * (4 - len(pt))
    if hit and 6 < hit[2] <= MAX_M:
        return base + [[round(hit[0], 6), round(hit[1], 6), bearing(hit[0], hit[1], lat, lon)]]
    return base


def _near_edge(lat, lon, x, y, dx, dy) -> bool:
    n = 2 ** Z
    fx = (lon + 180) / 360 * n - x
    fy = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n - y
    tile_m = 40075016 * math.cos(math.radians(lat)) / n
    edge = MAX_M / tile_m
    return (dx == -1 and fx < edge or dx == 1 and fx > 1 - edge or dx == 0) and (dy == -1 and fy < edge or dy == 1 and fy > 1 - edge or dy == 0)


def main() -> None:
    url_tmpl = json.loads(get(TILEJSON))["tiles"][0]
    views = json.loads(VIEWS.read_text())
    jobs = [(sid, i, pt) for sid, pts in views.items() for i, pt in enumerate(pts)]
    snapped = 0
    with ThreadPoolExecutor(8) as ex:
        for k, ((sid, i, _), res) in enumerate(zip(jobs, ex.map(lambda j: snap(url_tmpl, j[2]), jobs)), 1):
            views[sid][i] = res
            snapped += len(res) > 4
            if k % 500 == 0:
                print(f"[snap] {k}/{len(jobs)} nokta, {snapped} tanesi yola kaydırıldı", flush=True)
    VIEWS.write_text(json.dumps(views, ensure_ascii=False, separators=(",", ":")))
    print(f"[snap] bitti: {snapped}/{len(jobs)} nokta en yakın yola/patikaya kaydırıldı -> {VIEWS}")


if __name__ == "__main__":
    main()
