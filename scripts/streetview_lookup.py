"""Her UNESCO mirası için en yakın Google Street View panosunu bulur.

Street View Static API'nin "metadata" isteği ücretsizdir ve kotası sınırsızdır
(Google Maps Platform fiyat listesi: "Street View Metadata", Unlimited). Betik her
miras için önce 300 m, bulamazsa 1.500 m içinde dış mekân panosu arar ve
web/data/streetview.json'a yazar:

    {"356": [41.00857, 28.97965, "pano_id", 132]}   # enlem, boylam, pano, mirasa bakış yönü

Gezi sayfası (miras.html) gömülü Street View'ı doğrudan bu panoyla ve mirasa
dönük açar. Anahtar, yalnızca bu betik çalışırken sunucu tarafında kullanılır:

    GOOGLE_MAPS_KEY=... python scripts/streetview_lookup.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "web" / "data" / "sites.json"
OUT = ROOT / "web" / "data" / "streetview.json"
API = "https://maps.googleapis.com/maps/api/streetview/metadata"


def bearing(lat1, lon1, lat2, lon2) -> int:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360)


def lookup(key: str, site: dict):
    for radius in (300, 1500):
        q = urllib.parse.urlencode({"location": f"{site['lat']},{site['lon']}", "radius": radius, "source": "outdoor", "key": key})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
                    j = json.load(r)
                break
            except Exception:  # noqa: BLE001 - ağ hatası, tekrar dene
                time.sleep(2 * (attempt + 1))
        else:
            return site["id"], None
        if j.get("status") == "OK":
            lat, lng = j["location"]["lat"], j["location"]["lng"]
            far = math.hypot((lat - site["lat"]) * 111320, (lng - site["lon"]) * 111320 * math.cos(math.radians(lat)))
            head = bearing(lat, lng, site["lat"], site["lon"]) if far > 15 else 0
            return site["id"], [round(lat, 6), round(lng, 6), j["pano_id"], head]
        if j.get("status") not in ("ZERO_RESULTS", "NOT_FOUND"):
            raise SystemExit(f"Street View metadata hatası: {j.get('status')} {j.get('error_message', '')}")
    return site["id"], None


def main() -> None:
    key = os.environ.get("GOOGLE_MAPS_KEY")
    if not key:
        raise SystemExit("GOOGLE_MAPS_KEY ortam değişkeni gerekli")
    sites = json.loads(SITES.read_text())["sites"]
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for i, (sid, v) in enumerate(ex.map(lambda s: lookup(key, s), sites), 1):
            if v:
                out[str(sid)] = v
            if i % 100 == 0:
                print(f"[streetview] {i}/{len(sites)} miras, {len(out)} panoda", flush=True)
    OUT.write_text(json.dumps(dict(sorted(out.items(), key=lambda kv: int(kv[0]))), separators=(",", ":")))
    print(f"[streetview] bitti: {len(out)}/{len(sites)} mirasın yakınında Street View var -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
