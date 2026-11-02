"""Her miras için gezi sayfalarının tek istekte yükleyeceği küçük bir JSON üretir.

miras.html, google3d.html ve arazi.html bir mirası açarken 360 KB'lık sites.json'u
ve 1 MB'lık sketchfab.json'u indirmek yerine yalnızca web/data/miras/<no>.json'u
(1–3 KB) okur:

    {"site": {...}, "scans": [...], "models": [...], "sv": [enlem, boylam, pano, yön] | null, "svChecked": bool,
     "pts": [[enlem, boylam, ad], ...]}   # Street View bakış noktaları (scripts/viewpoints.py)

sites.json, scans.json, sketchfab.json ya da streetview.json değişince yeniden çalıştır:

    python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "data"
OUT = DATA / "miras"
KEEP = ("id", "q", "tr", "en", "c", "lat", "lon", "y", "cat", "img", "trw", "enw")


def load(name: str, default):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else default


def main() -> None:
    sites = load("sites.json", {"sites": []})["sites"]
    scans = load("scans.json", {"scans": []}).get("scans", [])
    models = load("sketchfab.json", {})
    sv = load("streetview.json", {})
    views = load("viewpoints.json", {})
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for s in sites:
        own = [x for x in scans if x.get("whc") == s["id"]]
        site = {k: s[k] for k in KEEP if k in s}
        if own and site["tr"] == site["en"] and own[0].get("site"):
            site["tr"] = own[0]["site"]
        doc = {"site": site, "scans": own, "models": models.get(str(s["id"]), []), "sv": sv.get(str(s["id"])), "svChecked": bool(sv),
               "pts": views.get(str(s["id"]), [])}
        (OUT / f"{s['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    # Küre için odak noktası: mirasın en bilinen yapısı (UNESCO kaydının kaba merkezi yerine)
    focus = {}
    for s in sites:
        pts = views.get(str(s["id"])) or []
        if pts and (abs(pts[0][0] - s["lat"]) > 0.001 or abs(pts[0][1] - s["lon"]) > 0.001):
            focus[str(s["id"])] = [pts[0][0], pts[0][1]]
    (DATA / "focus.json").write_text(json.dumps(focus, separators=(",", ":")))

    # Küre sayfasının tek ve küçük verisi (globe.json): liste, arama, filtre ve uçuş için gerekenler.
    # Fotoğraf, Vikipedi ve Wikidata bağlantıları seçilince data/miras/<no>.json'dan gelir.
    cams = load("views.json", {})
    countries = sorted({c for s in sites for c in s["c"]})
    cidx = {c: i for i, c in enumerate(countries)}
    cats = {"kültürel": 0, "doğal": 1, "karma": 2}
    rows = []
    for s in sites:
        sid = str(s["id"])
        own = [x for x in scans if x.get("whc") == s["id"]]
        tr = own[0]["site"] if own and s["tr"] == s["en"] and own[0].get("site") else s["tr"]
        lat, lon = focus.get(sid, [s["lat"], s["lon"]])
        v = cams.get(sid) or [lat, lon, 15.3, 62, -18, 0]
        rows.append([s["id"], tr, s["en"] if s["en"] != tr else "", [cidx[c] for c in s["c"]], s.get("y") or 0,
                     cats.get(s.get("cat"), 3), round(lon, 5), round(lat, 5), len(models.get(sid, [])),
                     v[2], v[3], v[4], v[5] if len(v) > 5 else 0])
    globe = {"about": "[no, tr, en, ülke indeksleri, yıl, tür(0 kültürel,1 doğal,2 karma,3 ?), boylam, enlem, model sayısı, zoom, pitch, bearing, yükseklik]",
             "countries": countries, "scans": [x for x in scans if x.get("whc")], "sites": rows}
    (DATA / "globe.json").write_text(json.dumps(globe, ensure_ascii=False, separators=(",", ":")))
    print(f"[miras] {len(sites)} dosya yazıldı -> {OUT}; {len(focus)} mirasın odak noktası düzeltildi")


if __name__ == "__main__":
    main()
