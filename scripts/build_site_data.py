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
    print(f"[miras] {len(sites)} dosya yazıldı -> {OUT}")


if __name__ == "__main__":
    main()
