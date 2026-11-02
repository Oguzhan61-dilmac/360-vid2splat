"""Her miras için Street View'ın açılacağı "bakış noktalarını" Wikidata'dan derler.

Bir mirasın UNESCO kaydındaki koordinat çoğu zaman alanın kaba merkezidir; Street View
oraya yakın görüntü bulamayabilir (ör. Efes'in kaydı Çukuriçi Höyük'e düşüyor). Bu
betik aynı mirasa ait Wikidata öğelerini toplar:

  * Dünya Mirası numarası (P757) taşıyan öğeler: Ayasofya, Efes antik kenti...
  * bu öğelerin parçaları (P361 "parçası"),
  * büyük mirasların içindeki ünlü yapılar (aşağıdaki LANDMARKS; Wikidata etiketiyle),

ve her miras için en bilinen (en çok Vikipedi bağlantısı olan) en fazla 8 noktayı,
birbirinden en az 60 m uzak olacak şekilde web/data/viewpoints.json'a yazar:

    {"1018": [[37.939722, 27.340833, "Efes"], [37.949722, 27.363889, "Artemis Tapınağı"], ...]}

Çalıştırma (anahtar gerekmez, Wikidata Sorgu Servisi):

    python scripts/viewpoints.py && python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "web" / "data" / "sites.json"
OUT = ROOT / "web" / "data" / "viewpoints.json"
WDQS = "https://query.wikidata.org/sparql"
UA = "MirasGezgini/1.0 (https://github.com/Oguzhan61-dilmac/360-vid2splat)"

LABELS = 'OPTIONAL { ?item rdfs:label ?tr FILTER(lang(?tr)="tr") } OPTIONAL { ?item rdfs:label ?en FILTER(lang(?en)="en") }'
Q_OWN = f"SELECT ?id ?item ?coord ?links ?tr ?en WHERE {{ ?item wdt:P757 ?id ; wdt:P625 ?coord ; wikibase:sitelinks ?links . {LABELS} }}"
Q_PARTS = (f"SELECT ?id ?item ?coord ?links ?tr ?en WHERE {{ ?whc wdt:P757 ?id . ?item wdt:P361 ?whc ; wdt:P625 ?coord ; "
           f"wikibase:sitelinks ?links . FILTER(?links > 2) {LABELS} }}")

# Büyük mirasların içindeki ünlü yapılar (Wikidata'daki İngilizce etiketleriyle)
LANDMARKS = {
    356: ["Hagia Sophia", "Sultan Ahmed Mosque", "Süleymaniye Mosque", "Topkapı Palace", "Basilica Cistern", "Hagia Irene"],
    91: ["Colosseum", "Pantheon", "Trevi Fountain", "Roman Forum", "St. Peter's Basilica", "Spanish Steps", "Piazza Navona"],
    600: ["Notre-Dame de Paris", "Eiffel Tower", "Louvre Palace", "Pont Neuf"],
    404: ["Parthenon", "Erechtheion", "Propylaea", "Odeon of Herodes Atticus"],
    86: ["Great Pyramid of Giza", "Great Sphinx of Giza", "Pyramid of Djoser", "Bent Pyramid", "Pyramid of Khafre"],
    320: ["Sagrada Família", "Casa Batlló", "Park Güell", "Casa Milà", "Palau Güell"],
    426: ["Westminster Abbey", "Big Ben", "Palace of Westminster"],
    545: ["Saint Basil's Cathedral", "Moscow Kremlin", "Red Square"],
    616: ["Charles Bridge", "Prague astronomical clock", "Prague Castle", "Old Town Square"],
    174: ["Florence Cathedral", "Ponte Vecchio", "Palazzo Vecchio", "Uffizi"],
    394: ["St Mark's Basilica", "Rialto Bridge", "Doge's Palace", "Piazza San Marco"],
    668: ["Angkor Wat", "Bayon", "Ta Prohm"],
    1018: ["Library of Celsus", "Great Theatre of Ephesus"],
    357: ["Göreme Open Air Museum", "Uçhisar"],
    485: ["Pamukkale", "Hierapolis"],
    483: ["El Castillo, Chichen Itza"],
    414: ["Pyramid of the Sun", "Pyramid of the Moon"],
    326: ["Al-Khazneh", "Ad Deir"],
    438: ["Mutianyu", "Badaling"],
    439: ["Forbidden City"],
    274: ["Machu Picchu"],
}


def sparql(q: str) -> list[dict]:
    url = WDQS + "?" + urllib.parse.urlencode({"query": q})
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:  # noqa: BLE001 - zaman aşımı/yoğunluk, tekrar dene
            print(f"[viewpoints] sorgu hatası ({e}); tekrar deneniyor")
            time.sleep(15 * (attempt + 1))
    raise SystemExit("Wikidata sorgusu başarısız")


def point(b: dict):
    c = re.match(r"Point\(([-\d.eE+]+) ([-\d.eE+]+)\)", b["coord"]["value"])
    if not c:
        return None
    name = (b.get("tr") or b.get("en") or {}).get("value", "")
    return float(c.group(2)), float(c.group(1)), name, int(b["links"]["value"])


def dist(a, b) -> float:
    return math.hypot((a[0] - b[0]) * 111320, (a[1] - b[1]) * 111320 * math.cos(math.radians(a[0])))


def main() -> None:
    sites = {s["id"]: s for s in json.loads(SITES.read_text())["sites"]}
    pts: dict[int, dict[str, tuple]] = defaultdict(dict)
    for q in (Q_OWN, Q_PARTS):
        for b in sparql(q):
            m = re.match(r"(\d+)", b["id"]["value"])
            p = point(b)
            if m and p and int(m.group(1)) in sites:
                pts[int(m.group(1))][b["item"]["value"]] = p

    labels = sorted({lab for v in LANDMARKS.values() for lab in v})
    values = " ".join(json.dumps(lab, ensure_ascii=False) + "@en" for lab in labels)
    q = (f"SELECT ?label ?item ?coord ?links ?tr ?en WHERE {{ VALUES ?label {{ {values} }} ?item rdfs:label ?label ; "
         f"wdt:P625 ?coord ; wikibase:sitelinks ?links . {LABELS} }}")
    found: dict[str, list] = defaultdict(list)
    for b in sparql(q):
        p = point(b)
        if p:
            found[b["label"]["value"]].append((b["item"]["value"], p))
    for sid, labs in LANDMARKS.items():
        site = sites[sid]
        for rank, lab in enumerate(labs):
            # aynı adlı birden çok yer olabilir (Paris'te ve Saint-Germain'de Sainte-Chapelle): mirasa en yakını
            near = [(dist((p[0], p[1]), (site["lat"], site["lon"])), item, p) for item, p in found.get(lab, [])]
            near = [x for x in near if x[0] < 30000]
            if near:
                _, item, p = min(near)
                pts[sid][item] = (p[0], p[1], p[2], 100000 - rank)  # listedeki sırayla öne

    out = {}
    for sid, site in sites.items():
        cand = sorted(pts.get(sid, {}).values(), key=lambda p: -p[3])
        cand.append((site["lat"], site["lon"], site["tr"], -1))
        chosen = []
        for p in cand:
            if all(dist(p, c) > 60 for c in chosen):
                chosen.append(p)
            if len(chosen) >= 8:
                break
        out[str(sid)] = [[round(p[0], 6), round(p[1], 6), p[2]] for p in chosen]
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    multi = sum(1 for v in out.values() if len(v) > 1)
    print(f"[viewpoints] {len(out)} miras, {multi} tanesinde birden çok nokta -> {OUT}")


if __name__ == "__main__":
    main()
