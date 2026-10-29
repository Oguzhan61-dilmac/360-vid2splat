"""UNESCO Dünya Mirası listesini Wikidata'dan derleyip web/data/sites.json dosyasını üretir.

Wikidata verisi CC0 lisanslıdır. Her miras için: UNESCO numarası, Türkçe ve İngilizce
ad, ülke(ler), koordinat, listeye giriş yılı, kategori (kültürel / doğal / karma;
UNESCO ölçütlerinden türetilir), Türkçe ve İngilizce Vikipedi başlığı ve Commons
görseli.

Kullanım:
    python scripts/build_sites.py              # web/data/sites.json yazar (önbelleği kullanır)
    python scripts/build_sites.py --yenile     # Wikidata'dan baştan çeker
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

# Önce QLever (Freiburg Üniversitesi'nin Wikidata kopyası), olmazsa Wikidata'nın kendi servisi
ENDPOINTS = ["https://qlever.dev/api/wikidata", "https://query.wikidata.org/sparql"]
PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
"""
USER_AGENT = "360-vid2splat-heritage/0.1 (https://github.com/Oguzhan61-dilmac/360-vid2splat)"
OUT = Path(__file__).resolve().parents[1] / "web" / "data" / "sites.json"

# Sadece hâlâ listede olan ana kayıtlar (Q9259 = World Heritage Site, bitiş tarihi yok)
BASE = """
SELECT ?item ?whc ?coord ?start ?en ?tr ?image WHERE {
  ?item p:P1435 ?st .
  ?st ps:P1435 wd:Q9259 .
  FILTER NOT EXISTS { ?st pq:P582 ?end }
  OPTIONAL { ?st pq:P580 ?start }
  ?item wdt:P757 ?whc .
  FILTER(REGEX(?whc, "^[0-9]+(bis|ter|quater|rev)?$"))
  OPTIONAL { ?item wdt:P625 ?coord }
  OPTIONAL { ?item wdt:P18 ?image }
  OPTIONAL { ?item rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?item rdfs:label ?tr FILTER(LANG(?tr) = "tr") }
}
"""

COUNTRIES = """
SELECT ?item ?country ?cen ?ctr ?iso WHERE {
  ?item p:P1435 ?st . ?st ps:P1435 wd:Q9259 .
  FILTER NOT EXISTS { ?st pq:P582 ?end }
  ?item wdt:P757 ?whc . FILTER(REGEX(?whc, "^[0-9]+(bis|ter|quater|rev)?$"))
  ?item wdt:P17 ?country .
  OPTIONAL { ?country wdt:P297 ?iso }
  OPTIONAL { ?country rdfs:label ?cen FILTER(LANG(?cen) = "en") }
  OPTIONAL { ?country rdfs:label ?ctr FILTER(LANG(?ctr) = "tr") }
}
"""

CRITERIA = """
SELECT ?item ?critLabel WHERE {
  ?item p:P1435 ?st . ?st ps:P1435 wd:Q9259 .
  FILTER NOT EXISTS { ?st pq:P582 ?end }
  ?item wdt:P757 ?whc . FILTER(REGEX(?whc, "^[0-9]+(bis|ter|quater|rev)?$"))
  ?item wdt:P2614 ?crit .
  ?crit rdfs:label ?critLabel FILTER(LANG(?critLabel) = "en")
}
"""

SITELINKS = """
SELECT ?item ?trwiki ?enwiki WHERE {
  ?item p:P1435 ?st . ?st ps:P1435 wd:Q9259 .
  FILTER NOT EXISTS { ?st pq:P582 ?end }
  ?item wdt:P757 ?whc . FILTER(REGEX(?whc, "^[0-9]+(bis|ter|quater|rev)?$"))
  OPTIONAL { ?tra schema:about ?item ; schema:isPartOf <https://tr.wikipedia.org/> ; schema:name ?trwiki }
  OPTIONAL { ?ena schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?enwiki }
}
"""

# UNESCO numarası girilmiş ama "Dünya Mirası" etiketi eksik öğeler (Wikidata'daki boşluklar)
EXTRA = """
SELECT ?item ?whc ?coord ?en ?tr ?image ?country ?cen ?ctr ?iso ?trwiki ?enwiki ?critLabel WHERE {
  ?item wdt:P757 ?whc .
  FILTER(REGEX(?whc, "^[0-9]+(bis|ter|quater|rev)?$"))
  FILTER NOT EXISTS { ?item p:P1435 ?st . ?st ps:P1435 wd:Q9259 . }
  OPTIONAL { ?item wdt:P625 ?coord }
  OPTIONAL { ?item wdt:P18 ?image }
  OPTIONAL { ?item rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?item rdfs:label ?tr FILTER(LANG(?tr) = "tr") }
  OPTIONAL { ?item wdt:P17 ?country .
             OPTIONAL { ?country wdt:P297 ?iso }
             OPTIONAL { ?country rdfs:label ?cen FILTER(LANG(?cen) = "en") }
             OPTIONAL { ?country rdfs:label ?ctr FILTER(LANG(?ctr) = "tr") } }
  OPTIONAL { ?tra schema:about ?item ; schema:isPartOf <https://tr.wikipedia.org/> ; schema:name ?trwiki }
  OPTIONAL { ?ena schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?enwiki }
  OPTIONAL { ?item wdt:P2614 ?crit . ?crit rdfs:label ?critLabel FILTER(LANG(?critLabel) = "en") }
}
"""
# Listeden çıkarılmış alanlar: Arap Oriksi Koruma Alanı, Liverpool, Dresden Elbe Vadisi
DELISTED = {654, 1150, 1156}

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


_last_query = 0.0
MIN_GAP_SEC = 20  # iki servis de sık sorguyu sınırlıyor


def sparql(query: str) -> list[dict]:
    global _last_query
    for attempt in range(8):
        endpoint = ENDPOINTS[attempt % len(ENDPOINTS)]
        wait = _last_query + MIN_GAP_SEC - time.time()
        if _last_query and wait > 0:
            time.sleep(wait)
        _last_query = time.time()
        url = endpoint + "?" + urllib.parse.urlencode({"query": PREFIXES + query})
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                rows = json.load(r)["results"]["bindings"]
                print(f"  {len(rows)} satır ({endpoint.split('/')[2]})")
                return rows
        except urllib.error.HTTPError as err:  # 429: hız sınırı, 5xx: geçici hata
            retry = int(err.headers.get("Retry-After") or 0) if err.headers else 0
            pause = min(max(retry, MIN_GAP_SEC), 120)
            print(f"  {endpoint.split('/')[2]}: HTTP {err.code}; {pause} sn sonra tekrar", file=sys.stderr)
            time.sleep(pause)
        except Exception as err:
            print(f"  {endpoint.split('/')[2]}: {err}; tekrar deneniyor", file=sys.stderr)
            time.sleep(MIN_GAP_SEC)
    raise RuntimeError("Wikidata sorgusu başarısız oldu")


def whc_num(value: str) -> int:
    """'1018rev', '208bis' gibi UNESCO numaralarının sayı kısmı."""
    return int(re.match(r"\d+", value).group(0))


def qid(b: dict) -> str:
    return b["item"]["value"].rsplit("/", 1)[-1]


def val(b: dict, key: str) -> str | None:
    return b[key]["value"] if key in b else None


def parse_point(wkt: str) -> tuple[float, float] | None:
    m = re.search(r"point\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", wkt or "", re.I)
    return (float(m.group(2)), float(m.group(1))) if m else None  # (enlem, boylam)


def category(criteria: set[int]) -> str:
    cultural = any(c <= 6 for c in criteria)
    natural = any(c >= 7 for c in criteria)
    if cultural and natural:
        return "karma"
    if natural:
        return "doğal"
    if cultural:
        return "kültürel"
    return "?"


CACHE = Path(__file__).resolve().parent / ".wikidata-cache.json"


def fetch_all(refresh: bool) -> dict[str, list[dict]]:
    """Sorgu sonuçlarını önbellekten okur; yoksa ya da --yenile verildiyse Wikidata'dan çeker."""
    cache = {} if refresh or not CACHE.exists() else json.loads(CACHE.read_text())
    for key, label, query in [("base", "temel bilgiler", BASE), ("countries", "ülkeler", COUNTRIES),
                              ("criteria", "UNESCO ölçütleri", CRITERIA), ("links", "Vikipedi bağlantıları", SITELINKS),
                              ("extra", "etiketi eksik kayıtlar", EXTRA)]:
        if key in cache:
            continue
        print(f"Wikidata: {label}…")
        cache[key] = sparql(query)
        CACHE.write_text(json.dumps(cache))
    return cache


def main() -> None:
    raw = fetch_all("--yenile" in sys.argv)
    base, countries, criteria, links = raw["base"], raw["countries"], raw["criteria"], raw["links"]
    # Etiketi eksik kayıtları aynı biçime çevirip ekle (ana listede olmayan numaralar)
    known = {whc_num(b["whc"]["value"]) for b in base}
    extra_added = set()
    for b in raw.get("extra", []):
        w = whc_num(b["whc"]["value"])
        if w in known or w in DELISTED:
            continue
        extra_added.add(w)
        base.append({k: b[k] for k in ("item", "whc", "coord", "en", "tr", "image") if k in b})
        if "country" in b:
            countries.append({k: b[k] for k in ("item", "country", "cen", "ctr", "iso") if k in b})
        if "critLabel" in b:
            criteria.append({"item": b["item"], "critLabel": b["critLabel"]})
        links.append({k: b[k] for k in ("item", "trwiki", "enwiki") if k in b})
    if extra_added:
        print(f"etiketi eksik {len(extra_added)} kayıt eklendi")

    items: dict[str, dict] = {}
    for b in base:
        q = qid(b)
        it = items.setdefault(q, {"qid": q, "coords": set(), "years": set(), "images": []})
        it["whc"] = whc_num(b["whc"]["value"])
        it["en"] = it.get("en") or val(b, "en")
        it["tr"] = it.get("tr") or val(b, "tr")
        p = parse_point(val(b, "coord"))
        if p:
            it["coords"].add(p)
        if val(b, "start"):
            it["years"].add(int(val(b, "start")[:4]))
        img = val(b, "image")
        if img and img not in it["images"]:
            it["images"].append(img)

    country_of: dict[str, dict] = defaultdict(dict)
    for b in countries:
        cid = b["country"]["value"].rsplit("/", 1)[-1]
        country_of[qid(b)][cid] = {"iso": val(b, "iso"), "en": val(b, "cen"), "tr": val(b, "ctr")}

    crit_of: dict[str, set[int]] = defaultdict(set)
    for b in criteria:
        m = re.search(r"\(([ivx]+)\)", b["critLabel"]["value"])
        if m and m.group(1) in ROMAN:
            crit_of[qid(b)].add(ROMAN[m.group(1)])

    links_of: dict[str, dict] = defaultdict(dict)
    for b in links:
        for k in ("trwiki", "enwiki"):
            if val(b, k):
                links_of[qid(b)][k] = val(b, k)

    # Aynı UNESCO numarasına bağlı birden çok Wikidata öğesi olabilir; en eksiksizini seç.
    by_whc: dict[int, list[dict]] = defaultdict(list)
    for it in items.values():
        by_whc[it["whc"]].append(it)

    sites = []
    for whc, group in sorted(by_whc.items()):
        def score(it):
            return (bool(it["coords"]), bool(links_of[it["qid"]].get("trwiki")), bool(it["tr"]),
                    bool(links_of[it["qid"]].get("enwiki")), bool(it["images"]), -int(it["qid"][1:]))
        it = max(group, key=score)
        coords = sorted(it["coords"]) or sorted(set().union(*(g["coords"] for g in group)))
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        crits = set().union(*(crit_of[g["qid"]] for g in group))
        cs = {}
        for g in group:
            cs.update(country_of[g["qid"]])
        years = set().union(*(g["years"] for g in group))
        lk = links_of[it["qid"]]
        # Türkçe etiket yoksa Türkçe Vikipedi maddesinin adını kullan ("Efes (antik kent)" -> "Efes")
        tr_name = it["tr"] or (re.sub(r"\s*\([^)]*\)$", "", lk["trwiki"]) if lk.get("trwiki") else None)
        site = {
            "id": whc,
            "q": it["qid"],
            "tr": tr_name or it["en"] or lk.get("enwiki") or f"UNESCO mirası {whc}",
            "en": it["en"] or lk.get("enwiki") or it["tr"] or f"World Heritage site {whc}",
            "c": sorted({(c["tr"] or c["en"] or "?") for c in cs.values()}),
            "iso": sorted({c["iso"] for c in cs.values() if c["iso"]}),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "y": min(years) if years else None,
            "cat": category(crits),
            "crit": sorted(crits),
        }
        if lk.get("trwiki"):
            site["trw"] = lk["trwiki"]
        if lk.get("enwiki"):
            site["enw"] = lk["enwiki"]
        if it["images"]:
            site["img"] = urllib.parse.unquote(it["images"][0].rsplit("/", 1)[-1])
        sites.append(site)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Wikidata (CC0), UNESCO Dünya Mirası Listesi kayıtları",
        "generated": time.strftime("%Y-%m-%d"),
        "count": len(sites),
        "sites": sites,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    cats = defaultdict(int)
    for s in sites:
        cats[s["cat"]] += 1
    print(f"{len(sites)} miras yazıldı -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB) | {dict(cats)}")
    print(f"Türkçe ad: {sum(1 for s in sites if s['tr'] != s['en'])}, Türkçe Vikipedi: {sum(1 for s in sites if 'trw' in s)}")


if __name__ == "__main__":
    main()
