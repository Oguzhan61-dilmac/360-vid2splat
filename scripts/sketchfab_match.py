"""Her UNESCO mirası için Sketchfab'daki hazır 3D modelleri bulur.

Sketchfab'ın herkese açık arama API'si anahtar istemez. Betik her miras için
adındaki ayırt edici kelimelerle (ve ünlü yapıları olan mirasler için yapı
adlarıyla) arar, alakasız sonuçları (karakter, araba, düşük çokgenli oyun
modeli...) eler, kalanları beğeni, görüntülenme ve fotogrametri işaretlerine
göre sıralar ve web/data/sketchfab.json'a yazar:

    {"356": [{"uid": "...", "n": "Hagia Sophia", "a": "yazar", "u": "kullanıcı",
              "l": "CC Attribution", "t": "küçük resim", "f": 3176556, "q": "Hagia Sophia"}, ...]}

Modeller sitede Sketchfab'ın kendi oynatıcısıyla gömülür (dosya indirilmez),
yazar adı ve modelin bağlantısı her zaman gösterilir.

Kullanım:
    python scripts/sketchfab_match.py                 # tüm mirasler
    python scripts/sketchfab_match.py --ids 356 91    # yalnızca bunlar (mevcut dosyaya eklenir)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "web" / "data" / "sites.json"
OUT = ROOT / "web" / "data" / "sketchfab.json"
INDEX = ROOT / "web" / "data" / "sketchfab-index.json"
API = "https://api.sketchfab.com/v3/search"

STOP = set("""of the and its in at on with de del della dei di la le les des du da do dos das el los las y et und der die
von zu a an to for from including other its their en au aux sur sous e o i that this see holy""".split())
GENERIC = set("""historic historical centre center city town old area areas site sites archaeological ancient national park
cultural landscape landscapes monuments monument complex ensemble ruins heritage world property properties group remains
reserve nature natural region sanctuary routes route works architecture buildings building early medieval prehistoric
settlement settlements cities towns district quarter extension serial transboundary system systems protected
environmental areas zone lands land traditional associated related outstanding people peoples reconstructed
route way road roads trail sacred royal imperial colonial modern new upper lower central northern southern eastern western""".split())
TYPEWORDS = set("""mosque mosques church churches cathedral temple temples palace palaces castle castles fortress fortresses
fort forts tomb tombs cave caves monastery monasteries abbey basilica wall walls bridge bridges tower towers pyramid pyramids
mount mountain mountains lake lakes island islands river falls canyon desert forest forests bay coast garden gardens villa
villas rock rocks art paintings synagogue theatre amphitheatre aqueduct mine mines port harbour canal canals house houses
village villages great valley necropolis city-state hill hills square fountain library mausoleum citadel kasbah medina""".split())

# Kategoriler: alakasız türleri (karakter, araç, silah...) dışarıda bırakır
OK_CATS = {"cultural-heritage-history", "architecture", "places-travel"}
OK_CATS_NATURAL = OK_CATS | {"nature-plants"}
PHOTO_HINTS = ("photogrammetry", "fotogrametria", "realitycapture", "reality-capture", "scan", "3dscan", "drone",
               "agisoft", "metashape", "zephyr", "lidar", "pix4d", "cultural-heritage")
LOWPOLY = ("lowpoly", "low-poly", "low_poly", "gameasset", "game-asset", "stylized", "minecraft", "voxel", "cartoon")

# Büyük mirasların içindeki ünlü yapılar: (sorgu, tüm kelimeler zorunlu mu)
LANDMARKS = {
    356: ["Hagia Sophia", "Blue Mosque", "Suleymaniye Mosque", "Topkapi Palace", "Basilica Cistern", "Hagia Irene"],
    91: ["Colosseum", "Pantheon Rome", "Trevi Fountain", "Roman Forum", "Saint Peter Basilica", "Arch of Constantine"],
    600: ["Notre-Dame Paris", "Eiffel Tower", "Sainte-Chapelle", "Arc de Triomphe"],
    404: ["Parthenon", "Erechtheion", "Acropolis Athens", "Propylaea"],
    86: ["Great Pyramid Giza", "Great Sphinx", "Pyramid Djoser", "Bent Pyramid"],
    252: ["Taj Mahal"],
    438: ["Great Wall China"],
    439: ["Forbidden City"],
    274: ["Machu Picchu"],
    320: ["Sagrada Familia", "Casa Batllo", "Park Guell", "Casa Mila"],
    373: ["Stonehenge"],
    426: ["Westminster Abbey", "Big Ben", "Palace of Westminster"],
    545: ["Saint Basil Cathedral", "Moscow Kremlin"],
    616: ["Charles Bridge", "Prague Astronomical Clock", "Prague Castle"],
    174: ["Florence Cathedral", "Ponte Vecchio", "Palazzo Vecchio"],
    394: ["Saint Mark Basilica", "Rialto Bridge", "Doge Palace"],
    668: ["Angkor Wat", "Bayon Temple", "Ta Prohm"],
    1018: ["Library of Celsus", "Ephesus"],
    357: ["Cappadocia", "Goreme"],
    485: ["Pamukkale", "Hierapolis"],
    1572: ["Gobekli Tepe"],
    448: ["Nemrut"],
    483: ["Chichen Itza", "Kukulcan"],
    414: ["Teotihuacan"],
    326: ["Petra Treasury", "Petra Monastery", "Petra Jordan"],
    114: ["Persepolis"],
    849: ["Troy"],
    377: ["Hattusa"],
    1405: ["Catalhoyuk"],
    1457: ["Pergamon"],
    1519: ["Aphrodisias"],
    1366: ["Selimiye Mosque"],
    1518: ["Ani Cathedral", "Ani Kars"],
}


COUNTRY_EXTRA = {
    "TR": ["turkey", "turkiye", "anatolia"], "GB": ["uk", "england", "scotland", "wales", "britain", "united kingdom"],
    "US": ["usa", "united states", "america"], "RU": ["russia"], "IR": ["iran", "persia"], "KR": ["korea"],
    "KP": ["north korea"], "CZ": ["czech", "czechia"], "VN": ["vietnam"], "SY": ["syria"], "BO": ["bolivia"],
    "VE": ["venezuela"], "TZ": ["tanzania"], "LA": ["laos"], "MD": ["moldova"], "MK": ["macedonia"],
    "PS": ["palestine"], "VA": ["vatican"], "IT": ["italia"], "ES": ["espana"], "DE": ["deutschland"],
    "MX": ["mexico"], "TW": ["taiwan"], "BA": ["bosnia"], "CD": ["congo"], "CG": ["congo"], "CI": ["ivory coast"],
    "NL": ["netherlands", "holland"], "MA": ["morocco", "maroc"], "GR": ["greece"], "EG": ["egypt"],
}
_COUNTRY_NAMES: dict[str, set[str]] = {}


def country_names() -> dict[str, set[str]]:
    if not _COUNTRY_NAMES:
        import pycountry
        for c in pycountry.countries:
            names = {c.name} | {getattr(c, a) for a in ("common_name", "official_name") if getattr(c, a, None)}
            names = {" ".join(words(n.split(",")[0])) for n in names}
            names |= set(COUNTRY_EXTRA.get(c.alpha_2, []))
            _COUNTRY_NAMES[c.alpha_2] = {n for n in names if len(n) > 3}
    return _COUNTRY_NAMES


def countries_in(text: str) -> set[str]:
    t = " " + " ".join(words(text)) + " "
    return {iso for iso, names in country_names().items() if any(f" {n} " in t for n in names)}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("ı", "i")


def words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", norm(s)) if w]


def site_query(site: dict) -> tuple[str, list[str]]:
    name = site.get("enw") or site["en"]
    name = re.sub(r"\(.*?\)", " ", name)
    ws = [w for w in words(name) if w not in STOP]
    distinct = [w for w in ws if w not in GENERIC and w not in TYPEWORDS and len(w) > 2]
    types = [w for w in ws if w in TYPEWORDS]
    if not distinct:
        distinct = types or [w for w in ws if len(w) > 2]
        types = []
    q = " ".join(distinct[:3] + types[:1])
    return q, distinct[:4]


def get(params: dict) -> list[dict]:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "miras-gezgini/1.0 (github.com/Oguzhan61-dilmac/360-vid2splat)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("results", [])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            time.sleep(2)
        except Exception:  # noqa: BLE001 - ağ hatası, tekrar dene
            time.sleep(2)
    return []


def has(tok: str, bag: set[str]) -> bool:
    if tok in bag:
        return True
    if len(tok) >= 5:
        return any(w.startswith(tok) or tok.startswith(w) and len(w) >= 5 for w in bag)
    return False


def relevant(m: dict, toks: list[str], need_all: bool, natural: bool, iso: set[str] | None = None) -> float:
    cats = {c.get("name") or c.get("slug") for c in m.get("categories") or []}
    tags = [t.get("name", "") for t in m.get("tags") or []]
    tagtxt = " ".join(tags)
    if cats and not cats & (OK_CATS_NATURAL if natural else OK_CATS):
        return 0
    if m.get("isAgeRestricted") or (m.get("faceCount") or 0) < 20000:
        return 0
    low = norm(m["name"] + " " + tagtxt)
    if any(k in low for k in LOWPOLY):
        return 0
    name_bag = set(words(m["name"]))
    tag_bag = set(words(tagtxt))
    in_name = [t for t in toks if has(t, name_bag)]
    in_any = [t for t in toks if has(t, name_bag | tag_bag)]
    if not in_name:
        return 0
    if need_all and len(in_any) < len(toks):
        return 0
    if not need_all and len(in_any) < min(2, len(toks)):
        return 0
    # başka bir ülkeden bahseden model (ör. Erivan'daki "Mavi Cami") bu miras değildir
    full = m["name"] + " " + tagtxt + " " + (m.get("description") or "")[:400]
    mentioned = countries_in(full)
    if iso and mentioned and not mentioned & iso:
        return 0
    # kısa ya da tek kelimelik adlar (Taï, Rabat...) ve doğal alanlar için ikinci bir işaret iste
    weak = natural or (len(in_any) == 1 and len(in_any[0]) < 6) or not cats
    if weak and not (iso and mentioned & iso) and len(in_any) < 2:
        return 0
    return len(in_name) * 2 + len(in_any)


def score(m: dict, rel: float) -> float:
    txt = norm(m["name"] + " " + " ".join(t.get("name", "") for t in m.get("tags") or []) + " " + (m.get("description") or "")[:400])
    s = rel * 3
    s += 3 * math.log10((m.get("likeCount") or 0) + 1) + math.log10((m.get("viewCount") or 0) + 1)
    if any(h in txt for h in PHOTO_HINTS):
        s += 3
    f = m.get("faceCount") or 0
    s += 1 if f > 300_000 else 0
    s += 1 if m.get("staffpickedAt") else 0
    if any((c.get("name") or c.get("slug")) == "cultural-heritage-history" for c in m.get("categories") or []):
        s += 2
    return s


def thumb(m: dict) -> str:
    imgs = sorted((m.get("thumbnails") or {}).get("images") or [], key=lambda i: i.get("width") or 0)
    for i in imgs:
        if (i.get("width") or 0) >= 250:
            return i["url"]
    return imgs[-1]["url"] if imgs else ""


def compact(m: dict, q: str) -> dict:
    lic = m.get("license")
    return {"uid": m["uid"], "n": m["name"][:90], "a": (m.get("user") or {}).get("displayName", "")[:60],
            "u": (m.get("user") or {}).get("username", ""), "l": lic.get("label") if isinstance(lic, dict) else "",
            "t": thumb(m), "f": m.get("faceCount") or 0, "q": q}


def match_site(site: dict) -> tuple[int, list[dict]]:
    natural = site.get("cat") == "doğal"
    iso = set(site.get("iso") or [])
    found: dict[str, tuple[float, dict]] = {}

    def add(results, toks, need_all, q, bonus=0.0):
        for m in results:
            r = relevant(m, toks, need_all, natural, iso)
            if r <= 0:
                continue
            sc = score(m, r) + bonus
            if m["uid"] not in found or found[m["uid"]][0] < sc:
                found[m["uid"]] = (sc, compact(m, q))

    for lm in LANDMARKS.get(site["id"], []):
        toks = [w for w in words(lm) if w not in STOP]
        add(get({"type": "models", "q": lm, "count": 24}), toks, True, lm, bonus=4)
    q, toks = site_query(site)
    if q:
        add(get({"type": "models", "q": q, "count": 24}), toks, False, q)
        if not found and len(toks) > 1:
            for t in sorted(toks, key=len, reverse=True)[:2]:
                if len(t) >= 5:
                    add(get({"type": "models", "q": t, "count": 24}), [t], True, t)
    best = sorted(found.values(), key=lambda kv: -kv[0])
    # ünlü yapılar listedeki sırayla öne gelsin (İstanbul'da önce Ayasofya, sonra Sultanahmet...)
    lms = LANDMARKS.get(site["id"], [])
    if lms:
        head = []
        for lm in lms:
            for kv in best:
                if kv[1]["q"] == lm and kv not in head:
                    head.append(kv)
                    break
        best = head + [kv for kv in best if kv not in head]
    # aynı yazarın aynı yapı serisini (ör. 30 plaka) tekrar tekrar göstermemek için yazar başına en fazla 2
    out, per_author = [], {}
    for _, c in best:
        if per_author.get(c["u"], 0) >= 2:
            continue
        per_author[c["u"]] = per_author.get(c["u"], 0) + 1
        out.append(c)
        if len(out) >= 6:
            break
    return site["id"], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    sites = json.loads(SITES.read_text())["sites"]
    if args.ids:
        sites = [s for s in sites if s["id"] in set(args.ids)]
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for sid, models in ex.map(match_site, sites):
            done += 1
            if models:
                data[str(sid)] = models
            else:
                data.pop(str(sid), None)
            if done % 50 == 0:
                print(f"[sketchfab] {done}/{len(sites)} miras, {len(data)} tanesinde model var", flush=True)
                OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    # küre sayfası için küçük özet: miras no -> model sayısı
    INDEX.write_text(json.dumps({k: len(v) for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))}, separators=(",", ":")))
    print(f"[sketchfab] bitti: {len(data)} mirasta model bulundu -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
