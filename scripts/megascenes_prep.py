"""MegaScenes'ten (Wikimedia Commons fotoğrafları + COLMAP) eğitime hazır sahne hazırlar.

MegaScenes (Tung ve ark., 2024) Commons'taki serbest lisanslı fotoğraflardan
yüz binlerce yer için kamera konumlarını hesaplamış ve herkese açık bir S3
kovasında yayınlamış. Bu betik bir sahnenin bir bileşenini indirir, iyi
fotoğrafları seçer, bozulmayı giderir ve `python -m vid2splat train` için
klasörü yazar:

    <out>/scene/images/...      bozulması giderilmiş fotoğraflar
    <out>/scene/sparse/0/       PINHOLE kameralı COLMAP modeli
    <out>/credits.json          kullanılan Commons dosyaları (atıf için)

Kullanım:
    python scripts/megascenes_prep.py --scene 174752 --component 2 --out /content/out/ayasofya
    python scripts/megascenes_prep.py --list "Hagia_Sophia"   # sahne numarasını bul

Fotoğrafların her biri kendi Commons lisansıyla (CC BY, CC BY-SA, kamu malı...)
gelir; üretilen sahne CC BY-SA 4.0 olarak paylaşılmalı ve credits.json'daki
dosyalara atıf yapılmalıdır.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pycolmap

BUCKET = "https://megascenes.s3.us-west-2.amazonaws.com"
# Sahnenin bugünkü halini göstermeyen alt kategoriler
SKIP_SUBCATS = re.compile(r"historical|scaffold|drawing|painting|engraving|map|plan|model|postcard|lithograph"
                          r"|night|restoration|construction|before|\b1[0-9]{3}\b", re.I)


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return True
        except Exception:  # noqa: BLE001 - ağ hataları tekrar denenir
            continue
    return False


def scene_prefix(scene_id: int) -> str:
    return f"{scene_id // 1000:03d}/{scene_id % 1000:03d}"


def list_categories(pattern: str) -> None:
    cache = Path("/tmp/megascenes_categories.json")
    if not cache.exists():
        fetch(f"{BUCKET}/metadata/categories.json", cache)
    cats = json.loads(cache.read_text())
    rx = re.compile(pattern, re.I)
    for name, sid in sorted(cats.items(), key=lambda kv: kv[0]):
        if rx.search(name):
            print(sid, name)


def camera_center(image: pycolmap.Image) -> np.ndarray:
    return np.asarray(image.projection_center())


def select_images(rec: pycolmap.Reconstruction, max_images: int, min_points: int) -> list[int]:
    cand = []
    for image_id, im in rec.images.items():
        sub = im.name.split("/")[1] if im.name.count("/") >= 2 else ""
        if SKIP_SUBCATS.search(sub):
            continue
        n = im.num_points3D
        if n < min_points:
            continue
        cand.append((n, image_id))
    if not cand:
        raise SystemExit("Uygun fotoğraf bulunamadı")
    cand.sort(reverse=True)
    ids = [i for _, i in cand]
    centers = np.array([camera_center(rec.images[i]) for i in ids])
    med = np.median(centers, 0)
    d = np.linalg.norm(centers - med, axis=1)
    ok = d < 4 * np.median(d) + 1e-9
    ids = [i for i, k in zip(ids, ok) if k]
    return ids[:max_images]


def subset(rec: pycolmap.Reconstruction, keep_ids: list[int], min_track: int) -> pycolmap.Reconstruction:
    keep = set(keep_ids)
    frames = {rec.images[i].frame_id for i in keep}
    for frame_id in list(rec.reg_frame_ids()):
        if frame_id not in frames:
            rec.deregister_frame(frame_id)
    for pid in list(rec.point3D_ids()):
        if rec.points3D[pid].track.length() < min_track:
            rec.delete_point3D(pid)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="kategori adında ara (düzenli ifade) ve sahne numaralarını yaz")
    ap.add_argument("--scene", type=int)
    ap.add_argument("--component", type=int, default=0, help="colmap/<n> bileşeni")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--min-points", type=int, default=400, help="fotoğraf başına en az 3D nokta")
    ap.add_argument("--max-side", type=int, default=1600, help="bozulma giderilmiş fotoğrafın uzun kenarı")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.list:
        return list_categories(args.list)
    if args.scene is None or args.out is None:
        ap.error("--scene ve --out gerekli")

    out: Path = args.out
    raw = out / "raw"
    pre = scene_prefix(args.scene)
    model_dir = raw / "model"
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (model_dir / f).exists():
            print(f"[megascenes] indiriliyor: {f}")
            if not fetch(f"{BUCKET}/reconstruct/{pre}/colmap/{args.component}/{f}", model_dir / f):
                raise SystemExit(f"{f} indirilemedi")

    rec = pycolmap.Reconstruction(model_dir)
    print(f"[megascenes] bileşen: {rec.num_reg_images()} fotoğraf, {rec.num_points3D():,} nokta")
    ids = select_images(rec, args.max_images, args.min_points)
    print(f"[megascenes] seçilen: {len(ids)} fotoğraf")

    img_dir = raw / "images"
    def get(image_id: int) -> tuple[int, bool]:
        name = rec.images[image_id].name
        dest = img_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            return image_id, True
        url = f"{BUCKET}/images/{pre}/" + urllib.parse.quote(name)
        return image_id, fetch(url, dest)

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(get, ids))
    ok_ids = [i for i, ok in results if ok]
    print(f"[megascenes] indirilen: {len(ok_ids)}/{len(ids)}")

    rec = subset(rec, ok_ids, min_track=3)
    sub_dir = raw / "subset"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rec.write(sub_dir)
    print(f"[megascenes] alt model: {rec.num_reg_images()} fotoğraf, {rec.num_points3D():,} nokta")

    und = raw / "undistorted"
    if und.exists():
        shutil.rmtree(und)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = args.max_side
    pycolmap.undistort_images(und, sub_dir, img_dir, undistort_options=opts)

    scene = out / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    (scene / "sparse" / "0").mkdir(parents=True)
    urec = pycolmap.Reconstruction(und / "sparse")
    urec.write(scene / "sparse" / "0")
    shutil.move(str(und / "images"), str(scene / "images"))

    credits = []
    for im in urec.images.values():
        file = im.name.split("/")[-1]
        credits.append({"file": file, "url": "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(file.replace(" ", "_"))})
    (out / "credits.json").write_text(json.dumps({
        "source": "MegaScenes (Tung et al., 2024), Wikimedia Commons",
        "scene": args.scene, "component": args.component, "files": sorted(credits, key=lambda c: c["file"]),
    }, ensure_ascii=False, indent=1))
    print(f"[megascenes] hazır: {scene} ({urec.num_reg_images()} fotoğraf)")


if __name__ == "__main__":
    sys.exit(main())
