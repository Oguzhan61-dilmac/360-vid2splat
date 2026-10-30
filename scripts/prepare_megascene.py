"""Colab'da eğitilen MegaScenes sahnesini (zip) siteye eklenecek hale getirir.

İnternet fotoğraflarında ölçek ve "yukarı" yönü bilinmez. Bunları fotoğrafların
kendisinden çıkarıyoruz:
- Yukarı: insanlar fotoğrafı çoğunlukla düz tutar; kameraların ortalama "yukarı"sı.
- Göz hizası: fotoğrafların çoğu yerde duran birinin elinden çekilir; kamera
  yüksekliklerinin en kalabalık olduğu düzey.
- Zemin: o kameraların altındaki en yoğun nokta katmanı. Göz hizası ile zemin
  arası 1,6 m kabul edilerek sahne metreye çevrilir (ya da --known ile bilinen
  bir ölçüden).
Yürüme alanı, yerdeki fotoğrafçıların durduğu noktalardan bir rota ve onun
çevresindeki koridordur.

    python scripts/prepare_megascene.py ayasofya.zip --id ayasofya --name "Ayasofya" --indoor
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_trevi import matrix_to_quat, quat_mul  # noqa: E402
from vid2splat.align import _rotation_between, _yaw_rotation  # noqa: E402
from vid2splat.export import read_ply, write_ply  # noqa: E402

EYE = 1.6
SPLAT_TRANSFORM = ["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q"]


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([x[lab == i].mean(0) if np.any(lab == i) else c[i] for i in range(k)])
    return c


def analyze(cams: list[dict], points: np.ndarray, known: tuple[str, float] | None = None,
            flat: bool = False) -> dict:
    """flat=True: düz zeminli iç mekân (cami, kilise). Fotoğrafların çoğu ana zeminden çekilir;
    galerilerden çekilenler daha yukarıda ayrı bir küme oluşturur. Halı gibi dokusuz zemin
    hiç yeniden yapılamasa da en alttaki kalabalık kamera katmanı zemin + göz hizasıdır."""
    from scipy.spatial import cKDTree

    Rw2c = np.array([c["R"] for c in cams])
    centers = np.array([c["center"] for c in cams])
    # Yukarı yön: fotoğraflar yukarı/aşağı eğik çekilse de (kubbe fotoğrafları) kameranın
    # sağ ekseni yatay kalır; yukarı, bütün sağ eksenlere en dik yöndür.
    right = Rw2c[:, 0, :]
    w, v = np.linalg.eigh(right.T @ right)
    up = v[:, 0]
    if up @ (-Rw2c[:, 1, :].mean(0)) < 0:
        up = -up
    fwd = Rw2c[:, 2, :]
    R_up = _rotation_between(up / np.linalg.norm(up), np.array([0.0, 1.0, 0.0]))
    c_up = centers @ R_up.T
    p_up = points @ R_up.T
    xz = c_up[:, [0, 2]]
    extent = float(np.linalg.norm(np.percentile(xz, 95, 0) - np.percentile(xz, 5, 0))) or 1.0

    # Her kameranın altındaki yerel zemin: yakınındaki, kendisinden alçak noktaların alt %10'u.
    # Engebeli alanlarda (yokuş, basamak) tek bir düz zeminden çok daha güvenilir.
    tree = cKDTree(p_up[:, [0, 2]])
    gaps = np.full(len(cams), np.nan)
    for i, c in enumerate(c_up):
        idx = tree.query_ball_point(c[[0, 2]], 0.04 * extent)
        if len(idx) < 15:
            continue
        hs = p_up[idx, 1]
        below = hs[hs < c[1]]
        if len(below) >= 10:
            gaps[i] = c[1] - np.percentile(below, 10)
    valid = np.isfinite(gaps) & (gaps > 0)
    h = c_up[:, 1]
    if flat:
        # En alttaki kalabalık (>= %15) kamera katmanı
        b = 0.02 * extent
        grid = np.linspace(h.min(), h.max(), 400)
        dens = np.exp(-((h[None, :] - grid[:, None]) ** 2) / (2 * b * b)).sum(1)
        peaks = [i for i in range(1, len(grid) - 1) if dens[i] >= dens[i - 1] and dens[i] >= dens[i + 1]]
        level = next((grid[i] for i in peaks if (np.abs(h - grid[i]) < 2 * b).mean() >= 0.15), grid[int(np.argmax(dens))])
        ground_cams = np.abs(h - level) < 2 * b
        cam_level = float(np.median(h[ground_cams]))
        gv = valid & ground_cams
        typical = float(np.median(gaps[gv])) if gv.sum() >= 8 else np.nan
    else:
        if valid.sum() < 8:
            raise RuntimeError(f"zemini görülen kamera çok az ({valid.sum()})")
        # Fotoğrafların çoğu yerde duran birinden: tipik kamera-zemin aralığı göz hizasıdır.
        typical = float(np.percentile(gaps[valid], 40))
        ground_cams = valid & (gaps > 0.45 * typical) & (gaps < 1.9 * typical)
        cam_level = float(np.median(h[ground_cams]))
    scale_eye = EYE / typical if np.isfinite(typical) else np.nan

    scale, how = scale_eye, "göz hizası"
    if known:
        what, meters = known
        near = tree.query_ball_point(np.median(xz[ground_cams], 0), 0.6 * extent)
        top = float(np.percentile(p_up[near, 1], 99.7))
        # tepe - kamera düzeyi = bilinen yükseklik - göz hizası
        scale_known = (meters - EYE) / (top - cam_level) if what == "height" else meters / extent
        print(f"[hizalama] bilinen ölçü x{scale_known:.3f}, göz hizasından x{scale_eye:.3f}")
        scale, how = scale_known, "bilinen ölçü"
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"ölçek bulunamadı ({scale})")
    floor_y = cam_level - EYE / scale
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"ölçek bulunamadı ({scale})")

    # Başlangıç: yerdeki kameraların ortasına en yakın olanı, onun baktığı yöne
    gidx = np.where(ground_cams)[0]
    mid = np.median(xz[gidx], 0)
    start_i = gidx[np.argmin(np.linalg.norm(xz[gidx] - mid, axis=1))]
    R = _yaw_rotation(R_up @ fwd[start_i]) @ R_up

    def to_m(p: np.ndarray) -> np.ndarray:
        q = p @ R.T
        q[..., 1] -= floor_y
        return q * scale

    cm = to_m(centers)
    g = cm[ground_cams][:, [0, 2]]
    k = int(np.clip(len(g) // 10, 4, 18))
    way = kmeans(g, min(k, len(g)))
    start = cm[start_i]
    order, rest = [], list(range(len(way)))
    cur = start[[0, 2]]
    while rest:
        j = min(rest, key=lambda i: np.linalg.norm(way[i] - cur))
        order.append(j)
        rest.remove(j)
        cur = way[j]
    path = np.vstack([start[[0, 2]], way[order]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    nn = [np.sort(np.linalg.norm(way - w, axis=1))[1] for w in way] if len(way) > 1 else [4.0]
    radius = float(np.clip(0.9 * np.median(nn), 3.0, 10.0))

    # Zemin ızgarası: yerdeki fotoğrafçıların ayak hizası (kamera - göz yüksekliği), IDW ile
    feet = np.column_stack([g, cm[ground_cams, 1] - EYE])
    lo = g.min(0) - radius - 3
    hi = g.max(0) + radius + 3
    cell = max(1.0, float(np.sqrt((hi - lo).prod() / 15000)))
    W, H = int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    gx, gz = np.meshgrid(lo[0] + np.arange(W) * cell, lo[1] + np.arange(H) * cell)
    ftree = cKDTree(feet[:, :2])
    d, ii = ftree.query(np.column_stack([gx.ravel(), gz.ravel()]), k=min(8, len(feet)))
    wgt = 1 / np.maximum(d, 0.5) ** 2
    gy = (feet[ii, 2] * wgt).sum(1) / wgt.sum(1)
    relief = float(np.percentile(feet[:, 2], 95) - np.percentile(feet[:, 2], 5))
    ground = None
    if relief > 0.6 and not flat:  # düz zeminde ızgaraya gerek yok
        ground = {"origin": [round(float(lo[0]), 2), round(float(lo[1]), 2)], "cell": round(cell, 3), "w": W, "h": H,
                  "y": np.round(gy, 2).tolist()}

    print(f"[hizalama] {len(cams)} kamera, yerde {ground_cams.sum()}; ölçek x{scale:.3f} ({how}); "
          f"kamera alanı ~{extent * scale:.0f} m; zemin farkı {relief:.1f} m; rota {seg.sum():.0f} m, koridor {radius:.1f} m")
    return {"R": R, "floor_y": floor_y, "scale": scale, "to_m": to_m, "cams_m": cm, "ground": ground_cams,
            "start": start, "path": path, "radius": radius, "extent_m": extent * scale, "grid": ground,
            "relief": relief}


def transform_splats(ply_in: Path, ply_out: Path, A: dict, keep_radius: float) -> int:
    props, arr = read_ply(ply_in)
    col = {p: i for i, p in enumerate(props)}
    xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    p = A["to_m"](xyz)
    s = A["scale"]
    sizes = np.exp(arr[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1) * s
    opacity = 1 / (1 + np.exp(-arr[:, col["opacity"]]))
    cm = A["cams_m"]
    center = np.median(cm, 0)
    dist = np.linalg.norm(p - center, axis=1)
    d_cam = np.full(len(p), np.inf)
    for c in cm[:: max(1, len(cm) // 300)]:
        d_cam = np.minimum(d_cam, np.linalg.norm(p - c, axis=1))
    near_blob = (d_cam < 3.0) & (sizes > 0.3 * np.maximum(d_cam, 0.3))
    keep = (opacity > 0.04) & (sizes < 6.0) & (dist < keep_radius) & np.isfinite(p).all(1) & ~near_blob
    print(f"[temizlik] {len(arr):,} -> {keep.sum():,} Gaussian (uzak {(dist >= keep_radius).sum():,}, "
          f"kameraya yapışık {near_blob.sum():,})")
    out = arr[keep].copy()
    out[:, col["x"]], out[:, col["y"]], out[:, col["z"]] = p[keep].T
    for kk in ("scale_0", "scale_1", "scale_2"):
        out[:, col[kk]] += np.log(s)
    q = out[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]] = quat_mul(matrix_to_quat(A["R"]), q)
    # SH katsayıları dönüşle birlikte döndürülmüyor; sadece renk (DC) bırakıp yönlü yansımaları atıyoruz.
    rest = [p_ for p_ in props if p_.startswith("f_rest_")]
    if rest:
        keep_cols = [i for i, p_ in enumerate(props) if p_ not in rest]
        props = [props[i] for i in keep_cols]
        out = out[:, keep_cols]
    write_ply(ply_out, props, out)
    return int(keep.sum())


def build_scene(src_zip: Path, out_dir: Path, name: str, known_height: float | None = None,
                keep_radius: float = 0, note: str = "", indoor: bool = False, max_pitch: float = 45,
                flat: bool = False) -> dict:
    """Colab zip'inden siteye hazır sahne klasörü yazar; scene.json içeriğini döndürür."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(src_zip) as z:
            z.extractall(td)
        cams = json.loads((td / "cameras.json").read_text())["cameras"]
        points = np.load(td / "points.npy").astype(np.float64)
        A = analyze(cams, points, ("height", known_height) if known_height else None, flat=flat)
        subprocess.run(SPLAT_TRANSFORM + [str(td / "scene.spz"), str(td / "raw.ply")], check=True)
        keep_r = keep_radius or max(60.0, 2.5 * A["extent_m"])
        n = transform_splats(td / "raw.ply", td / "aligned.ply", A, keep_r)
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(SPLAT_TRANSFORM + [str(td / "aligned.ply"), str(out_dir / "scene.spz"), "--spz-version", "3"], check=True)
        credits = json.loads((td / "credits.json").read_text())
        metrics = json.loads((td / "metrics.json").read_text())

    (out_dir / "credits.json").write_text(json.dumps(credits, ensure_ascii=False, indent=1))
    (out_dir / "LICENSE.txt").write_text(
        f"{name} 3D sahnesi, Wikimedia Commons'taki {len(credits['files'])} serbest lisanslı fotoğraftan\n"
        "(MegaScenes veri setinin kamera konumlarıyla) Miras Gezgini için üretildi.\n\n"
        "Lisans: CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Fotoğrafların listesi ve her birinin yazar/lisans sayfası: credits.json\n"
        "MegaScenes: Tung, Chou, Cai, Yang, Zhang, Wetzstein, Hariharan, Snavely (ECCV 2024), https://megascenes.github.io\n")
    s = A["start"]
    info = {
        "format": "vid2splat-scene/1",
        "name": name,
        "splat": "scene.spz",
        "mode": "photos",
        "up": [0, 1, 0],
        "floorY": 0.0,
        "eyeHeight": EYE,
        "walkRadius": round(A["radius"], 2),
        "maxPitchDeg": max_pitch,
        "sky": not indoor,
        **({"floorDisc": {"radius": round(float(A["extent_m"]) * 0.9, 1), "color": "#4a443d"}} if flat else {}),
        "start": {"position": [round(float(s[0]), 3), EYE, round(float(s[2]), 3)], "forward": [0, 0, -1]},
        **({"ground": A["grid"]} if A["grid"] else {}),
        "path": np.round(A["path"], 3).tolist(),
        "stats": {"points": n, "scale": round(A["scale"], 4), "photos": len(cams),
                  "psnr": round(metrics.get("psnr", 0), 2),
                  "pathLengthM": round(float(np.linalg.norm(np.diff(A["path"], axis=0), axis=1).sum()), 1)},
        "note": note or (f"{name}, Wikimedia Commons'a yüklenmiş {len(cams)} serbest lisanslı fotoğraftan "
                         "üretildi. Fotoğrafçıların durduğu alanda yürüyebilirsin; ölçek göz hizasından tahmin edildi."),
        "credit": {
            "title": f"Wikimedia Commons'tan {len(cams)} fotoğraf (MegaScenes kamera konumlarıyla)",
            "authors": "Wikimedia Commons katkıcıları",
            "url": "https://megascenes.github.io",
            "license": "CC BY-SA 4.0",
            "files": "credits.json",
        },
    }
    (out_dir / "scene.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"yazıldı: {out_dir} ({(out_dir / 'scene.spz').stat().st_size / 1e6:.1f} MB)", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, help="varsayılan: web/scenes/<id>")
    ap.add_argument("--known-height", type=float, help="sahnedeki en yüksek noktanın yerden yüksekliği (m)")
    ap.add_argument("--keep-radius", type=float, default=0, help="merkezden bu kadar uzaktaki Gaussian'ları sil (m)")
    ap.add_argument("--note", default="")
    ap.add_argument("--indoor", action="store_true", help="iç mekân: arka plana gökyüzü koyma")
    ap.add_argument("--flat", action="store_true", help="düz zeminli iç mekân: zemin en alttaki kamera katmanından")
    ap.add_argument("--max-pitch", type=float, default=45, help="yürürken başın en fazla kalkacağı açı (derece); kubbeli iç mekânda büyük ver")
    args = ap.parse_args()
    build_scene(args.zip, args.out or ROOT / "web" / "scenes" / args.id, args.name, args.known_height,
                args.keep_radius, args.note, args.indoor, args.max_pitch, args.flat)


if __name__ == "__main__":
    main()
