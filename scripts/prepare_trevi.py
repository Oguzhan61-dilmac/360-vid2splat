"""Trevi Çeşmesi taramasını sitede yürünebilir hale getirir.

Kaynak: WildGaussians (Kulhanek ve ark., 2024), Photo Tourism "trevi-fountain"
sahnesi, MIT lisansı. https://huggingface.co/jkulhanek/wild-gaussians

Yaptıkları:
1. Sahneyi fotoğrafçıların kamera yönlerine göre yerçekimine hizalar (Y yukarı),
   çeşme başlangıçta -Z yönünde, yani karşıda kalır.
2. Sahneyi çeşmenin bilinen yüksekliğine (26,3 m) göre metreye ölçekler; göz
   hizasını fotoğrafların çekildiği ortanca yüksekliğe koyar.
3. Şeffaf ve çok uzaktaki "hayalet" Gaussian'ları siler.
4. web/scenes/trevi/scene.spz ve scene.json yazar. Yürüme alanı, fotoğrafların
   çekildiği noktaları birleştiren bir koridordur.

Kullanım:
    python scripts/prepare_trevi.py <point_cloud.ply> <cameras.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vid2splat.align import _rotation_between, _yaw_rotation  # noqa: E402
from vid2splat.export import read_ply, write_ply  # noqa: E402

EYE_HEIGHT_M = 1.6
FOUNTAIN_HEIGHT_M = 26.3  # Trevi Çeşmesi'nin yüksekliği
OUT_DIR = ROOT / "web" / "scenes" / "trevi"


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(w, x, y, z) dörtlüleri: a ⊗ b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], 1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main(ply_path: Path, cams_path: Path) -> None:
    props, arr = read_ply(ply_path)
    col = {p: i for i, p in enumerate(props)}
    xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    opacity = 1 / (1 + np.exp(-arr[:, col["opacity"]]))

    cams = np.array([c["poses"] for c in json.load(open(cams_path))])  # kameradan dünyaya, OpenCV
    centers, down, fwd = cams[:, :, 3], cams[:, :, 1], cams[:, :, 2]

    # 1) Hizalama: kameraların "yukarı"sı +Y, ortalama bakış yönü -Z
    up = -down.mean(0)
    R_up = _rotation_between(up, np.array([0.0, 1.0, 0.0]))
    R = _yaw_rotation(R_up @ fwd.mean(0)) @ R_up
    p = xyz @ R.T
    c = centers @ R.T

    # 2) Ölçek ve zemin. Fotoğrafçıların altında görünen yüzey çoğu zaman havuzun
    #    dibi olduğu için zemin tahmini güvenilmez; ölçeği çeşmenin bilinen
    #    yüksekliğinden (26,3 m) alıyoruz. Göz hizası fotoğrafların ortanca
    #    yüksekliğine, zemin onun 1,6 m altına konur.
    solid = opacity > 0.5
    hist, edges = np.histogram(p[solid, 2], bins=200, range=(c[:, 2].mean() - 30, c[:, 2].mean() + 5))
    facade_z = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    band = p[solid & (np.abs(p[:, 2] - facade_z) < 1.2) & (np.abs(p[:, 0] - c[:, 0].mean()) < 9)]
    facade_h = float(np.percentile(band[:, 1], 99) - np.percentile(band[:, 1], 1))
    scale = FOUNTAIN_HEIGHT_M / facade_h
    eye_y = float(np.median(c[:, 1]))
    floor = eye_y - EYE_HEIGHT_M / scale
    print(f"cephe {abs(facade_z - c[:, 2].mean()):.2f} birim ileride, yüksekliği {facade_h:.2f} birim -> ölçek x{scale:.3f} "
          f"(fotoğraf mesafesi ~{abs(facade_z - c[:, 2].mean()) * scale:.0f} m)")

    # Metre uzayı: zemin y=0, birim = metre. Konumlar ve Gaussian boyutları
    # AYNI katsayıyla ölçeklenmeli; yoksa her Gaussian olduğundan büyük görünür
    # ve sahne bulanıklaşır.
    shift = np.array([0.0, floor, 0.0])
    p = (p - shift) * scale
    c = (c - shift) * scale
    facade_z = (facade_z - 0.0) * scale

    # 3) Temizlik
    sizes = np.exp(arr[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1) * scale
    cz0, cz1 = c[:, 2].min(), c[:, 2].max()
    cx0, cx1 = c[:, 0].min(), c[:, 0].max()
    # a) Fotoğrafçıların arkası: hiçbir fotoğraf o yöne bakmadığı için oradaki
    #    Gaussian'lar rastgele lekelerdir.
    behind = p[:, 2] > cz1 + 2.5
    # b) Fotoğrafçıların durduğu alanda zeminden 0,25 m ile 3 m arasında kalanlar:
    #    çoğu kez kareye giren turistler ve kameraya çok yakın sis.
    crowd = ((p[:, 2] > cz0 - 3.0) & (p[:, 2] < cz1 + 2.5) & (p[:, 0] > cx0 - 4) & (p[:, 0] < cx1 + 4)
             & (p[:, 1] > 0.25) & (p[:, 1] < 3.0))
    # c) Kameraya yakın büyük ve yarı saydam lekeler (yaklaşınca ekranı kaplayan bulanıklık)
    d_cam = np.full(len(p), np.inf)
    for cc in c:
        d_cam = np.minimum(d_cam, np.hypot(p[:, 0] - cc[0], p[:, 2] - cc[2]))
    near_blob = (d_cam < 6.0) & (sizes > 0.25 * np.maximum(d_cam, 0.5))
    dist = np.linalg.norm(p - np.array([c[:, 0].mean(), 10.0, facade_z]), axis=1)
    keep = ((opacity > 0.05) & (sizes < 4.0) & (dist < 120) & np.isfinite(p).all(1)
            & ~behind & ~crowd & ~near_blob)
    print(f"temizlik: {len(arr):,} -> {keep.sum():,} Gaussian "
          f"(arka {behind.sum():,}, kalabalık {crowd.sum():,}, yakın leke {near_blob.sum():,})")

    out = arr[keep].copy()
    out[:, col["x"]], out[:, col["y"]], out[:, col["z"]] = p[keep].T
    for k in ("scale_0", "scale_1", "scale_2"):
        out[:, col[k]] += np.log(scale)
    q = out[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]] = quat_mul(matrix_to_quat(R), q)
    for k in ("nx", "ny", "nz"):
        if k in col:
            out[:, col[k]] = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "_aligned.ply"
    write_ply(tmp, props, out)
    subprocess.run(["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q", str(tmp),
                    str(OUT_DIR / "scene.spz"), "--spz-version", "3"], check=True)
    tmp.unlink()

    # 4) Yürüme koridoru: fotoğraf noktaları soldan sağa
    order = np.argsort(c[:, 0])
    path = c[order][:, [0, 2]]
    start = c[np.argmin(np.abs(c[:, 0]))]
    info = {
        "format": "vid2splat-scene/1",
        "name": "Trevi Çeşmesi",
        "splat": "scene.spz",
        "mode": "photos",
        "up": [0, 1, 0],
        "floorY": 0.0,
        "eyeHeight": EYE_HEIGHT_M,
        "walkRadius": 3.0,
        "start": {"position": [round(float(start[0]), 3), EYE_HEIGHT_M, round(float(start[2]), 3)], "forward": [0, 0, -1]},
        "path": np.round(path, 3).tolist(),
        "stats": {"points": int(keep.sum()), "scale": round(scale, 4), "pathLengthM": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 1)},
        "note": "Trevi Çeşmesi, turistlerin çektiği binlerce fotoğraftan üretildi (WildGaussians, MIT lisansı). Fotoğrafların çekildiği alanda yürüyebilirsin; ölçek çeşmenin 26,3 m yüksekliğinden hesaplandı.",
        "credit": {
            "title": "WildGaussians: 3D Gaussian Splatting in the Wild",
            "authors": "Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, Torsten Sattler",
            "url": "https://huggingface.co/jkulhanek/wild-gaussians",
            "license": "MIT",
        },
    }
    (OUT_DIR / "scene.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    size = (OUT_DIR / "scene.spz").stat().st_size / 1e6
    print(f"yazıldı: {OUT_DIR}/scene.spz ({size:.1f} MB), koridor {info['stats']['pathLengthM']} m")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
