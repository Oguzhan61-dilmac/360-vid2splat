"""Test için sentetik video üretir: dokulu bir oda içinde yürüyen 360° (veya normal) kamera.

Gerçek çekim olmadan işlem hattını uçtan uca doğrulamak için kullanılır. Sahne
Y-yukarı bir dünyadadır, zemin y=0'dır ve kameranın gerçek konumları
`<çıktı>.json` dosyasına yazılır. 360 modunda kameranın altında, onunla birlikte
hareket eden bir "çekimi yapan kişi" (baş ve omuzlar) de vardır.

Kullanım:
    python tests/make_synthetic.py out/synthetic_360.mp4 --mode 360
    python tests/make_synthetic.py out/synthetic_phone.mp4 --mode perspective
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOM_LO = np.array([-4.0, 0.0, -6.0])
ROOM_HI = np.array([4.0, 3.0, 6.0])
BOXES = [  # mobilyalar: (alt köşe, üst köşe)
    (np.array([-3.5, 0.0, -1.0]), np.array([-2.3, 0.8, 0.6])),
    (np.array([2.2, 0.0, 2.0]), np.array([3.6, 1.2, 3.2])),
    (np.array([1.8, 0.0, -4.5]), np.array([2.6, 2.0, -3.7])),
    (np.array([-1.2, 0.0, 4.6]), np.array([0.4, 0.5, 5.6])),
]
PX_PER_M = 160.0
ATLAS = 4096


def make_atlas(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((ATLAS, ATLAS, 3), np.float32)
    for cells, weight in [(16, 0.5), (64, 1.0), (256, 1.0), (1024, 0.8)]:
        small = rng.random((cells, cells, 3), dtype=np.float32)
        acc += weight * cv2.resize(small, (ATLAS, ATLAS), interpolation=cv2.INTER_CUBIC)
    acc -= acc.min()
    acc /= acc.max()
    return (np.clip(acc * 1.2 - 0.1, 0, 1) * 255).astype(np.uint8)


def _face_uv(points: np.ndarray, axis: np.ndarray, face_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Çarpışma noktasının yüzey koordinatlarını atlas piksellerine çevirir."""
    other = {0: (2, 1), 1: (0, 2), 2: (0, 1)}
    u = np.zeros(len(points), np.float32)
    v = np.zeros(len(points), np.float32)
    for a, (i, j) in other.items():
        sel = axis == a
        u[sel] = points[sel, i]
        v[sel] = points[sel, j]
    offset_u = (face_id * 977) % ATLAS
    offset_v = (face_id * 541) % ATLAS
    return (u * PX_PER_M + offset_u) % (ATLAS - 1), (v * PX_PER_M + offset_v) % (ATLAS - 1)


def trace(origin: np.ndarray, dirs: np.ndarray, atlas: np.ndarray, operator: bool) -> np.ndarray:
    n = len(dirs)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        # Oda içinden çıkış
        t_far = np.where(dirs > 0, (ROOM_HI - origin) * inv, (ROOM_LO - origin) * inv)
        t_far = np.where(np.isfinite(t_far), t_far, np.inf)
        axis = np.argmin(t_far, axis=1)
        t_hit = t_far[np.arange(n), axis]
        sign = np.sign(dirs[np.arange(n), axis])
        face = axis * 2 + (sign > 0)
        # Mobilyalar
        for k, (lo, hi) in enumerate(BOXES):
            t1 = (lo - origin) * inv
            t2 = (hi - origin) * inv
            tmin_axes = np.minimum(t1, t2)
            tmin = np.nanmax(tmin_axes, axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmin > 1e-4) & (tmin < t_hit)
            ax = np.nanargmax(tmin_axes, axis=1)
            t_hit = np.where(hit, tmin, t_hit)
            axis = np.where(hit, ax, axis)
            face = np.where(hit, 10 + k * 6 + ax * 2 + (dirs[np.arange(n), ax] > 0), face)
    points = origin + dirs * t_hit[:, None]
    u, v = _face_uv(points, axis, face)
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    u1, v1 = np.minimum(u0 + 1, ATLAS - 1), np.minimum(v0 + 1, ATLAS - 1)
    a = atlas  # float32 atlas
    colors = (a[v0, u0] * (1 - fu) * (1 - fv) + a[v0, u1] * fu * (1 - fv)
              + a[v1, u0] * (1 - fu) * fv + a[v1, u1] * fu * fv)
    shade = np.array([0.80, 0.80, 0.65, 1.0, 0.90, 0.90])[np.minimum(axis * 2 + (face % 2), 5)]
    colors *= shade[:, None]

    if operator:
        # Çekimi yapan kişi: kameranın 0.35 m altında baş, 0.6 m altında omuzlar.
        for depth, radius, color in [(0.35, 0.13, (60, 50, 45)), (0.6, 0.26, (40, 60, 120))]:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -depth / dirs[:, 1]
            p = dirs * t[:, None]
            hit = (dirs[:, 1] < 0) & (np.hypot(p[:, 0], p[:, 2]) < radius) & (t < t_hit)
            stripes = 0.75 + 0.25 * np.sin(p[:, 0] * 90) * np.cos(p[:, 2] * 70)
            colors[hit] = np.array(color, np.float32) * stripes[hit, None]
            t_hit = np.where(hit, t, t_hit)
    return np.clip(colors, 0, 255).astype(np.uint8)


def camera_path(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Odanın içinde kıvrımlı bir yürüyüş rotası ve yürüme yönleri."""
    s = np.linspace(0, 1, n)
    z = -4.0 + 8.0 * s
    x = 1.4 * np.sin(s * 2 * np.pi * 0.9)
    y = 1.9 + 0.025 * np.sin(s * 2 * np.pi * 14)  # yürürken hafif inip kalkma
    pos = np.stack([x, y, z], 1)
    vel = np.gradient(pos, axis=0)
    heading = np.arctan2(vel[:, 0], vel[:, 2])
    return pos, heading


def rotation_for(heading: float, tilt_x: float, tilt_z: float) -> np.ndarray:
    """Kamera eksenleri (x sağ, y aşağı, z ileri) -> dünya dönüşümü."""
    fwd = np.array([np.sin(heading), 0.0, np.cos(heading)])
    down = np.array([0.0, -1.0, 0.0])
    right = np.cross(down, fwd)
    M = np.stack([right, down, fwd], 1)
    cx, sx, cz, sz = np.cos(tilt_x), np.sin(tilt_x), np.cos(tilt_z), np.sin(tilt_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return M @ Rx @ Rz


def equirect_dirs(w: int, h: int) -> np.ndarray:
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    yaw = (2 * uu - 1) * np.pi
    pitch = (1 - 2 * vv) * np.pi / 2
    return np.stack([np.cos(pitch) * np.sin(yaw), -np.sin(pitch), np.cos(pitch) * np.cos(yaw)], -1).reshape(-1, 3)


def pinhole_dirs(w: int, h: int, hfov_deg: float) -> np.ndarray:
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    x = (np.arange(w) + 0.5 - w / 2) / f
    y = (np.arange(h) + 0.5 - h / 2) / f
    xx, yy = np.meshgrid(x, y)
    d = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--mode", choices=["360", "perspective"], default="360")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    is360 = args.mode == "360"
    w = args.width
    h = w // 2 if is360 else int(w * 9 / 16)
    n = int(args.seconds * args.fps)
    atlas = make_atlas().astype(np.float32)
    local_dirs = equirect_dirs(w, h) if is360 else pinhole_dirs(w, h, 75.0)
    pos, heading = camera_path(n)
    rng = np.random.default_rng(3)
    tilts = np.cumsum(rng.normal(0, 0.003, (n, 2)), 0)
    tilts -= tilts.mean(0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(args.output)],
        stdin=subprocess.PIPE,
    )
    gt = []
    for i in range(n):
        R = rotation_for(heading[i], tilts[i, 0], tilts[i, 1])
        dirs = local_dirs @ R.T
        img = trace(pos[i], dirs, atlas, operator=is360).reshape(h, w, 3)
        if i % 7 == 3:  # ara sıra sarsıntı bulanıklığı
            k = np.zeros((1, 15), np.float32)
            k[0, :] = 1 / 15
            img = cv2.filter2D(img, -1, k)
        proc.stdin.write(img[:, :, ::-1].tobytes())
        gt.append({"frame": i, "position": pos[i].tolist(), "rotation": R.tolist()})
    proc.stdin.close()
    proc.wait()
    meta = {"mode": args.mode, "fps": args.fps, "capture_height": 1.9, "frames": gt}
    args.output.with_suffix(".json").write_text(json.dumps(meta))
    print(f"{args.output} yazıldı: {n} kare, {w}x{h}")


if __name__ == "__main__":
    main()
