"""Sahneyi yerçekimine hizalar, zemini bulur ve yaklaşık metrik ölçeğe getirir.

SfM çıktısının yönü ve ölçeği keyfidir. Web'de "yürüyerek" gezinebilmek için:

1. Kameraların ortalama "yukarı" yönü +Y eksenine çevrilir (three.js Y-yukarı).
2. İlk karenin bakış yönü -Z olacak şekilde sahne kendi etrafında döndürülür.
3. Kameraların altındaki noktalardan zemin yüksekliği tahmin edilir.
4. Kamera-zemin mesafesi, kullanıcının verdiği çekim yüksekliğine (metre)
   eşitlenerek ölçek bulunur. Zemin y=0'a, çekim rotasının merkezi orijine taşınır.

Çıktı olarak web görüntüleyicinin kullandığı `scene.json` yazılır.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap

WORLD_UP = np.array([0.0, 1.0, 0.0])


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a birim vektörünü b birim vektörüne çeviren dönüşüm matrisi."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        # 180°: a'ya dik herhangi bir eksen etrafında döndür.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _yaw_rotation(forward: np.ndarray) -> np.ndarray:
    """Y ekseni etrafında, yatay `forward` vektörünü -Z'ye çeviren dönüşüm."""
    f = np.array([forward[0], 0.0, forward[2]])
    if np.linalg.norm(f) < 1e-6:
        return np.eye(3)
    return _rotation_between(f, np.array([0.0, 0.0, -1.0]))


def _camera_poses(rec: pycolmap.Reconstruction):
    """(isim, merkez, ileri, yukarı, frame_id) listesi, isme göre sıralı."""
    poses = []
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        cam_from_world = image.cam_from_world()
        R = np.asarray(cam_from_world.rotation.matrix())
        t = np.asarray(cam_from_world.translation).reshape(3)
        center = -R.T @ t
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        up = R.T @ np.array([0.0, -1.0, 0.0])
        poses.append((image.name, center, forward, up, image.frame_id))
    poses.sort(key=lambda p: (Path(p[0]).name, p[0]))
    return poses


def _frame_centers(poses) -> np.ndarray:
    """Her çekim anı için tek bir merkez (360 modunda 12 sanal kamera aynı merkezi paylaşır)."""
    seen: dict[int, np.ndarray] = {}
    for _, center, _, _, frame_id in poses:
        seen.setdefault(frame_id, center)
    order = []
    used = set()
    for _, _, _, _, frame_id in poses:
        if frame_id not in used:
            used.add(frame_id)
            order.append(seen[frame_id])
    return np.array(order)


def _estimate_floor(cam_y: np.ndarray, pts_y: np.ndarray) -> tuple[float, bool]:
    """Kameraların altındaki noktaların yükseklik dağılımından zemini bulur."""
    med_cam = float(np.median(cam_y))
    below = pts_y[pts_y < med_cam]
    if below.size < 50:
        return med_cam - 1.0, False
    lo = float(np.percentile(below, 1.0))
    if med_cam - lo < 1e-6:
        return med_cam - 1.0, False
    counts, edges = np.histogram(below, bins=64, range=(lo, med_cam))
    # Masa üstü gibi yatay yüzeyler de yoğunlaşma yaratır; yeterince dolu
    # olan en alçak seviyeyi zemin kabul ediyoruz.
    threshold = 0.5 * counts.max()
    idx = int(np.argmax(counts >= threshold))
    bin_w = edges[1] - edges[0]
    center = 0.5 * (edges[idx] + edges[idx + 1])
    near = below[np.abs(below - center) <= 1.5 * bin_w]
    floor = float(np.median(near)) if near.size else center
    return floor, True


def align_reconstruction(
    rec: pycolmap.Reconstruction,
    *,
    capture_height_m: float,
    walk_radius_m: float = 1.5,
    heading: str = "camera",  # "camera" | "motion"
) -> dict:
    """`rec`i yerinde dönüştürür ve görüntüleyici için sahne bilgisini döndürür."""
    poses = _camera_poses(rec)
    if len(poses) < 2:
        raise RuntimeError("Hizalama için en az 2 konumlandırılmış kamera gerekli.")

    up = np.mean([p[3] for p in poses], axis=0)
    R_up = _rotation_between(up, WORLD_UP)
    centers = _frame_centers(poses)

    # Başlangıç yönü: 360'ta kameranın tutuluş yönü keyfi olduğu için yürüme yönü,
    # telefonda ise kameranın baktığı yön.
    start_forward = poses[0][2]
    if heading == "motion" and len(centers) > 1:
        dists = np.linalg.norm(centers - centers[0], axis=1)
        ahead = np.nonzero(dists > 0.1 * dists.max())[0]
        if len(ahead):
            start_forward = centers[ahead[0]] - centers[0]
    R = _yaw_rotation(R_up @ start_forward) @ R_up

    centers = centers @ R.T
    points = np.array([p.xyz for p in rec.points3D.values()]) @ R.T if rec.num_points3D() else np.zeros((0, 3))

    floor_y, floor_found = _estimate_floor(centers[:, 1], points[:, 1] if len(points) else np.zeros(0))
    cam_height = float(np.median(centers[:, 1])) - floor_y
    scale = capture_height_m / cam_height if cam_height > 1e-9 else 1.0
    offset = np.array([centers[:, 0].mean(), floor_y, centers[:, 2].mean()])
    translation = -scale * offset

    sim = pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), translation.reshape(3, 1))
    rec.transform(sim)

    # Dönüşüm sonrası değerler
    poses = _camera_poses(rec)
    centers = _frame_centers(poses)
    points = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((0, 3))
    fwd = R @ start_forward
    fwd = np.array([fwd[0], 0.0, fwd[2]])
    fwd /= max(np.linalg.norm(fwd), 1e-9)

    eye = float(np.clip(capture_height_m, 1.2, 1.8))
    step = max(1, len(centers) // 300)
    path = centers[::step][:, [0, 2]]
    if len(points):
        lo = np.percentile(points, 2, axis=0)
        hi = np.percentile(points, 98, axis=0)
    else:
        lo, hi = centers.min(0) - 1, centers.max(0) + 1

    info = {
        "format": "vid2splat-scene/1",
        "up": [0.0, 1.0, 0.0],
        "floorY": 0.0,
        "floorDetected": floor_found,
        "captureHeight": capture_height_m,
        "eyeHeight": eye,
        "walkRadius": walk_radius_m,
        "start": {
            "position": [float(centers[0, 0]), eye, float(centers[0, 2])],
            "forward": fwd.round(6).tolist(),
        },
        "path": np.round(path, 4).tolist(),
        "bounds": {"min": np.round(lo, 3).tolist(), "max": np.round(hi, 3).tolist()},
        "stats": {
            "frames": int(len(centers)),
            "images": int(rec.num_reg_images()),
            "points": int(rec.num_points3D()),
            "scale": float(scale),
            "pathLengthM": float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0,
        },
    }
    status = "bulundu" if floor_found else "BULUNAMADI (varsayılan kullanıldı)"
    print(f"[hizalama] zemin {status}; ölçek x{scale:.3f}; rota {info['stats']['pathLengthM']:.1f} m, "
          f"{len(centers)} çekim noktası")
    return info


def write_scene_json(info: dict, path: str | Path, splat_file: str | None = None) -> None:
    data = dict(info)
    if splat_file:
        data["splat"] = splat_file
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
