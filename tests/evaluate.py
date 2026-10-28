"""Sentetik videonun gerçek kamera konumlarıyla SfM + hizalama sonucunu karşılaştırır.

Kullanım:
    python tests/evaluate.py <sentetik_video.json> <çıktı_klasörü>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s * R @ src + t için en küçük kareler benzerlik dönüşümü."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def evaluate(gt_json: Path, out_dir: Path) -> dict:
    gt = json.loads(gt_json.read_text())
    frames = json.loads((out_dir / "frames.json").read_text())["frames"]
    rec = pycolmap.Reconstruction(out_dir / "scene" / "sparse" / "0")

    # Panorama/kare adı -> kayıtlı kamera merkezi (360'ta 12 sanal kamera aynı merkezi paylaşır)
    centers: dict[str, np.ndarray] = {}
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        pose = image.cam_from_world()
        R = np.asarray(pose.rotation.matrix())
        centers.setdefault(Path(image.name).name, -R.T @ np.asarray(pose.translation).reshape(3))

    est, ref = [], []
    for f in frames:
        if f["file"] in centers:
            est.append(centers[f["file"]])
            ref.append(gt["frames"][f["frame_index"]]["position"])
    est, ref = np.array(est), np.array(ref)
    s, R, t = umeyama(est, ref)
    err = np.linalg.norm((s * est @ R.T + t) - ref, axis=1)
    # Hizalama doğruysa dünya "yukarı"sı değişmeden kalmalı (R sadece Y etrafında dönmeli).
    up_err = np.degrees(np.arccos(np.clip((R @ np.array([0, 1, 0]))[1], -1, 1)))
    result = {
        "registered": f"{len(est)}/{len(frames)}",
        "scale_ratio": round(s, 4),  # 1.0 = ölçek ve zemin doğru tahmin edildi
        "up_error_deg": round(float(up_err), 3),
        "position_rmse_m": round(float(np.sqrt((err**2).mean())), 4),
        "camera_height_m": round(float(np.median(est[:, 1])), 3),
        "true_camera_height_m": round(float(np.median(ref[:, 1])), 3),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
