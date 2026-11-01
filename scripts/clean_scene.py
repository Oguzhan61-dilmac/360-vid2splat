"""Yayındaki bir fotoğraf sahnesinden yürüme alanındaki "sis" lekelerini temizler.

İnternet fotoğraflarıyla eğitilen sahnelerde, fotoğrafçıların durduğu yerde büyük,
yarı saydam lekeler kalır (kareye giren insanlar, ışık farkları). Yürürken bunlar
görüşü kapatır. Bu betik scene.json'daki yürüme rotasını kullanır:
  - rota koridorunun içinde, yerden 0,2–4 m yüksekte duran büyük Gaussian'lar silinir;
  - rotaya yakın olup rotaya uzaklığına göre büyük kalanlar silinir.

    python scripts/clean_scene.py web/scenes/pantheon
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vid2splat.export import read_ply, write_ply  # noqa: E402

ST = ["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q"]


def dist_to_path(xz: np.ndarray, path: np.ndarray) -> np.ndarray:
    d = np.full(len(xz), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        t = np.clip(((xz - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        p = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(xz - p, axis=1))
    return d


def ground_at(info: dict, xz: np.ndarray) -> np.ndarray:
    g = info.get("ground")
    if not g:
        return np.full(len(xz), info.get("floorY", 0.0))
    y = np.array(g["y"]).reshape(g["h"], g["w"])
    i = np.clip(((xz[:, 0] - g["origin"][0]) / g["cell"]).round().astype(int), 0, g["w"] - 1)
    j = np.clip(((xz[:, 1] - g["origin"][1]) / g["cell"]).round().astype(int), 0, g["h"] - 1)
    return y[j, i]


def main(scene_dir: Path, corridor_extra: float = 3.0, max_size_in_corridor: float = 0.45) -> None:
    info = json.loads((scene_dir / "scene.json").read_text())
    path = np.array(info["path"], float)
    R = float(info.get("walkRadius", 3.0))
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "s.ply"
        subprocess.run(ST + [str(scene_dir / "scene.spz"), str(ply)], check=True)
        props, a = read_ply(ply)
        c = {p: i for i, p in enumerate(props)}
        xyz = a[:, [c["x"], c["y"], c["z"]]].astype(np.float64)
        size = np.exp(a[:, [c["scale_0"], c["scale_1"], c["scale_2"]]]).max(1)
        d = dist_to_path(xyz[:, [0, 2]], path)
        h = xyz[:, 1] - ground_at(info, xyz[:, [0, 2]])
        opacity = 1 / (1 + np.exp(-a[:, c["opacity"]]))
        in_corr = (d < R + corridor_extra) & (h > 0.2) & (h < 4.0) & (size > max_size_in_corridor)
        near = (d < R + 10) & (h > 0.2) & (size > 0.2 * np.maximum(d - R, 0.5)) & (size > 0.3)
        # Meydandaki kalabalığın izi: insan boyunda, büyük ve yumuşak lekeler
        crowd = (d < R + 25) & (h > 0.2) & (h < 3.5) & (size > 0.7)
        # Havada asılı toz: yürüme alanının üstünde yarı saydam küçük lekeler
        dust = (d < R + 15) & (h > 0.2) & (h < 8.0) & (opacity < 0.25)
        drop = in_corr | near | crowd | dust
        print(f"{scene_dir.name}: {len(a):,} -> {(~drop).sum():,} (koridor {in_corr.sum():,}, yakın {near.sum():,}, "
              f"kalabalık {crowd.sum():,}, toz {dust.sum():,})")
        write_ply(ply, props, a[~drop])
        subprocess.run(ST + [str(ply), str(scene_dir / "scene.spz"), "--spz-version", "3"], check=True)
    info["stats"]["points"] = int((~drop).sum())
    (scene_dir / "scene.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
