"""Birden çok UNESCO mirasını MegaScenes fotoğraflarından sırayla eğitir (Colab için).

Her sahne için: fotoğraf seçimi + bozulma giderme -> gsplat eğitimi (--wild) ->
SPZ -> <out>/<ad>.zip (scene.spz, cameras.json, credits.json, metrics.json, önizlemeler).
Zip'ler sonra scripts/prepare_megascene.py ile hizalanıp siteye eklenir.

    python scripts/megascenes_batch.py --names ayasofya,efes --out /content/miras
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ad: (MegaScenes sahne no, bileşen, en fazla fotoğraf)
SCENES = {
    "ayasofya": (174752, 2, 600),
    "efes": (261059, 0, 400),
    "tac-mahal": (113197, 0, 600),
    "sagrada-familia": (28023, 0, 600),
    "pantheon": (28027, 0, 600),
    "aziz-vasil": (87549, 0, 500),
    "stonehenge": (126767, 0, 400),
    "suleymaniye": (174764, 0, 400),
    "karlov-koprusu": (13356, 0, 600),
    "partenon": (206978, 1, 400),
    "sfenks": (131955, 0, 400),
    "westminster": (27749, 0, 500),
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def export_cameras(scene_dir: Path, dest: Path) -> None:
    import pycolmap

    rec = pycolmap.Reconstruction(scene_dir / "sparse" / "0")
    cams = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        cams.append({
            "name": im.name.split("/")[-1],
            "center": np.asarray(im.projection_center()).round(5).tolist(),
            "R": R.round(6).tolist(),  # dünyadan kameraya (OpenCV: x sağ, y aşağı, z ileri)
            "f": round(float(cam.params[0]), 2), "w": cam.width, "h": cam.height,
        })
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    if len(pts) > 300_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 300_000, replace=False)]
    dest.write_text(json.dumps({"cameras": cams}))
    np.save(dest.with_name("points.npy"), pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="virgülle: " + ",".join(SCENES))
    ap.add_argument("--out", type=Path, default=Path("/content/miras"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-splats", type=int, default=800_000)
    ap.add_argument("--max-side", type=int, default=1000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        scene_id, comp, max_images = SCENES[name]
        work = args.out / name
        t0 = time.time()
        print(f"\n===== {name} (MegaScenes {scene_id}/{comp}) =====", flush=True)
        try:
            if not (work / "scene" / "sparse" / "0").exists():
                run([sys.executable, "scripts/megascenes_prep.py", "--scene", str(scene_id), "--component", str(comp),
                     "--out", str(work), "--max-images", str(max_images)])
            run([sys.executable, "-m", "vid2splat", "train", str(work), "--steps", str(args.steps),
                 "--max-splats", str(args.max_splats), "--wild", "--max-side", str(args.max_side)])
            ply = work / "splat" / "scene.ply"
            spz = work / "scene.spz"
            run(["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q", str(ply), str(spz), "--spz-version", "3"])
            export_cameras(work / "scene", work / "cameras.json")
            zpath = args.out / f"{name}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
                for f in ("scene.spz", "cameras.json", "points.npy", "credits.json"):
                    z.write(work / f, f)
                z.write(work / "splat" / "metrics.json", "metrics.json")
                for p in sorted((work / "splat" / "previews").glob("*.jpg")):
                    z.write(p, "previews/" + p.name)
            print(f"[{name}] hazır: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, {(time.time() - t0) / 60:.0f} dk)", flush=True)
            shutil.rmtree(work / "raw", ignore_errors=True)
        except Exception as e:  # noqa: BLE001 - bir sahne bozulsa da diğerleri devam etsin
            print(f"[{name}] HATA: {e}", flush=True)


if __name__ == "__main__":
    main()
