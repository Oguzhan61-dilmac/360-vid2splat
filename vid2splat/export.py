"""Eğitilmiş sahneyi web görüntüleyici için hazırlar.

1. Temizlik (Python): şeffaf, aşırı büyük ve yalnız kalmış "hayalet" Gaussian'lar
   silinir. Bunlar gezinirken havada asılı duran lekeler olarak görünür.
2. Sıkıştırma (Node.js): PlayCanvas'ın splat-transform aracıyla PLY, ~10 kat
   küçük SPZ formatına çevrilir.

Sonuç `web/` klasörüne scene.spz + scene.json olarak yazılır; bu klasör doğrudan
görüntüleyicinin `scenes/` dizinine kopyalanabilir.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.10.3"


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        data = f.read()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Sadece binary little-endian PLY destekleniyor.")
    count = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
    props = [h.split()[-1] for h in header if h.startswith("property")]
    if any(h.split()[1] != "float" for h in header if h.startswith("property")):
        raise ValueError("Sadece float özellikli PLY destekleniyor.")
    arr = np.frombuffer(data, dtype=np.float32, count=count * len(props)).reshape(count, len(props))
    return props, arr


def write_ply(path: Path, props: list[str], arr: np.ndarray) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def clean_splats(
    ply_in: Path,
    ply_out: Path,
    *,
    min_opacity: float = 0.02,
    max_scale_m: float = 2.0,
    knn: int = 8,
    outlier_factor: float = 8.0,
) -> dict:
    """Hayalet Gaussian'ları siler. Sahne align.py ile metreye ölçeklendiği için eşikler metredir."""
    from scipy.spatial import cKDTree

    props, arr = read_ply(ply_in)
    col = {p: i for i, p in enumerate(props)}
    xyz = arr[:, [col["x"], col["y"], col["z"]]]
    opacity = 1.0 / (1.0 + np.exp(-arr[:, col["opacity"]]))
    scale = np.exp(arr[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1)

    keep = np.isfinite(arr).all(1)
    stats = {"total": int(len(arr)), "nan": int((~keep).sum())}
    low = opacity < min_opacity
    big = scale > max_scale_m
    stats["transparent"] = int((low & keep).sum())
    stats["oversized"] = int((big & keep & ~low).sum())
    keep &= ~low & ~big

    idx = np.nonzero(keep)[0]
    if len(idx) > knn + 1:
        dist, _ = cKDTree(xyz[idx]).query(xyz[idx], k=knn + 1)
        mean_d = dist[:, 1:].mean(1)
        isolated = mean_d > outlier_factor * np.median(mean_d)
        stats["isolated"] = int(isolated.sum())
        keep[idx[isolated]] = False
    stats["kept"] = int(keep.sum())
    write_ply(ply_out, props, arr[keep])
    removed = stats["total"] - stats["kept"]
    print(f"[temizlik] {removed:,} / {stats['total']:,} Gaussian silindi "
          f"(şeffaf {stats['transparent']}, dev {stats['oversized']}, yalnız {stats.get('isolated', 0)})")
    return stats


def to_spz(ply: Path, out_file: Path) -> Path:
    if shutil.which("npx") is None:
        raise RuntimeError("Node.js (npx) bulunamadı: https://nodejs.org adresinden kurun.")
    # SPZ v3 (gzip) yazıyoruz: Spark 2.3 v4'ü okuyamıyor.
    subprocess.run(
        ["npx", "-y", SPLAT_TRANSFORM, "-w", "-q", str(ply), "--filter-nan", str(out_file), "--spz-version", "3"],
        check=True,
    )
    return out_file


def export_web(out: str | Path, *, remove_floaters: bool = True, name: str | None = None) -> Path:
    out = Path(out)
    ply = out / "splat" / "scene.ply"
    if not ply.exists():
        raise FileNotFoundError(f"{ply} yok; önce `python -m vid2splat train {out}` çalıştırın.")
    web = out / "web"
    web.mkdir(parents=True, exist_ok=True)

    info = json.loads((out / "scene" / "scene.json").read_text())
    source = ply
    if remove_floaters:
        source = out / "splat" / "scene_clean.ply"
        info["cleanup"] = clean_splats(ply, source)
    spz = to_spz(source, web / "scene.spz")

    info["splat"] = "scene.spz"
    info["name"] = name or out.name
    metrics_file = out / "splat" / "metrics.json"
    if metrics_file.exists():
        info["metrics"] = json.loads(metrics_file.read_text())
    (web / "scene.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[dışa aktarma] {spz} ({spz.stat().st_size / 1e6:.1f} MB) ve scene.json hazır -> {web}")
    return web
