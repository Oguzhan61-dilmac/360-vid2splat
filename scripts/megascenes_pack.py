"""Colab'da biten sahne zip'lerini küçültür: yönlü renk (SH) bantlarını atıp yeniden paketler.

Siteye eklerken sahne zaten döndürülüp yalnızca temel renk tutulduğu için SH
bantları gereksiz; atınca zip ~3 kat küçülür ve sohbete/GitHub'a kolay yüklenir.

    python scripts/megascenes_pack.py /content/miras      # -> /content/miras/paket/<ad>.zip
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main(folder: Path) -> None:
    out = folder / "paket"
    out.mkdir(exist_ok=True)
    for z in sorted(folder.glob("*.zip")):
        dest = out / z.name
        if dest.exists():
            continue
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(z) as src:
            td = Path(td)
            src.extract("scene.spz", td)
            subprocess.run(["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q", str(td / "scene.spz"),
                            "-H", "0", str(td / "small.spz"), "--spz-version", "3"], check=True)
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as dst:
                for name in src.namelist():
                    if name == "scene.spz":
                        dst.write(td / "small.spz", "scene.spz")
                    else:
                        dst.writestr(name, src.read(name))
        print(f"{dest} ({dest.stat().st_size / 1e6:.1f} MB, önce {z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/content/miras"))
