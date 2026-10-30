"""UNESCO miraslarını MegaScenes fotoğraflarından sırayla eğitip siteye yayınlar (Colab için).

Her sahne bağımsız ve kaldığı yerden devam edecek şekilde işlenir:
  1. fotoğraf seçimi + bozulma giderme   (scripts/megascenes_prep.py)
  2. gsplat eğitimi, --wild              (python -m vid2splat train)
  3. SPZ + kamera/nokta özeti -> <out>/<ad>.zip
  4. hizalama, ölçek, temizlik           (scripts/prepare_megascene.py)
  5. --publish verilirse: web/scenes/<ad>/ + data/scans.json -> commit -> push
     (çakışmada en güncel main'e yeniden uygulayıp tekrar dener)

Colab bağlantısı koparsa aynı komutu yeniden çalıştırmak yeter: yayınlanmış
sahneler atlanır, yarım kalanlar baştan eğitilir.

    python scripts/megascenes_batch.py --names ayasofya,efes --out /content/miras --publish
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REPO_URL = "https://github.com/Oguzhan61-dilmac/360-vid2splat.git"
AUTHOR = ("Oğuzhan Dilmaç", "dilmacoguz33@gmail.com")

# ad -> MegaScenes sahne/bileşen, en fazla fotoğraf ve sitedeki bilgiler
SCENES = {
    "ayasofya": dict(scene=174752, comp=2, max_images=600, name="Ayasofya", whc=356,
                     site="İstanbul'un Tarihî Alanları", country="Türkiye", year=1985,
                     indoor=True, known_height=55.6, max_pitch=85),
    "efes": dict(scene=261059, comp=0, max_images=400, name="Efes", whc=1018,
                 site="Efes", country="Türkiye", year=2015),
    "tac-mahal": dict(scene=113197, comp=0, max_images=600, name="Tac Mahal", whc=252,
                      site="Tac Mahal", country="Hindistan", year=1983),
    "sagrada-familia": dict(scene=28023, comp=0, max_images=600, name="Sagrada Família", whc=320,
                            site="Antoni Gaudí'nin Eserleri", country="İspanya", year=1984),
    "pantheon": dict(scene=28027, comp=0, max_images=600, name="Pantheon", whc=91,
                     site="Roma Tarihi Merkezi", country="İtalya, Vatikan", year=1980,
                     indoor=True, known_height=43.3, max_pitch=85),
    "suleymaniye": dict(scene=174764, comp=0, max_images=400, name="Süleymaniye Camii", whc=356,
                        site="İstanbul'un Tarihî Alanları", country="Türkiye", year=1985),
    "stonehenge": dict(scene=126767, comp=0, max_images=400, name="Stonehenge", whc=373,
                       site="Stonehenge, Avebury ve İlişkili Alanlar", country="Birleşik Krallık", year=1986),
    "aziz-vasil": dict(scene=87549, comp=0, max_images=500, name="Aziz Vasil Katedrali", whc=545,
                       site="Kremlin ve Kızıl Meydan", country="Rusya", year=1990),
    "karlov-koprusu": dict(scene=13356, comp=0, max_images=600, name="Karl Köprüsü", whc=616,
                           site="Prag Tarihi Merkezi", country="Çekya", year=1992),
    "partenon": dict(scene=206978, comp=1, max_images=400, name="Partenon", whc=404,
                     site="Atina Akropolisi", country="Yunanistan", year=1987),
    "sfenks": dict(scene=131955, comp=0, max_images=400, name="Büyük Sfenks", whc=86,
                   site="Memphis ve Nekropolü: Gize'den Dahşur'a Piramit Alanları", country="Mısır", year=1979),
    "westminster": dict(scene=27749, comp=0, max_images=500, name="Westminster Abbey", whc=426,
                        site="Westminster Sarayı ve Westminster Manastırı", country="Birleşik Krallık", year=1987),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_AUTHOR_NAME=AUTHOR[0], GIT_AUTHOR_EMAIL=AUTHOR[1],
               GIT_COMMITTER_NAME=AUTHOR[0], GIT_COMMITTER_EMAIL=AUTHOR[1])
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=check, capture_output=True, text=True)


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


def train_scene(name: str, cfg: dict, out: Path, steps: int, max_splats: int, max_side: int) -> Path:
    work = out / name
    zpath = out / f"{name}.zip"
    if zpath.exists():
        return zpath
    if not (work / "scene" / "sparse" / "0").exists():
        run([sys.executable, "-u", "scripts/megascenes_prep.py", "--scene", str(cfg["scene"]), "--component", str(cfg["comp"]),
             "--out", str(work), "--max-images", str(cfg["max_images"])])
    ply = work / "splat" / "scene.ply"
    if not ply.exists():
        run([sys.executable, "-u", "-m", "vid2splat", "train", str(work), "--steps", str(steps),
             "--max-splats", str(max_splats), "--wild", "--max-side", str(max_side)])
    spz = work / "scene.spz"
    run(["npx", "-y", "@playcanvas/splat-transform@3.10.3", "-w", "-q", str(ply), "-H", "0", str(spz), "--spz-version", "3"])
    export_cameras(work / "scene", work / "cameras.json")
    tmp = zpath.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in ("scene.spz", "cameras.json", "points.npy", "credits.json"):
            z.write(work / f, f)
        z.write(work / "splat" / "metrics.json", "metrics.json")
        for p in sorted((work / "splat" / "previews").glob("*.jpg")):
            z.write(p, "previews/" + p.name)
    tmp.rename(zpath)
    shutil.rmtree(work / "raw", ignore_errors=True)
    return zpath


def scan_entry(name: str, cfg: dict, size_mb: float, photos: int) -> dict:
    return {"id": name, "whc": cfg["whc"], "name": cfg["name"], "site": cfg["site"], "country": cfg["country"],
            "year": cfg["year"], "path": f"scenes/{name}/", "v": int(time.time()), "size": f"{size_mb:.0f} MB",
            "sourceShort": f"{photos} Commons fotoğrafından, CC BY-SA"}


def publish(name: str, cfg: dict, staged: Path, info: dict, repo: Path) -> None:
    """Sahneyi en güncel main'e ekleyip push eder; başkası araya push ettiyse yeniden dener."""
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(repo)], check=True)
    size_mb = (staged / "scene.spz").stat().st_size / 1e6
    for attempt in range(6):
        git(["fetch", "-q", "--depth", "1", "origin", "main"], repo)
        git(["reset", "-q", "--hard", "origin/main"], repo)
        dest = repo / "web" / "scenes" / name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(staged, dest)
        sp = repo / "web" / "data" / "scans.json"
        data = json.loads(sp.read_text())
        scans = [s for s in data["scans"] if s.get("id") != name]
        entry = scan_entry(name, cfg, size_mb, info["stats"]["photos"])
        last_unesco = max([i for i, s in enumerate(scans) if s.get("whc")], default=-1)
        scans.insert(last_unesco + 1, entry)
        data["scans"] = scans
        sp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        git(["add", "-A", f"web/scenes/{name}", "web/data/scans.json"], repo)
        msg = (f"{cfg['name']} 3D sahnesini ekle\n\n"
               f"- Wikimedia Commons'taki {info['stats']['photos']} serbest lisanslı fotoğraftan, MegaScenes kamera konumlarıyla\n"
               f"- {info['stats']['points']:,} Gaussian, {size_mb:.0f} MB; test PSNR {info['stats']['psnr']} dB\n"
               f"- UNESCO {cfg['whc']}: {cfg['site']}; lisans CC BY-SA 4.0 (credits.json)")
        git(["commit", "-q", "-m", msg], repo)
        r = git(["push", "-q", "origin", "HEAD:main"], repo, check=False)
        if r.returncode == 0:
            log(f"[{name}] yayınlandı ({size_mb:.0f} MB)")
            return
        log(f"[{name}] push reddedildi, yeniden deneniyor: {r.stderr.strip()[:200]}")
        time.sleep(5 + 5 * attempt)
    raise RuntimeError("push 6 denemede başarısız")


def already_published(name: str) -> bool:
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/Oguzhan61-dilmac/360-vid2splat/main/web/data/scans.json"
        with urllib.request.urlopen(url + f"?t={int(time.time())}", timeout=30) as r:
            return any(s.get("id") == name for s in json.load(r)["scans"])
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    from prepare_megascene import build_scene

    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="virgülle: " + ",".join(SCENES))
    ap.add_argument("--out", type=Path, default=Path("/content/miras"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-splats", type=int, default=800_000)
    ap.add_argument("--max-side", type=int, default=1000)
    ap.add_argument("--publish", action="store_true", help="biten sahneyi GitHub'a push et")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        cfg = SCENES[name]
        t0 = time.time()
        log(f"===== {name} (MegaScenes {cfg['scene']}/{cfg['comp']}) =====")
        if args.publish and already_published(name):
            log(f"[{name}] zaten yayında, atlanıyor")
            continue
        try:
            zpath = train_scene(name, cfg, args.out, args.steps, args.max_splats, args.max_side)
            staged = args.out / "staged" / name
            shutil.rmtree(staged, ignore_errors=True)
            info = build_scene(zpath, staged, cfg["name"], cfg.get("known_height"), indoor=cfg.get("indoor", False),
                               max_pitch=cfg.get("max_pitch", 45))
            if args.publish:
                publish(name, cfg, staged, info, args.out / "repo")
            log(f"[{name}] hazır ({(time.time() - t0) / 60:.0f} dk)")
        except Exception as e:  # noqa: BLE001 - bir sahne bozulsa da diğerleri devam etsin
            log(f"[{name}] HATA: {e}")
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
