"""Komut satırı arayüzü.

Örnekler:
    # Her şey: kareler + SfM + eğitim + web dosyaları (GPU gerekir)
    python -m vid2splat run video.mp4 --out output/oda

    # Sadece SfM (CPU'da da çalışır)
    python -m vid2splat sfm video.mp4 --out output/oda

    # Hazır sahneyi eğit ve web için dışa aktar
    python -m vid2splat train output/oda
    python -m vid2splat export output/oda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_reconstruction
from .sfm import LAYOUTS, SfmOptions


def _add_sfm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", type=Path, help="360° (equirectangular) veya normal video dosyası ya da kare klasörü")
    p.add_argument("--out", type=Path, required=True, help="çıktı klasörü")
    p.add_argument("--mode", choices=["auto", "360", "perspective"], default="auto")
    p.add_argument("--capture-height", type=float, default=None,
                   help="kameranın yerden yüksekliği, metre (360 için varsayılan 1.7, telefon için 1.5)")
    p.add_argument("--interval", type=float, default=0.5, help="kaç saniyede bir kare seçilsin")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--every", type=int, default=1, help="kare klasörü girdisinde her kaçıncı görüntü alınsın")
    p.add_argument("--pano-width", type=int, default=3840)
    p.add_argument("--layout", choices=list(LAYOUTS), default="overlapping")
    p.add_argument("--nadir-mask", type=float, default=35.0,
                   help="360'ta kameranın altında maskelenecek koni (derece); 0 = kapalı")
    p.add_argument("--max-features", type=int, default=4096)
    p.add_argument("--cpu", action="store_true", help="GPU olsa bile COLMAP'i CPU'da çalıştır")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--steps", type=int, default=15000, help="eğitim adımı (30000 = tam kalite)")
    p.add_argument("--max-splats", type=int, default=1_000_000, help="Gaussian sayısı üst sınırı")
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--no-floaters", action="store_true", help="dışa aktarmada havada asılı Gaussian temizliğini kapat")


def _config(args) -> PipelineConfig:
    return PipelineConfig(
        video=args.video,
        out=args.out,
        mode=args.mode,
        capture_height=args.capture_height,
        interval=args.interval,
        every=args.every,
        max_frames=args.max_frames,
        pano_width=args.pano_width,
        layout=args.layout,
        nadir_mask=args.nadir_mask,
        sfm=SfmOptions(use_gpu=False if args.cpu else None, max_features=args.max_features),
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="vid2splat", description="Videodan gezilebilir 3D Gaussian Splat sahnesi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="videodan web'e hazır sahneye kadar her şey")
    _add_sfm_args(p_run)
    _add_train_args(p_run)

    p_sfm = sub.add_parser("sfm", help="kare seçimi + SfM + hizalama")
    _add_sfm_args(p_sfm)

    p_train = sub.add_parser("train", help="hazır sahneyi gsplat ile eğit")
    p_train.add_argument("out", type=Path, help="sfm komutunun çıktı klasörü")
    _add_train_args(p_train)

    p_export = sub.add_parser("export", help="eğitilmiş sahneyi web formatına çevir")
    p_export.add_argument("out", type=Path)
    p_export.add_argument("--no-floaters", action="store_true")

    args = ap.parse_args()

    if args.cmd in ("run", "sfm"):
        run_reconstruction(_config(args))
    if args.cmd in ("run", "train"):
        from .train import TrainConfig, train

        train(TrainConfig(out=args.out, steps=args.steps, max_splats=args.max_splats, sh_degree=args.sh_degree))
    if args.cmd in ("run", "train", "export"):
        from .export import export_web

        export_web(args.out, remove_floaters=not args.no_floaters)


if __name__ == "__main__":
    main()
