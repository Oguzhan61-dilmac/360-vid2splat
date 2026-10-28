"""Uçtan uca işlem hattı: video -> kareler -> SfM -> hizalama -> (eğitim) -> web dosyaları.

Klasör yapısı:

    <çıktı>/
      frames/      seçilen kareler (360 panorama veya normal kare)
      frames.json  kare seçim raporu
      work/        COLMAP ara dosyaları
      scene/       eğitime hazır veri: images/, sparse/0, train_masks/, scene.json
      splat/       eğitim çıktısı: scene.ply, metrics.json
      web/         görüntüleyiciye kopyalanacak: scene.spz + scene.json
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap

from . import align, frames, sfm


@dataclass
class PipelineConfig:
    video: Path
    out: Path
    mode: str = "auto"  # auto | 360 | perspective
    capture_height: float | None = None  # metre; None: moda göre varsayılan
    interval: float = 0.5
    every: int = 1  # kare klasörü girdisinde her kaçıncı görüntü alınsın
    max_frames: int = 200
    pano_width: int = 3840  # 360 karelerin en fazla genişliği (sanal görüntüler bunun 1/4'ü)
    frame_width: int = 1600  # normal videoda en fazla kare genişliği
    layout: str = "overlapping"
    nadir_mask: float = 35.0
    walk_radius: float = 1.5
    sfm: sfm.SfmOptions = field(default_factory=sfm.SfmOptions)

    def resolved_mode(self, info: frames.VideoInfo) -> str:
        if self.mode != "auto":
            return self.mode
        return "360" if info.is_equirectangular else "perspective"


def _timer(label: str, timings: dict):
    class _T:
        def __enter__(self):
            self.t = time.time()

        def __exit__(self, *exc):
            timings[label] = round(time.time() - self.t, 1)

    return _T()


def run_reconstruction(cfg: PipelineConfig) -> Path:
    """Kare seçimi + SfM + hizalama. Eğitime hazır `scene/` klasörünü döndürür."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    info = frames.probe_input(cfg.video)
    mode = cfg.resolved_mode(info)
    capture_height = cfg.capture_height or (1.7 if mode == "360" else 1.5)
    print(f"[hat] mod: {mode} | çekim yüksekliği varsayımı: {capture_height} m")

    frames_dir = cfg.out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    max_width = cfg.pano_width if mode == "360" else cfg.frame_width
    with _timer("frames", timings):
        if Path(cfg.video).is_dir():
            frames.select_images(cfg.video, frames_dir, every=cfg.every, max_frames=cfg.max_frames, max_width=max_width)
        else:
            frames.select_frames(
                cfg.video, frames_dir, interval_sec=cfg.interval, max_frames=cfg.max_frames, max_width=max_width
            )

    work = cfg.out / "work"
    if work.exists():
        shutil.rmtree(work)
    with _timer("sfm", timings):
        if mode == "360":
            rec, image_dir, masks = sfm.reconstruct_360(
                frames_dir, work, layout=cfg.layout, nadir_mask_deg=cfg.nadir_mask, opts=cfg.sfm
            )
        else:
            rec, image_dir, masks = sfm.reconstruct_perspective(frames_dir, work, opts=cfg.sfm)

    with _timer("align", timings):
        scene_info = align.align_reconstruction(
            rec,
            capture_height_m=capture_height,
            walk_radius_m=cfg.walk_radius,
            heading="motion" if mode == "360" else "camera",
        )
    scene_dir = sfm.export_scene_dir(rec, image_dir, masks, cfg.out / "scene")
    scene_info["mode"] = mode
    scene_info["timings"] = timings
    align.write_scene_json(scene_info, scene_dir / "scene.json")
    print(f"[hat] eğitime hazır sahne: {scene_dir} ({timings})")
    return scene_dir


def load_scene_info(scene_dir: Path) -> dict:
    return json.loads((scene_dir / "scene.json").read_text())


def summarize(scene_dir: Path) -> str:
    rec = pycolmap.Reconstruction(scene_dir / "sparse" / "0")
    return rec.summary()
