"""Structure-from-Motion: seçilen karelerden kamera konumlarını ve seyrek nokta bulutunu çıkarır.

İki mod var:

* 360°: Her panorama, ortak merkezli 12 sanal pinhole kameraya (4 yön x 3 eğim,
  90° görüş alanı) bölünür ve bu kameralar COLMAP'te sabit bir "rig" olarak
  çözülür. Kameranın hemen altındaki bölge (çekimi yapan kişi) maskelenir.
* Perspektif: Normal telefon videosu tek kamera olarak çözülür, ardından
  lens bozulması giderilir.

İki modun çıktısı da aynı yapıdadır: `images/` ve `sparse/0` içeren, sadece
PINHOLE/SIMPLE_PINHOLE kameralar kullanan bir COLMAP modeli.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from pycolmap import panorama as pano

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LAYOUTS = {
    # 4 yön x 3 eğim (-35°, 0°, +35°), komşu görüntüler örtüşür. Önerilen.
    "overlapping": pano.PanoRenderType.PERSPECTIVE_OVERLAPPING,
    # Klasik küp haritası (üst ve alt yüz olmadan), örtüşme yok.
    "cubemap": pano.PanoRenderType.PERSPECTIVE_NON_OVERLAPPING,
}


@dataclass
class SfmOptions:
    use_gpu: bool | None = None  # None: CUDA varsa otomatik kullan
    num_threads: int = -1
    seed: int = 0
    max_features: int = 4096
    overlap: int = 10  # sıralı eşleştirmede her kareyi kaç sonraki kareyle eşleştirelim

    @property
    def gpu(self) -> bool:
        return bool(pycolmap.has_cuda) if self.use_gpu is None else self.use_gpu


def _list_images(folder: Path) -> list[str]:
    return sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _extraction_options(opts: SfmOptions) -> pycolmap.FeatureExtractionOptions:
    extraction = pycolmap.FeatureExtractionOptions(
        use_gpu=opts.gpu, gpu_index="0" if opts.gpu else "-1", num_threads=opts.num_threads
    )
    extraction.sift.max_num_features = opts.max_features
    return extraction


def _matching_options(opts: SfmOptions, **kwargs) -> pycolmap.FeatureMatchingOptions:
    return pycolmap.FeatureMatchingOptions(
        use_gpu=opts.gpu, gpu_index="0" if opts.gpu else "-1", num_threads=opts.num_threads, **kwargs
    )


def _match_sequential(database: Path, opts: SfmOptions, matching: pycolmap.FeatureMatchingOptions) -> None:
    pycolmap.match_sequential(
        database,
        pairing_options=pycolmap.SequentialPairingOptions(overlap=opts.overlap),
        matching_options=matching,
    )


def _best_reconstruction(recs: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    if not recs:
        raise RuntimeError(
            "COLMAP hiçbir kamerayı konumlandıramadı. Video çok kısa, çok bulanık "
            "ya da sahnede yeterli doku yok olabilir. Daha yavaş yürüyerek tekrar çekin."
        )
    return max(recs.values(), key=lambda r: r.num_reg_images())


def _nadir_keep_mask(processor: pano.PanoProcessor, cam_idx: int, nadir_deg: float) -> np.ndarray:
    """Sanal kamera için, kameranın altındaki `nadir_deg` derecelik konide False olan maske."""
    camera = processor._camera
    rays = processor._rays_in_cam @ processor.cams_from_pano_rotation[cam_idx]
    # pycolmap.panorama ile aynı açı tanımı: +y aşağı bakar.
    elevation = -np.arctan2(rays[:, 1], np.linalg.norm(rays[:, [0, 2]], axis=1))
    keep = elevation > np.deg2rad(-(90.0 - nadir_deg))
    return keep.reshape(camera.width, camera.height).T


def reconstruct_360(
    pano_dir: str | Path,
    work_dir: str | Path,
    *,
    layout: str = "overlapping",
    nadir_mask_deg: float = 35.0,
    opts: SfmOptions | None = None,
) -> tuple[pycolmap.Reconstruction, Path, Path | None]:
    """360° panoramalardan rig tabanlı SfM.

    Dönüş: (en büyük rekonstrüksiyon, sanal görüntü klasörü, eğitim maskeleri klasörü)
    """
    opts = opts or SfmOptions()
    pano_dir, work_dir = Path(pano_dir), Path(work_dir)
    image_dir = work_dir / "images"
    feature_mask_dir = work_dir / "feature_masks"
    train_mask_dir = work_dir / "train_masks"
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    for d in (image_dir, feature_mask_dir, sparse_dir):
        d.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()

    names = _list_images(pano_dir)
    if not names:
        raise RuntimeError(f"{pano_dir} içinde panorama bulunamadı.")
    print(f"[sfm-360] {len(names)} panorama sanal kameralara bölünüyor ({layout})")
    processor = pano.render_perspective_images(
        names, pano_dir, image_dir, feature_mask_dir, pano.PANO_RENDER_OPTIONS[LAYOUTS[layout]], show_progress=True
    )
    rig_config = processor.rig_config

    # Çekimi yapan kişiyi maskele: hem öznitelik çıkarımında hem eğitimde kullanılır.
    train_masks: Path | None = None
    if nadir_mask_deg > 0:
        train_mask_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        for cam_idx, rig_camera in enumerate(rig_config.cameras):
            keep = _nadir_keep_mask(processor, cam_idx, nadir_mask_deg)
            prefix = rig_camera.image_prefix.rstrip("/")
            if keep.all():
                continue
            mask_img = keep.astype(np.uint8) * 255
            cv2.imwrite(str(train_mask_dir / f"{prefix}.png"), mask_img)
            index[prefix] = f"{prefix}.png"
            for name in names:
                path = feature_mask_dir / f"{rig_camera.image_prefix}{name}.png"
                feat = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if feat is not None:
                    cv2.imwrite(str(path), np.where(keep, feat, 0).astype(np.uint8))
        (train_mask_dir / "index.json").write_text(json.dumps(index, indent=2))
        train_masks = train_mask_dir
        print(f"[sfm-360] alttaki {nadir_mask_deg:.0f}° koni maskelendi ({len(index)} sanal kamerada)")

    camera = rig_config.cameras[0].camera
    pycolmap.extract_features(
        database,
        image_dir,
        reader_options=pycolmap.ImageReaderOptions(
            mask_path=feature_mask_dir,
            camera_model=camera.model_name,
            camera_params=camera.params_to_string(),
        ),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        extraction_options=_extraction_options(opts),
    )
    with pycolmap.Database.open(database) as db:
        pycolmap.apply_rig_config([rig_config], db)

    # Sanal kameralar arası göreli pozlar kesin olarak bilindiği için rig kısıtıyla
    # doğrulama yapılır; aynı panoramanın görüntüleri maskeler sayesinde örtüşmez.
    _match_sequential(
        database, opts, _matching_options(opts, rig_verification=True, skip_image_pairs_in_same_frame=True)
    )

    mapper = pycolmap.IncrementalPipelineOptions(
        num_threads=opts.num_threads,
        random_seed=opts.seed,
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )
    recs = pycolmap.incremental_mapping(database, image_dir, sparse_dir, mapper)
    rec = _best_reconstruction(recs)
    print(f"[sfm-360] {rec.num_reg_frames()}/{len(names)} panorama konumlandırıldı, "
          f"{rec.num_points3D()} nokta, ort. yeniden izdüşüm hatası {rec.compute_mean_reprojection_error():.2f} px")
    return rec, image_dir, train_masks


def reconstruct_perspective(
    frames_dir: str | Path,
    work_dir: str | Path,
    *,
    opts: SfmOptions | None = None,
) -> tuple[pycolmap.Reconstruction, Path, None]:
    """Normal (perspektif) videodan SfM + lens bozulması giderme."""
    opts = opts or SfmOptions()
    frames_dir, work_dir = Path(frames_dir), Path(work_dir)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    undistorted = work_dir / "undistorted"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()

    names = _list_images(frames_dir)
    print(f"[sfm] {len(names)} kare işleniyor (perspektif mod)")
    pycolmap.extract_features(
        database,
        frames_dir,
        reader_options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL"),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=_extraction_options(opts),
    )
    _match_sequential(database, opts, _matching_options(opts))
    mapper = pycolmap.IncrementalPipelineOptions(num_threads=opts.num_threads, random_seed=opts.seed)
    recs = pycolmap.incremental_mapping(database, frames_dir, sparse_dir, mapper)
    rec = _best_reconstruction(recs)
    print(f"[sfm] {rec.num_reg_images()}/{len(names)} kare konumlandırıldı, {rec.num_points3D()} nokta")

    best_dir = work_dir / "sparse_best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True)
    rec.write(best_dir)
    if undistorted.exists():
        shutil.rmtree(undistorted)
    pycolmap.undistort_images(undistorted, best_dir, frames_dir)
    rec = pycolmap.Reconstruction(undistorted / "sparse")
    return rec, undistorted / "images", None


def export_scene_dir(
    rec: pycolmap.Reconstruction,
    image_dir: Path,
    train_masks: Path | None,
    scene_dir: str | Path,
) -> Path:
    """Eğitime hazır klasörü yazar: images/, sparse/0, (varsa) train_masks/."""
    scene_dir = Path(scene_dir)
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    (scene_dir / "sparse" / "0").mkdir(parents=True)
    rec.write(scene_dir / "sparse" / "0")
    # Sadece konumlandırılmış görüntüleri taşı; eğitim klasörü sade kalsın.
    for image_id in rec.reg_image_ids():
        name = rec.images[image_id].name
        dst = scene_dir / "images" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_dir / name, dst)
    if train_masks is not None:
        shutil.copytree(train_masks, scene_dir / "train_masks")
    return scene_dir
