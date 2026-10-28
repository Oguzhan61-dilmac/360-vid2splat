"""Videodan rekonstrüksiyon için en net kareleri seçer.

Sabit bir bulanıklık eşiği sahneden sahneye değiştiği için video zaman
dilimlerine bölünür ve her dilimden Laplacian varyansı en yüksek kare alınır.
Dilimin en iyisi bile genel ortancanın çok altındaysa (koşarken çekilmiş,
tamamen bulanık bir an) o dilim atlanır.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    @property
    def is_equirectangular(self) -> bool:
        """360° videolar 2:1 en-boy oranlı equirectangular formatta gelir."""
        return abs(self.width / self.height - 2.0) < 0.02


@dataclass
class SelectedFrame:
    file: str
    frame_index: int
    time_sec: float
    sharpness: float


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def probe_input(path: str | Path) -> VideoInfo:
    """Video dosyası ya da hazır kare klasörü (örneğin bir veri setinin images/ klasörü)."""
    path = Path(path)
    if not path.is_dir():
        return probe_video(path)
    images = list_images(path)
    if not images:
        raise RuntimeError(f"{path} içinde .jpg/.png bulunamadı.")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Görüntü okunamadı: {images[0]}")
    h, w = first.shape[:2]
    return VideoInfo(str(path), w, h, fps=1.0, num_frames=len(images))


def select_images(
    folder: str | Path,
    out_dir: str | Path,
    *,
    every: int = 1,
    max_frames: int = 200,
    max_width: int | None = None,
    jpeg_quality: int = 95,
) -> list[SelectedFrame]:
    """Hazır kare klasöründen her `every`. kareyi alır (en fazla `max_frames`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(folder)
    chosen = images[:: max(1, every)]
    if max_frames > 0 and len(chosen) > max_frames:
        step = len(chosen) / max_frames
        chosen = [chosen[int(i * step)] for i in range(max_frames)]
    selected: list[SelectedFrame] = []
    for idx, src in enumerate(chosen):
        img = cv2.imread(str(src))
        if img is None:
            continue
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), _resize_max_width(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        selected.append(SelectedFrame(name, idx, float(idx), sharpness_score(img, False)))
    report = {"source_folder": str(folder), "every": every, "frames": [asdict(s) for s in selected]}
    (out_dir.parent / "frames.json").write_text(json.dumps(report, indent=2))
    print(f"[kareler] {len(images)} görüntüden {len(selected)} tanesi seçildi (her {every}. kare)")
    return selected


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Videodan kare okunamadı: {path}")
    height, width = frame.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoInfo(str(path), width, height, fps, num_frames)


def sharpness_score(frame: np.ndarray, equirectangular: bool) -> float:
    """Laplacian varyansı: yüksek değer = net kare."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 960:
        scale = 960 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if equirectangular:
        # Kutuplar aşırı gerilmiş ve alt kısımda çekimi yapan kişi var;
        # netliği ufuk çizgisi etrafındaki banttan ölçüyoruz.
        h = gray.shape[0]
        gray = gray[h // 4 : 3 * h // 4]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resize_max_width(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    if not max_width or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    new_size = (max_width, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def select_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    interval_sec: float = 0.5,
    max_frames: int = 200,
    max_width: int | None = None,
    blur_ratio: float = 0.35,
    jpeg_quality: int = 95,
) -> list[SelectedFrame]:
    """Her `interval_sec` saniyelik dilimden en net kareyi `out_dir`e yazar.

    Toplam kare sayısı `max_frames`i geçecekse dilim süresi otomatik uzatılır.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    equirect = info.is_equirectangular

    interval = interval_sec
    if info.duration > 0 and max_frames > 0:
        interval = max(interval_sec, info.duration / max_frames)
    window_len = max(1, int(round(interval * info.fps)))

    cap = cv2.VideoCapture(str(video_path))
    best: tuple[float, int, np.ndarray] | None = None
    current_window = 0
    candidates: list[tuple[float, int, np.ndarray]] = []
    frame_index = 0

    def flush() -> None:
        nonlocal best
        if best is not None:
            candidates.append(best)
            best = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window = frame_index // window_len
        if window != current_window:
            flush()
            current_window = window
        score = sharpness_score(frame, equirect)
        if best is None or score > best[0]:
            best = (score, frame_index, _resize_max_width(frame, max_width))
        frame_index += 1
    flush()
    cap.release()

    if not candidates:
        raise RuntimeError("Videodan hiç kare seçilemedi.")

    # Netlik sahne içeriğine de bağlı (düz duvar az, dokulu zemin çok puan alır).
    # Bu yüzden her dilimi sadece komşu dilimlerle kıyaslıyoruz.
    scores = np.array([c[0] for c in candidates])
    selected: list[SelectedFrame] = []
    skipped = 0
    for i, (score, idx, frame) in enumerate(candidates):
        local = float(np.median(scores[max(0, i - 3) : i + 4]))
        if score < blur_ratio * local:
            skipped += 1
            continue
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        selected.append(SelectedFrame(name, idx, idx / info.fps, score))

    report = {
        "video": asdict(info),
        "equirectangular": equirect,
        "interval_sec": interval,
        "windows": len(candidates),
        "skipped_blurry_windows": skipped,
        "frames": [asdict(s) for s in selected],
    }
    (out_dir.parent / "frames.json").write_text(json.dumps(report, indent=2))
    print(
        f"[kareler] {info.width}x{info.height} @ {info.fps:.1f} fps, "
        f"{info.duration:.1f} sn -> {len(selected)} kare seçildi "
        f"(dilim {interval:.2f} sn, {skipped} bulanık dilim atlandı)"
    )
    return selected
