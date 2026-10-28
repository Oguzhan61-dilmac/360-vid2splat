"""gsplat ile 3D Gaussian Splatting eğitimi (MCMC yoğunlaştırma).

gsplat'in örnek eğiticisi çok sayıda ek bağımlılık istediği için burada sadece
kütüphanenin kararlı API'sini (rasterization, MCMCStrategy, export_splats)
kullanan sade bir eğitici var. MCMC stratejisi Gaussian sayısını `max_splats`
ile sınırlar; bu hem ücretsiz Colab GPU'sunun belleğine hem de web'de
yüklenecek dosya boyutuna sığmak için önemli.

360 modunda çekimi yapan kişinin bulunduğu bölge (train_masks/) kayıp
hesabından çıkarılır, böylece sahnede hayalet bir figür oluşmaz.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SH_C0 = 0.28209479177387814


@dataclass
class TrainConfig:
    out: Path
    steps: int = 15_000
    max_splats: int = 1_000_000
    sh_degree: int = 3
    test_every: int = 10
    ssim_lambda: float = 0.2
    init_opacity: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    log_every: int = 250
    num_workers: int = 2
    seed: int = 0
    device: str = "cuda"


class SceneDataset(Dataset):
    """`scene/` klasöründeki COLMAP modelini ve görüntüleri okur (sadece pinhole)."""

    def __init__(self, scene_dir: Path, split: str, test_every: int):
        rec = pycolmap.Reconstruction(scene_dir / "sparse" / "0")
        items = []
        for image_id in rec.reg_image_ids():
            image = rec.images[image_id]
            camera = rec.cameras[image.camera_id]
            if camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
                raise ValueError(f"Sadece pinhole kameralar destekleniyor, bulunan: {camera.model_name}")
            w2c = np.eye(4)
            w2c[:3, :4] = np.asarray(image.cam_from_world().matrix())
            items.append({
                "name": image.name,
                "K": np.asarray(camera.calibration_matrix(), dtype=np.float64),
                "viewmat": w2c,
                "size": (camera.width, camera.height),
            })
        items.sort(key=lambda it: it["name"])
        if split == "train":
            self.items = [it for i, it in enumerate(items) if test_every <= 0 or i % test_every != 0]
        else:
            self.items = [it for i, it in enumerate(items) if test_every > 0 and i % test_every == 0]
        self.image_dir = scene_dir / "images"

        self.masks: dict[str, np.ndarray] = {}
        index_file = scene_dir / "train_masks" / "index.json"
        if index_file.exists():
            for prefix, file in json.loads(index_file.read_text()).items():
                m = cv2.imread(str(scene_dir / "train_masks" / file), cv2.IMREAD_GRAYSCALE)
                self.masks[prefix] = m > 127

        self.points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
        self.colors = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
        centers = np.array([np.linalg.inv(it["viewmat"])[:3, 3] for it in items])
        self.scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1) or 1.0

    def __len__(self) -> int:
        return len(self.items)

    def mask_for(self, name: str) -> np.ndarray | None:
        prefix = name.split("/")[0] if "/" in name else ""
        return self.masks.get(prefix)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        bgr = cv2.imread(str(self.image_dir / it["name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.image_dir / it["name"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        data = {
            "image": torch.from_numpy(rgb),
            "K": torch.from_numpy(it["K"]).float(),
            "viewmat": torch.from_numpy(it["viewmat"]).float(),
        }
        mask = self.mask_for(it["name"])
        if mask is not None:
            data["mask"] = torch.from_numpy(mask)
        return data


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device="cpu") -> torch.Tensor:
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum()).float()
    return (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """x, y: [B, H, W, 3], 0..1 aralığında."""
    x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    c = x.shape[1]
    mu_x = F.conv2d(x, window, groups=c)
    mu_y = F.conv2d(y, window, groups=c)
    sxx = F.conv2d(x * x, window, groups=c) - mu_x**2
    syy = F.conv2d(y * y, window, groups=c) - mu_y**2
    sxy = F.conv2d(x * y, window, groups=c) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return ssim_map.mean()


def _knn_scales(points: np.ndarray) -> np.ndarray:
    """Her noktanın en yakın 3 komşusuna ortalama uzaklığı (başlangıç boyutu için)."""
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(points).query(points, k=4)
    return np.sqrt((dist[:, 1:] ** 2).mean(1)).clip(min=1e-7)


def init_splats(data: SceneDataset, cfg: TrainConfig, device: str):
    points, colors = data.points, data.colors
    if len(points) > cfg.max_splats:
        keep = np.random.default_rng(cfg.seed).choice(len(points), cfg.max_splats, replace=False)
        points, colors = points[keep], colors[keep]
    n = len(points)
    scales = np.log(_knn_scales(points) * cfg.init_scale)[:, None].repeat(3, 1)
    sh = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = (torch.from_numpy(colors) - 0.5) / SH_C0
    params = {
        "means": (torch.from_numpy(points), 1.6e-4 * data.scene_scale),
        "scales": (torch.from_numpy(scales).float(), 5e-3),
        "quats": (torch.rand((n, 4)), 1e-3),
        "opacities": (torch.logit(torch.full((n,), cfg.init_opacity)), 5e-2),
        "sh0": (sh[:, :1, :], 2.5e-3),
        "shN": (sh[:, 1:, :], 2.5e-3 / 20),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float()) for k, (v, _) in params.items()}).to(device)
    optimizers = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
        for k, (_, lr) in params.items()
    }
    return splats, optimizers


def render(splats, viewmat, K, width, height, sh_degree):
    from gsplat import rasterization

    colors, alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
    )
    return colors[..., :3].clamp(0, 1), alphas, info


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def train(cfg: TrainConfig) -> Path:
    from gsplat import export_splats
    from gsplat.strategy import MCMCStrategy

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU bulunamadı. Colab'da: Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU."
        )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    scene_dir = Path(cfg.out) / "scene"
    out_dir = Path(cfg.out) / "splat"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainset = SceneDataset(scene_dir, "train", cfg.test_every)
    valset = SceneDataset(scene_dir, "val", cfg.test_every)
    splats, optimizers = init_splats(trainset, cfg, device)
    print(f"[eğitim] {len(trainset)} eğitim / {len(valset)} test görüntüsü, "
          f"{len(splats['means'])} başlangıç noktası, sahne ölçeği {trainset.scene_scale:.2f}")

    f = cfg.steps / 30_000  # gsplat varsayılanları 30k adım için ayarlı
    strategy = MCMCStrategy(
        cap_max=cfg.max_splats,
        refine_start_iter=max(1, int(500 * f)),
        refine_stop_iter=max(2, int(25_000 * f)),
        refine_every=max(1, int(100 * f)),
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state()
    sh_interval = max(1, int(1000 * f))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps))
    window = _gaussian_window(device=device)

    loader = DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0, pin_memory=device == "cuda",
    )
    it = iter(loader)
    tic = time.time()
    for step in range(cfg.steps):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        pixels = data["image"].to(device).float() / 255.0  # [1, H, W, 3]
        height, width = pixels.shape[1:3]
        sh_degree = min(step // sh_interval, cfg.sh_degree)
        colors, _, info = render(
            splats, data["viewmat"].to(device), data["K"].to(device), width, height, sh_degree
        )

        if "mask" in data:
            m = data["mask"].to(device)[..., None].float()
            l1 = ((colors - pixels).abs() * m).sum() / (m.sum() * 3).clamp(min=1)
            ssim_val = ssim(colors * m, pixels * m, window)
        else:
            l1 = F.l1_loss(colors, pixels)
            ssim_val = ssim(colors, pixels, window)
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim_val)
        loss = loss + cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        loss = loss + cfg.scale_reg * torch.exp(splats["scales"]).mean()
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(
            params=splats, optimizers=optimizers, state=state, step=step, info=info,
            lr=scheduler.get_last_lr()[0],
        )

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            elapsed = time.time() - tic
            eta = elapsed / (step + 1) * (cfg.steps - step - 1)
            progress = {
                "step": step + 1, "steps": cfg.steps, "loss": round(loss.item(), 5),
                "num_splats": len(splats["means"]), "elapsed_sec": round(elapsed), "eta_sec": round(eta),
            }
            _write_json(out_dir / "progress.json", progress)
            print(f"[eğitim] {step + 1}/{cfg.steps} kayıp={loss.item():.4f} "
                  f"gaussian={len(splats['means']):,} kalan≈{eta / 60:.1f} dk")

    metrics = evaluate(splats, valset, cfg.sh_degree, out_dir, device, window)
    metrics.update({"steps": cfg.steps, "num_splats": len(splats["means"]),
                    "train_minutes": round((time.time() - tic) / 60, 1)})
    _write_json(out_dir / "metrics.json", metrics)
    print(f"[eğitim] test PSNR {metrics.get('psnr', float('nan')):.2f} dB, SSIM {metrics.get('ssim', float('nan')):.3f}")

    ply = out_dir / "scene.ply"
    with torch.no_grad():
        export_splats(
            means=splats["means"], scales=splats["scales"], quats=splats["quats"],
            opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
            format="ply", save_to=str(ply),
        )
    print(f"[eğitim] {ply} yazıldı ({ply.stat().st_size / 1e6:.0f} MB)")
    return ply


@torch.no_grad()
def evaluate(splats, valset: SceneDataset, sh_degree: int, out_dir: Path, device: str, window) -> dict:
    """Eğitimde görülmemiş görüntülerde PSNR/SSIM ve birkaç karşılaştırma görseli."""
    if len(valset) == 0:
        return {}
    psnrs, ssims = [], []
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    preview_ids = set(np.linspace(0, len(valset) - 1, min(4, len(valset))).astype(int).tolist())
    for i in range(len(valset)):
        data = valset[i]
        pixels = data["image"].to(device).float()[None] / 255.0
        height, width = pixels.shape[1:3]
        colors, _, _ = render(splats, data["viewmat"].to(device)[None], data["K"].to(device)[None],
                              width, height, sh_degree)
        if "mask" in data:
            m = data["mask"].to(device)[None, ..., None].float()
            mse = (((colors - pixels) ** 2) * m).sum() / (m.sum() * 3).clamp(min=1)
            s = ssim(colors * m, pixels * m, window)
        else:
            mse = F.mse_loss(colors, pixels)
            s = ssim(colors, pixels, window)
        psnrs.append(float(-10 * math.log10(max(mse.item(), 1e-10))))
        ssims.append(float(s.item()))
        if i in preview_ids:
            side = torch.cat([pixels[0], colors[0]], 1).cpu().numpy()
            cv2.imwrite(str(preview_dir / f"val_{i:03d}.jpg"),
                        cv2.cvtColor((side * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "num_test_images": len(valset)}
