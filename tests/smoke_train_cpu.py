"""GPU olmadan eğitim kodunun tesisatını test eder.

gsplat'in rasterizer'ı ve MCMC adımları CUDA ister; burada bunlar basit,
türevlenebilir sahte fonksiyonlarla değiştirilir. Böylece veri okuma, maskeler,
başlatma, kayıp, optimizasyon, değerlendirme ve PLY dışa aktarma CPU'da
denenebilir. Gerçek eğitim kalitesini ölçmez.

Kullanım:
    python tests/smoke_train_cpu.py <sfm_çıktı_klasörü>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from vid2splat import train as T


def fake_render(splats, viewmat, K, width, height, sh_degree):
    # Tüm parametrelere bağlı, sabit renkli bir görüntü üretir.
    base = torch.sigmoid(splats["sh0"].mean(0).reshape(1, 1, 1, 3))
    extra = (splats["means"].mean() + splats["scales"].mean() + splats["quats"].mean()
             + splats["opacities"].mean() + splats["shN"].mean()) * 1e-3
    colors = (base + extra).expand(viewmat.shape[0], height, width, 3)
    return colors, None, {}


def main(out: Path) -> None:
    T.render = fake_render
    from gsplat.strategy import MCMCStrategy

    MCMCStrategy.step_post_backward = lambda self, **kw: None
    cfg = T.TrainConfig(out=out, steps=8, device="cpu", num_workers=0, log_every=4, max_splats=50_000)
    ply = T.train(cfg)
    metrics = json.loads((out / "splat" / "metrics.json").read_text())
    assert ply.exists() and ply.stat().st_size > 0
    assert metrics["num_test_images"] > 0
    print("smoke test OK:", ply, metrics)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
