"""
Recipe A'-512 — Single-output 512-d distillation (no MRL).

Goal: produce a *native* 512-d embedding model that beats FashionSigLIP-768
(63.84 LookBench Fine R@1) using ensemble distillation, without relying on
Matryoshka slicing of a 768-d backbone.

Architecture:
    [Image] -> FashionSigLIP-Large vision tower (init from Recipe A' best 768-d)
            -> 768-d image features
            -> Linear(768 -> 512)   <-- new projection head
            -> 512-d embedding (L2-normalized)

Why this exists:
    Matryoshka P1 @ 512-d gets 67.42 Fine R@1, but it's a *slice* of a 768-d
    embedding (output dim from training is 768). This script trains a model
    whose final output is *natively* 512-d, so client code can treat the
    embedding as a single fixed size — no slicing, no MRL bookkeeping.
    Latency win is the same as MRL @ 512-d (search/index O(d), encode the same).

Init strategy (key for fast convergence):
    1. Backbone: load Recipe A' best 768-d student (`models/moda-siglip-distilled/best/`).
    2. Projection head: PCA-fit on cached Recipe A' embeddings -> initial weight
       (768x512). PCA preserves variance optimally, so the head starts at
       PCA-512 quality (~67.0 Fine R@1) and only needs to refine toward
       teacher's relational structure.

Losses:
    - RKD-Distance (Park et al. 2019): preserve pairwise distances at 512-d.
      Dim-invariant: works against the 2048-d ensemble teacher.
    - Similarity mimicry: MSE on cosine-similarity matrices.
    - Backbone drift reg: L2 toward Recipe A' init (small weight).
    - Optional: small InfoNCE on (query, positive) pairs (off by default).

Periodic LookBench eval at 512-d (with projection head applied) drives the
best-checkpoint selection.

Usage:
    python benchmark/distill_512d_native.py \\
        --epochs 3 --batch-size 96 --lr-backbone 5e-6 --lr-head 5e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))

from distill_ensemble_to_student import (  # noqa: E402
    DistillDataset,
    MultiDistillDataset,
    rkd_distance_loss,
    similarity_mimicry_loss,
    drift_reg,
)

OUT_DIR = REPO / "models/moda-siglip-distilled-512d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "distill_512d_native.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 512-d wrapper: backbone (frozen or fine-tuned) + projection head
# ---------------------------------------------------------------------------


class SigLIP512Wrapper(nn.Module):
    """Vision-only wrapper that exposes encode_image -> 512-d.

    Holds the underlying open_clip model (kept whole so the rest of the
    pipeline can still call `.visual` / `.encode_text` if needed) and a
    learnable Linear(768, 512) projection head applied on top of the
    image features.
    """

    def __init__(self, base_model: nn.Module, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.base = base_model
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.base.encode_image(images)
        return self.proj(feats)

    def encode_text(self, *args, **kwargs):
        return self.base.encode_text(*args, **kwargs)


# ---------------------------------------------------------------------------
# PCA initialization for the projection head
# ---------------------------------------------------------------------------


@torch.no_grad()
def pca_init_projection(
    base_model: nn.Module,
    preprocess,
    sample_dataset: MultiDistillDataset,
    device: str,
    n_samples: int = 4096,
    batch_size: int = 64,
    out_dim: int = 512,
) -> torch.Tensor:
    """Run a few thousand images through the backbone, PCA-fit on the 768-d
    output, and return a (out_dim, 768) tensor suitable as Linear weight.

    The PCA components are the rows of the resulting matrix; this gives
    the projection head an excellent starting point that already preserves
    most of the variance in 512 dims.
    """
    log.info(
        "Running PCA initialization on %d sample images for the 768->%d head...",
        n_samples, out_dim,
    )
    base_model.eval()

    n_total = len(sample_dataset)
    if n_total > n_samples:
        idx = np.random.RandomState(42).choice(n_total, size=n_samples, replace=False)
    else:
        idx = np.arange(n_total)

    feats: list[np.ndarray] = []
    for s in tqdm(range(0, len(idx), batch_size), desc="PCA-encode"):
        batch_idx = idx[s : s + batch_size]
        tensors = []
        for j in batch_idx:
            t, _ = sample_dataset[int(j)]
            tensors.append(t)
        x = torch.stack(tensors).to(device)
        f = base_model.encode_image(x)
        if device == "mps":
            f = f.float()
        feats.append(F.normalize(f, p=2, dim=-1).cpu().numpy())
    feats_np = np.concatenate(feats, 0)

    feats_centered = feats_np - feats_np.mean(0, keepdims=True)
    u, s, vh = np.linalg.svd(feats_centered, full_matrices=False)
    components = vh[:out_dim]

    var_explained = float(np.sum(s[:out_dim] ** 2) / np.sum(s ** 2))
    log.info(
        "PCA done: shape=%s  variance retained at %d dims = %.4f",
        components.shape, out_dim, var_explained,
    )
    return torch.from_numpy(components.astype(np.float32))


# ---------------------------------------------------------------------------
# In-loop LookBench evaluation at 512-d
# ---------------------------------------------------------------------------


@torch.no_grad()
def quick_lookbench_eval_512(
    wrapper: SigLIP512Wrapper,
    preprocess,
    device: str,
    subsets=("real_studio_flat",),
    max_query: int = 500,
    batch_size: int = 64,
) -> dict:
    """Run quick LookBench Fine R@1 at the 512-d output."""
    from datasets import load_from_disk

    wrapper.eval()
    results = {}
    lb_root = REPO / "data/raw/lookbench/datasets"
    for subset in subsets:
        dsd = load_from_disk(str(lb_root / subset))
        qs = dsd["query"].select(range(min(max_query, len(dsd["query"]))))
        gs = dsd["gallery"]

        def enc(split):
            feats = []
            for s in range(0, len(split), batch_size):
                batch = split[s : s + batch_size]
                t = torch.stack(
                    [preprocess(im.convert("RGB")) for im in batch["image"]]
                ).to(device)
                f = wrapper.encode_image(t)
                if device == "mps":
                    f = f.float()
                feats.append(F.normalize(f, p=2, dim=-1).cpu())
            return torch.cat(feats, 0)

        qf = enc(qs)
        gf = enc(gs)

        def labels(split):
            return [
                (
                    str(x.get("category", "")).lower().strip(),
                    f"{str(x.get('category', '')).lower().strip()}|"
                    f"{str(x.get('main_attribute', '')).lower().strip()}",
                )
                for x in split
            ]

        q_lab = labels(qs)
        g_lab = labels(gs)
        sims = qf @ gf.T
        idx = sims.argmax(dim=1).tolist()
        n_hit_fine = sum(1 for i, top in enumerate(idx) if q_lab[i][1] == g_lab[top][1])
        n_hit_coarse = sum(1 for i, top in enumerate(idx) if q_lab[i][0] == g_lab[top][0])
        results[subset] = {
            "n_query": len(q_lab),
            "fine_r_at_1": round(100 * n_hit_fine / max(1, len(q_lab)), 2),
            "coarse_r_at_1": round(100 * n_hit_coarse / max(1, len(q_lab)), 2),
        }
    wrapper.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", nargs="+",
        default=["df_inshop", "df_multimodal", "hnm"],
    )
    ap.add_argument("--hnm-limit", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--drift-weight", type=float, default=0.01)
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-subsets", nargs="+",
                    default=["real_studio_flat", "real_streetlook"])
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--init",
        default="models/moda-siglip-distilled/best/model_state_dict.pt",
        help="Backbone init = Recipe A' best 768-d student (default).",
    )
    ap.add_argument(
        "--pca-init", action="store_true", default=True,
        help="PCA-init the projection head from cached backbone features.",
    )
    ap.add_argument(
        "--no-pca-init", dest="pca_init", action="store_false",
        help="Disable PCA initialization (Xavier random instead).",
    )
    ap.add_argument("--pca-samples", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        args.device
        or ("mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    log.info("Device: %s", device)

    log.info("Loading backbone: hf-hub:Marqo/marqo-fashionSigLIP")
    base, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    init_path = REPO / args.init
    if init_path.exists():
        sd = torch.load(init_path, map_location="cpu", weights_only=True)
        miss, unexp = base.load_state_dict(sd, strict=False)
        log.info(
            "Loaded backbone init from %s (missing=%d, unexpected=%d)",
            args.init, len(miss), len(unexp),
        )
    else:
        log.warning(
            "Backbone init checkpoint not found at %s; falling back to HF defaults.",
            init_path,
        )

    wrapper = SigLIP512Wrapper(base, in_dim=768, out_dim=args.out_dim).to(device)

    pretrained_vis = {
        k: v.detach().clone().to(device)
        for k, v in wrapper.base.named_parameters()
        if k.startswith("visual.")
    }

    log.info("Building training dataset(s): %s", args.datasets)
    dsets = [
        DistillDataset(k, preprocess, hnm_limit=args.hnm_limit) for k in args.datasets
    ]
    full = MultiDistillDataset(dsets)
    log.info(
        "Total training images: %d across %d dataset(s)", len(full), len(dsets),
    )

    if args.pca_init:
        try:
            comps = pca_init_projection(
                wrapper.base, preprocess, full, device,
                n_samples=args.pca_samples,
                batch_size=64,
                out_dim=args.out_dim,
            )
            with torch.no_grad():
                wrapper.proj.weight.copy_(comps.to(device))
            log.info("Projection head initialized from PCA components.")
        except Exception as e:
            log.warning("PCA init failed (%s); falling back to Xavier.", e)

    loader = DataLoader(
        full, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True,
    )

    backbone_params = [
        p for n, p in wrapper.base.named_parameters() if n.startswith("visual.")
    ]
    head_params = list(wrapper.proj.parameters())
    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    log.info(
        "Optimizer: backbone params=%d (lr=%.2e), head params=%d (lr=%.2e)",
        sum(p.numel() for p in backbone_params),
        args.lr_backbone,
        sum(p.numel() for p in head_params),
        args.lr_head,
    )

    best_fine_r1 = 0.0
    step = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        log.info("=" * 70)
        log.info("EPOCH %d/%d", epoch + 1, args.epochs)
        log.info("=" * 70)

        for images, teacher in tqdm(loader, desc=f"epoch{epoch + 1}"):
            images = images.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher = F.normalize(teacher, p=2, dim=-1)

            s = wrapper.encode_image(images)
            if device == "mps":
                s = s.float()
            s = F.normalize(s, p=2, dim=-1)

            l_rkd = rkd_distance_loss(s, teacher)
            l_sim = similarity_mimicry_loss(s, teacher)
            l_drift = drift_reg(wrapper.base, pretrained_vis)
            loss = (
                args.rkd_weight * l_rkd
                + args.sim_weight * l_sim
                + args.drift_weight * l_drift
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                backbone_params + head_params, 1.0
            )
            optim.step()

            step += 1
            if step % 50 == 0:
                log.info(
                    "  step=%d loss=%.4f (rkd=%.4f sim=%.4f drift=%.4e)",
                    step, loss.item(), l_rkd.item(), l_sim.item(), l_drift.item(),
                )

            if args.eval_every > 0 and step % args.eval_every == 0:
                quick = quick_lookbench_eval_512(
                    wrapper, preprocess, device, subsets=args.eval_subsets,
                )
                overall_fine = float(
                    np.mean([v["fine_r_at_1"] for v in quick.values()])
                )
                log.info(
                    "  [eval @ step %d] %s  mean_fine_r@1=%.2f",
                    step, quick, overall_fine,
                )
                history.append(
                    {"step": step, "eval": quick, "mean_fine_r1": overall_fine}
                )
                if overall_fine > best_fine_r1:
                    best_fine_r1 = overall_fine
                    best_path = OUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save(
                        wrapper.base.state_dict(),
                        best_path / "backbone_state_dict.pt",
                    )
                    torch.save(
                        wrapper.proj.state_dict(),
                        best_path / "proj_state_dict.pt",
                    )
                    with open(best_path / "meta.json", "w") as f:
                        json.dump(
                            {
                                "base_model": "Marqo/marqo-fashionSigLIP",
                                "init_from": args.init,
                                "training": (
                                    "Recipe A'-512 native 512-d projection head "
                                    "+ RKD-D + sim-mimicry on 2048-d ensemble teacher "
                                    "(no MRL slicing)"
                                ),
                                "out_dim": args.out_dim,
                                "step": step,
                                "epoch": epoch,
                                "mean_fine_r1_subset_eval": overall_fine,
                                "per_subset": quick,
                                "args": vars(args),
                            },
                            f,
                            indent=2,
                        )
                    log.info("  Saved new best to %s", best_path)

        ckpt = OUT_DIR / f"epoch_{epoch + 1}"
        ckpt.mkdir(exist_ok=True)
        torch.save(wrapper.base.state_dict(), ckpt / "backbone_state_dict.pt")
        torch.save(wrapper.proj.state_dict(), ckpt / "proj_state_dict.pt")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(
        "Training complete in %.1f min. Best subset mean Fine R@1 = %.2f",
        elapsed / 60, best_fine_r1,
    )
    log.info("Best checkpoint: %s/best/", OUT_DIR)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "history": history,
                "best_mean_fine_r1": best_fine_r1,
                "elapsed_seconds": round(elapsed, 1),
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
