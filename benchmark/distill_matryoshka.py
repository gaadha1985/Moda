"""
P1 — Matryoshka Representation Learning (MRL) on top of the Recipe A' student.

Goal: produce a single 768-d embedding model whose leading prefixes
[64, 128, 256, 384, 512, 768] are *also* valid embeddings — i.e. one
forward pass yields multiple usable resolutions, and you can pick the
shortest one that still satisfies your accuracy/cost target.

This is a continuation of Recipe A' distillation:
  - Init from models/moda-siglip-distilled/best (our 768-d ensemble-distilled student)
  - Same 2048-d ensemble teacher cache (FashionSigLIP + FashionCLIP + MODA-DF2)
  - Same training pool (DF-InShop + DF-MultiModal, optionally + H&M)
  - NEW: per-slice RKD-D + per-slice sim-mimicry, summed (Kusupati et al. 2022 style)

After this run, eval each slice on LookBench separately to get a Pareto curve
of (dim, Fine R@1).

Usage:
    python benchmark/distill_matryoshka.py \
        --epochs 2 --batch-size 128 --lr 5e-6 \
        --slices 64 128 256 384 512 768
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))
# Reuse the dataset class + losses from the Recipe A' script
from distill_ensemble_to_student import (  # noqa: E402
    DistillDataset, MultiDistillDataset,
    rkd_distance_loss, similarity_mimicry_loss, drift_reg,
    quick_lookbench_eval,
)

OUT_DIR = REPO / "models/moda-siglip-matryoshka"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "distill_matryoshka.log")],
)
log = logging.getLogger(__name__)


def matryoshka_loss(student_full: torch.Tensor,
                    teacher: torch.Tensor,
                    slices: list[int],
                    slice_weights: list[float],
                    rkd_weight: float, sim_weight: float):
    """Compute RKD-D + sim-mimicry on each prefix [: dim] of the student.

    Each slice is L2-renormalised so cosine geometry is well-defined at
    that dim. Teacher is the full 2048-d ensemble for every slice (since
    RKD-D is dimension-invariant; we want each slice to mimic the same
    teacher's *relational* structure).

    Returns: total_loss, per_slice_dict {dim: {rkd, sim}}
    """
    assert len(slices) == len(slice_weights)
    total = torch.zeros((), device=student_full.device)
    per = {}
    for d, w in zip(slices, slice_weights):
        s_slice = F.normalize(student_full[:, :d], p=2, dim=-1)
        l_rkd = rkd_distance_loss(s_slice, teacher)
        l_sim = similarity_mimicry_loss(s_slice, teacher)
        total = total + w * (rkd_weight * l_rkd + sim_weight * l_sim)
        per[d] = {"rkd": float(l_rkd.detach()), "sim": float(l_sim.detach())}
    return total, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal"])
    ap.add_argument("--hnm-limit", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--drift-weight", type=float, default=0.01)
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[64, 128, 256, 384, 512, 768])
    ap.add_argument("--slice-weights", type=float, nargs="+", default=None,
                    help="Defaults to 1.0 per slice (uniform).")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-subsets", nargs="+",
                    default=["real_studio_flat", "real_streetlook"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--init",
                    default="models/moda-siglip-distilled/best/model_state_dict.pt",
                    help="Init from Recipe A' distilled student")
    args = ap.parse_args()

    if args.slice_weights is None:
        args.slice_weights = [1.0] * len(args.slices)
    assert len(args.slice_weights) == len(args.slices)

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("Using device: %s", device)
    log.info("Slices: %s  weights: %s", args.slices, args.slice_weights)

    log.info("Loading student arch from FashionSigLIP, init=%s", args.init)
    student, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    init_path = REPO / args.init
    if init_path.exists():
        sd = torch.load(init_path, map_location="cpu", weights_only=True)
        student.load_state_dict(sd, strict=False)
        log.info("  loaded init weights (%d keys)", len(sd))
    else:
        log.warning("  init checkpoint not found, starting from HF weights")
    student = student.to(device)

    pretrained_vis = {k: v.detach().clone().to(device)
                      for k, v in student.named_parameters()
                      if k.startswith("visual.")}

    dsets = [DistillDataset(k, preprocess, hnm_limit=args.hnm_limit)
             for k in args.datasets]
    full = MultiDistillDataset(dsets)
    log.info("Total training images across %d datasets: %d",
             len(dsets), len(full))

    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda"), drop_last=True)

    optim_params = [p for n, p in student.named_parameters()
                    if n.startswith("visual.")]
    optim = torch.optim.AdamW(optim_params, lr=args.lr,
                              weight_decay=args.weight_decay)

    best_score = 0.0
    step = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        log.info("=" * 70)
        log.info("EPOCH %d/%d", epoch + 1, args.epochs)
        log.info("=" * 70)

        for images, teacher in tqdm(loader, desc=f"mrl-ep{epoch+1}"):
            images = images.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher = F.normalize(teacher, p=2, dim=-1)

            s_full = student.encode_image(images)
            if device == "mps":
                s_full = s_full.float()

            l_mrl, per_slice = matryoshka_loss(
                s_full, teacher, args.slices, args.slice_weights,
                args.rkd_weight, args.sim_weight)
            l_drift = drift_reg(student, pretrained_vis)
            loss = l_mrl + args.drift_weight * l_drift

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
            optim.step()

            step += 1
            if step % 50 == 0:
                head = (f"  step={step} loss={loss.item():.4f} "
                        f"drift={l_drift.item():.4e}")
                detail = "  ".join(f"d{d}:rkd={p['rkd']:.4f}/sim={p['sim']:.4f}"
                                    for d, p in per_slice.items())
                log.info("%s | %s", head, detail)

            if args.eval_every > 0 and step % args.eval_every == 0:
                # Evaluate quick LookBench at every slice (uses encode_image
                # full → we just truncate before normalisation in the eval
                # helper below).
                slice_results = {}
                for d in args.slices:
                    student.eval()
                    metrics = _quick_eval_at_dim(student, preprocess, device,
                                                  d, args.eval_subsets)
                    slice_results[d] = metrics
                    log.info("  [eval @ step %d dim=%d] %s", step, d, metrics)
                student.train()
                # Track using mean of dim=768 only (full student) so we don't
                # over-prefer tiny dims.
                full_metrics = slice_results[args.slices[-1]]
                mean_full = float(np.mean([v["fine_r_at_1"]
                                            for v in full_metrics.values()]))
                history.append({
                    "step": step,
                    "slice_eval": {d: m for d, m in slice_results.items()},
                    "mean_fine_r1_full": mean_full,
                })
                if mean_full > best_score:
                    best_score = mean_full
                    best_path = OUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save(student.state_dict(),
                               best_path / "model_state_dict.pt")
                    with open(best_path / "meta.json", "w") as f:
                        json.dump({
                            "base_model": "Marqo/marqo-fashionSigLIP",
                            "init_from": args.init,
                            "training": "Matryoshka MRL distill on ensemble teacher",
                            "slices": args.slices,
                            "slice_weights": args.slice_weights,
                            "step": step,
                            "epoch": epoch,
                            "best_score_full_dim": mean_full,
                            "per_slice_eval": slice_results,
                        }, f, indent=2)
                    log.info("  *** new best: dim=%d mean_fine_r1=%.2f ***",
                             args.slices[-1], mean_full)

        ckpt = OUT_DIR / f"epoch_{epoch + 1}"
        ckpt.mkdir(exist_ok=True)
        torch.save(student.state_dict(), ckpt / "model_state_dict.pt")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("Training complete in %.1f min. Best mean Fine R@1 (full dim) = %.2f",
             elapsed / 60, best_score)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "best_mean_fine_r1_full_dim": best_score,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    return 0


@torch.no_grad()
def _quick_eval_at_dim(model, preprocess, device, dim, subsets, max_query=500,
                        batch_size=64):
    """LookBench fine/coarse R@1 using the prefix [:dim] of student embeddings."""
    from datasets import load_from_disk
    results = {}
    lb_root = REPO / "data/raw/lookbench/datasets"
    for subset in subsets:
        dsd = load_from_disk(str(lb_root / subset))
        qs = dsd["query"].select(range(min(max_query, len(dsd["query"]))))
        gs = dsd["gallery"]

        def enc(split):
            feats = []
            for s in range(0, len(split), batch_size):
                batch = split[s:s + batch_size]
                t = torch.stack([preprocess(im.convert("RGB"))
                                 for im in batch["image"]]).to(device)
                f = model.encode_image(t)
                if device == "mps":
                    f = f.float()
                f = f[:, :dim]
                feats.append(F.normalize(f, p=2, dim=-1).cpu())
            return torch.cat(feats, 0)

        qf, gf = enc(qs), enc(gs)
        sims = qf @ gf.T
        idx = sims.argmax(dim=1).tolist()

        def lab(split):
            return [(str(x.get("category", "")).lower().strip(),
                     f"{str(x.get('category', '')).lower().strip()}|"
                     f"{str(x.get('main_attribute', '')).lower().strip()}")
                    for x in split]

        ql, gl = lab(qs), lab(gs)
        n_fine = sum(1 for i, top in enumerate(idx) if ql[i][1] == gl[top][1])
        n_coarse = sum(1 for i, top in enumerate(idx) if ql[i][0] == gl[top][0])
        results[subset] = {
            "n_query": len(ql),
            "fine_r_at_1": round(100 * n_fine / max(1, len(ql)), 2),
            "coarse_r_at_1": round(100 * n_coarse / max(1, len(ql)), 2),
        }
    return results


if __name__ == "__main__":
    sys.exit(main())
