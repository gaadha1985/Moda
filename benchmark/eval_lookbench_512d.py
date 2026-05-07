"""
Full LookBench evaluation for the Recipe A'-512 native 512-d distilled model.

Loads:
  - FashionSigLIP-Large vision tower with the Recipe A'-512 backbone weights.
  - Linear(768 -> 512) projection head.
  - Wraps `encode_image` so it returns 512-d (projection applied + L2 norm
    handled by the standard eval pipeline).

Evaluates against the full LookBench protocol (all 4 subsets + 58K-image noise
gallery) using the same metrics as `eval_lookbench_baseline.py`:
  - Fine Recall@K   (category|main_attribute composite match)
  - Coarse Recall@K (category match)
  - ID Recall@K     (item_ID match)
  - nDCG@5          (binary cat+main relevance)
  - MRR             (item_ID-based)

Usage:
    python benchmark/eval_lookbench_512d.py \\
        --backbone models/moda-siglip-distilled-512d/best/backbone_state_dict.pt \\
        --proj models/moda-siglip-distilled-512d/best/proj_state_dict.pt \\
        --output distilled_512d_eval.json
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
from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))

from eval_lookbench_baseline import (  # noqa: E402
    SUBSETS,
    NOISE_LABEL,
    NOISE_CAT,
    extract_labels,
    compute_all_metrics,
    compute_mrr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

RESULTS_DIR = REPO / "results" / "lookbench"


class SigLIP512Wrapper(nn.Module):
    """Same as in distill_512d_native.py — duplicated here for eval-only use."""

    def __init__(self, base_model: nn.Module, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.base = base_model
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.base.encode_image(images)
        return self.proj(feats)


def load_512d_model(backbone_ckpt: Path, proj_ckpt: Path, device: str):
    log.info("Loading FashionSigLIP backbone arch...")
    base, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    sd = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)
    miss, unexp = base.load_state_dict(sd, strict=False)
    log.info(
        "Loaded backbone state dict from %s (missing=%d, unexpected=%d)",
        backbone_ckpt, len(miss), len(unexp),
    )

    wrapper = SigLIP512Wrapper(base, in_dim=768, out_dim=512)
    head_sd = torch.load(proj_ckpt, map_location="cpu", weights_only=True)
    wrapper.proj.load_state_dict(head_sd)
    log.info("Loaded projection head from %s", proj_ckpt)

    wrapper = wrapper.to(device).eval()
    return wrapper, preprocess


@torch.no_grad()
def extract_image_features(
    data, wrapper, preprocess, device: str, batch_size: int = 32, desc: str = "encode",
) -> tuple[np.ndarray, list]:
    all_feats = []
    all_labels = []
    for start in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[start : start + batch_size]
        images = batch["image"]
        labels = batch.get(
            "item_ID",
            batch.get("item_id", list(range(start, start + len(images)))),
        )
        tensors = torch.stack(
            [preprocess(img.convert("RGB")) for img in images]
        ).to(device)
        feats = wrapper.encode_image(tensors)
        if device == "mps":
            feats = feats.float()
        feats = F.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu())
        if isinstance(labels, (list, np.ndarray)):
            all_labels.extend(labels)
        else:
            all_labels.extend(
                labels.tolist() if hasattr(labels, "tolist") else [labels]
            )
    features = torch.cat(all_feats, dim=0).numpy()
    return features, all_labels


def evaluate_subset(
    wrapper, preprocess, subset_name: str,
    noise_feats, noise_ids, noise_labels, noise_cats,
    device: str, batch_size: int,
) -> dict:
    log.info("=" * 70)
    log.info("Subset: %s", subset_name)
    log.info("=" * 70)
    ds = load_dataset("srpone/look-bench", subset_name)
    qd = ds["query"]
    gd = ds["gallery"]
    log.info(
        "  queries=%d  subset_gallery=%d  +noise=%d  total=%d",
        len(qd), len(gd), len(noise_ids), len(gd) + len(noise_ids),
    )

    q_cats, _, q_labels = extract_labels(qd)
    g_cats_subset, _, g_labels_subset = extract_labels(gd)

    log.info("  Encoding queries...")
    q_feats, q_ids = extract_image_features(
        qd, wrapper, preprocess, device, batch_size, f"{subset_name} q",
    )
    log.info("  Encoding subset gallery...")
    g_feats_subset, g_ids_subset = extract_image_features(
        gd, wrapper, preprocess, device, batch_size, f"{subset_name} g",
    )

    g_feats = np.concatenate([g_feats_subset, noise_feats], axis=0)
    g_ids = g_ids_subset + noise_ids
    g_labels = g_labels_subset + noise_labels
    g_cats = g_cats_subset + noise_cats

    metrics = compute_all_metrics(
        q_feats, q_labels, q_cats, q_ids,
        g_feats, g_labels, g_cats, g_ids,
    )
    mrr = compute_mrr(q_feats, q_ids, g_feats, g_ids)
    log.info(
        "  %s  Fine_R@1=%.2f  Coarse_R@1=%.2f  ID_R@1=%.2f  nDCG@5=%.2f  MRR=%.2f",
        subset_name,
        metrics["fine_recall"]["recall@1"],
        metrics["coarse_recall"]["recall@1"],
        metrics["id_recall"]["recall@1"],
        metrics["ndcg@5"],
        mrr,
    )
    return {
        "subset": subset_name,
        "n_queries": len(qd),
        "n_gallery_subset": len(gd),
        "n_gallery_noise": len(noise_ids),
        "n_gallery_total": len(g_ids),
        **metrics,
        "mrr": mrr,
    }


def compute_overall(per_subset: dict) -> dict:
    weights = {
        "real_studio_flat": 1011,
        "aigen_studio": 193,
        "real_streetlook": 981,
        "aigen_streetlook": 160,
    }
    total_w = sum(weights[s] for s in per_subset)
    fine = sum(per_subset[s]["fine_recall"]["recall@1"] * weights[s] for s in per_subset)
    coarse = sum(per_subset[s]["coarse_recall"]["recall@1"] * weights[s] for s in per_subset)
    idr = sum(per_subset[s]["id_recall"]["recall@1"] * weights[s] for s in per_subset)
    ndcg = sum(per_subset[s]["ndcg@5"] * weights[s] for s in per_subset)
    return {
        "fine_recall@1": round(fine / total_w, 2),
        "coarse_recall@1": round(coarse / total_w, 2),
        "id_recall@1": round(idr / total_w, 2),
        "ndcg@5": round(ndcg / total_w, 2),
        "total_queries": total_w,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        default="models/moda-siglip-distilled-512d/best/backbone_state_dict.pt",
    )
    ap.add_argument(
        "--proj",
        default="models/moda-siglip-distilled-512d/best/proj_state_dict.pt",
    )
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="distilled_512d_eval.json")
    args = ap.parse_args()

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    backbone = REPO / args.backbone
    proj = REPO / args.proj
    if not backbone.exists():
        log.error("Backbone checkpoint not found: %s", backbone)
        return 1
    if not proj.exists():
        log.error("Projection head checkpoint not found: %s", proj)
        return 1

    wrapper, preprocess = load_512d_model(backbone, proj, device)

    log.info("Encoding shared noise gallery (58K distractors)...")
    noise_ds = load_dataset("srpone/look-bench", "noise")
    t0 = time.time()
    noise_feats, noise_ids = extract_image_features(
        noise_ds["gallery"], wrapper, preprocess, device, args.batch_size, "noise",
    )
    noise_labels = [NOISE_LABEL] * len(noise_ids)
    noise_cats = [NOISE_CAT] * len(noise_ids)
    log.info(
        "Noise encoded: %d items in %.1fs", len(noise_ids), time.time() - t0,
    )

    per_subset = {}
    for s in args.subsets:
        per_subset[s] = evaluate_subset(
            wrapper, preprocess, s,
            noise_feats, noise_ids, noise_labels, noise_cats,
            device, args.batch_size,
        )

    overall = compute_overall(per_subset)
    log.info("=" * 70)
    log.info(
        "OVERALL  Fine_R@1=%.2f  Coarse_R@1=%.2f  nDCG@5=%.2f  ID_R@1=%.2f  (n=%d)",
        overall["fine_recall@1"],
        overall["coarse_recall@1"],
        overall["ndcg@5"],
        overall["id_recall@1"],
        overall["total_queries"],
    )
    log.info("=" * 70)

    out = {
        "model": "MODA-SigLIP-Distilled-512d (Recipe A'-512, native projection head, no MRL)",
        "out_dim": 512,
        "device": device,
        "backbone_ckpt": str(backbone),
        "proj_ckpt": str(proj),
        "per_subset": per_subset,
        "overall": overall,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved results to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
