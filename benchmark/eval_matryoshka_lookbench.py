"""
Evaluate the Matryoshka student on LookBench full-protocol at every slice
dimension {64, 128, 256, 384, 512, 768}, producing a Pareto curve of
(embedding_dim, Fine R@1).

Reuses the encoding pipeline from benchmark/eval_lookbench_baseline.py but
loads the matryoshka student once and slices the embedding before
similarity computation.

Output: results/lookbench/matryoshka_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "lookbench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SUBSETS = ["real_studio_flat", "aigen_studio",
           "real_streetlook", "aigen_streetlook"]


def device_pick(arg):
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def encode(model, preprocess, images, device, batch_size=32):
    feats = []
    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size]
        t = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        f = model.encode_image(t)
        if device == "mps":
            f = f.float()
        feats.append(f.cpu())
    return torch.cat(feats, 0)  # not normalised — we slice then normalise


def labels(split):
    return [(str(x.get("category", "")).lower().strip(),
             f"{str(x.get('category', '')).lower().strip()}|"
             f"{str(x.get('main_attribute', '')).lower().strip()}",
             x.get("item_ID"))
            for x in split]


def metrics_for(qf_full, gf_full, q_lab, g_lab, dim, ks=(1, 5, 10)):
    qf = F.normalize(qf_full[:, :dim], p=2, dim=-1)
    gf = F.normalize(gf_full[:, :dim], p=2, dim=-1)
    sims = qf @ gf.T

    # Recall@K
    out = {"fine_recall": {}, "coarse_recall": {}, "id_recall": {}}
    for k in ks:
        topk = sims.topk(k, dim=1).indices.tolist()
        nf = nc = ni = 0
        for i, idxs in enumerate(topk):
            qc, qfine, qid = q_lab[i]
            top_lab = [g_lab[j] for j in idxs]
            if any(t[1] == qfine for t in top_lab):
                nf += 1
            if any(t[0] == qc for t in top_lab):
                nc += 1
            if qid is not None and any(t[2] == qid for t in top_lab):
                ni += 1
        n = len(q_lab)
        out["fine_recall"][f"recall@{k}"] = round(100 * nf / n, 2)
        out["coarse_recall"][f"recall@{k}"] = round(100 * nc / n, 2)
        out["id_recall"][f"recall@{k}"] = round(100 * ni / n, 2)

    # nDCG@5 (binary cat|main_attribute relevance)
    k = 5
    top_idx = sims.topk(k, dim=1).indices.tolist()
    ndcg_total = 0.0
    for i, idxs in enumerate(top_idx):
        qfine = q_lab[i][1]
        rels = [1 if g_lab[j][1] == qfine else 0 for j in idxs]
        dcg = sum((2**r - 1) / np.log2(p + 2) for p, r in enumerate(rels))
        n_rel = sum(1 for x in g_lab if x[1] == qfine)
        ideal = sum(1 / np.log2(p + 2) for p in range(min(k, n_rel))) if n_rel else 1.0
        ndcg_total += (dcg / ideal) if ideal else 0.0
    out["ndcg@5"] = round(100 * ndcg_total / len(q_lab), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/moda-siglip-matryoshka/best/model_state_dict.pt")
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[64, 128, 256, 384, 512, 768])
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="matryoshka_eval.json")
    args = ap.parse_args()

    device = device_pick(args.device)
    log.info("device=%s slices=%s", device, args.slices)

    log.info("Loading matryoshka student from %s", args.checkpoint)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    sd = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    lb_root = REPO / "data/raw/lookbench/datasets"
    out = {"checkpoint": args.checkpoint, "slices": args.slices, "subsets": {}}

    # Load shared noise once
    log.info("Loading noise gallery ...")
    noise_dsd = load_from_disk(str(lb_root / "noise"))
    noise_g = noise_dsd["gallery"] if "gallery" in noise_dsd else noise_dsd
    log.info("  noise size: %d", len(noise_g))

    for subset in args.subsets:
        log.info("=" * 60)
        log.info("Subset: %s", subset)
        log.info("=" * 60)
        dsd = load_from_disk(str(lb_root / subset))
        qs, gs = dsd["query"], dsd["gallery"]
        log.info("  queries=%d  gallery=%d", len(qs), len(gs))

        t0 = time.time()
        qf_full = encode(model, preprocess, qs["image"], device, args.batch_size)
        gf_full = encode(model, preprocess, gs["image"], device, args.batch_size)
        nf_full = encode(model, preprocess, noise_g["image"], device, args.batch_size)
        gf_combined = torch.cat([gf_full, nf_full], 0)
        log.info("  encoded q+g+noise in %.1fs", time.time() - t0)

        q_lab = labels(qs)
        g_lab = labels(gs) + labels(noise_g)

        out["subsets"][subset] = {}
        for d in args.slices:
            m = metrics_for(qf_full, gf_combined, q_lab, g_lab, d)
            out["subsets"][subset][d] = m
            log.info("  dim=%d -> Fine R@1=%.2f Coarse R@1=%.2f nDCG@5=%.2f",
                     d, m["fine_recall"]["recall@1"],
                     m["coarse_recall"]["recall@1"], m["ndcg@5"])

    # Compute query-weighted overall per dim
    n_per = {s: len(load_from_disk(str(lb_root / s))["query"])
             for s in args.subsets}
    total_q = sum(n_per.values())
    out["overall_per_dim"] = {}
    for d in args.slices:
        weighted = {"fine_recall@1": 0.0, "coarse_recall@1": 0.0,
                    "ndcg@5": 0.0, "id_recall@1": 0.0}
        for s in args.subsets:
            sm = out["subsets"][s][d]
            w = n_per[s] / total_q
            weighted["fine_recall@1"] += w * sm["fine_recall"]["recall@1"]
            weighted["coarse_recall@1"] += w * sm["coarse_recall"]["recall@1"]
            weighted["ndcg@5"] += w * sm["ndcg@5"]
            weighted["id_recall@1"] += w * sm["id_recall"]["recall@1"]
        weighted = {k: round(v, 2) for k, v in weighted.items()}
        out["overall_per_dim"][d] = weighted
        log.info("OVERALL dim=%d: %s", d, weighted)

    out_path = RESULTS_DIR / args.output
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
