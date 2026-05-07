"""
Evaluate the MoDA Matryoshka student on LookBench under several
post-training quantization regimes:

  fp32        : baseline (full precision cosine)
  fp16        : embeddings cast to float16 then cosine
  int8        : per-dimension asymmetric min/max -> int8 -> dequant -> cosine
  binary      : sign() -> ±1 -> cosine over signs (equivalent to Hamming-derived sim)
  binary+rrnk : Hamming top-K rerank with fp16 cosine

The model is encoded ONCE per subset on CPU (safe to run alongside an MPS
training job). Embeddings are cached to disk so subsequent sweeps cost
microseconds.

Output: results/lookbench/quantized_eval.json

Usage:
  python benchmark/eval_lookbench_quantized.py
  python benchmark/eval_lookbench_quantized.py --slices 256 384 512
  python benchmark/eval_lookbench_quantized.py --variants fp32 int8 binary
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
EMB_CACHE = REPO / "data" / "processed" / "embeddings" / "lookbench_matryoshka"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SUBSETS = ["real_studio_flat", "aigen_studio",
           "real_streetlook", "aigen_streetlook"]

ALL_VARIANTS = ["fp32", "fp16", "int8", "binary", "binary_rerank"]


# ---------------------------------------------------------------------------
# Encoding (CPU-only by default to avoid MPS contention with training jobs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(model, preprocess, images, device, batch_size=8):
    feats = []
    for s in tqdm(range(0, len(images), batch_size), desc="encode"):
        batch = images[s:s + batch_size]
        t = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        f = model.encode_image(t)
        if device == "mps":
            f = f.float()
        feats.append(f.cpu())
    return torch.cat(feats, 0).numpy().astype(np.float32)  # raw, not normalised


def labels(split):
    return [(str(x.get("category", "")).lower().strip(),
             f"{str(x.get('category', '')).lower().strip()}|"
             f"{str(x.get('main_attribute', '')).lower().strip()}",
             x.get("item_ID"))
            for x in split]


def cache_path(subset: str, kind: str) -> Path:
    return EMB_CACHE / f"{subset}__{kind}.npy"


def load_or_encode(model, preprocess, device, lb_root, subset, batch_size,
                   include_noise: bool = True):
    """Return (qf, gf, q_lab, g_lab) for a subset.

    If include_noise=True (default, official LookBench protocol) the shared
    noise gallery is concatenated to the gallery. If False (smoke-test mode),
    only the subset's own gallery is used — much faster but inflated metrics.
    """
    suffix = "with_noise" if include_noise else "no_noise"
    q_path = cache_path(subset, "query")
    g_path = cache_path(subset, f"gallery_{suffix}")
    ql_path = cache_path(subset, "query_labels").with_suffix(".json")
    gl_path = cache_path(subset, f"gallery_{suffix}_labels").with_suffix(".json")

    if all(p.exists() for p in (q_path, g_path, ql_path, gl_path)):
        log.info("[%s] loading cached embeddings (q=%s g=%s)",
                 subset, q_path.name, g_path.name)
        qf = np.load(q_path)
        gf = np.load(g_path)
        q_lab = json.loads(ql_path.read_text())
        g_lab = json.loads(gl_path.read_text())
        # JSON loses tuples -> back to tuples for comparison consistency
        q_lab = [tuple(x) for x in q_lab]
        g_lab = [tuple(x) for x in g_lab]
        return qf, gf, q_lab, g_lab

    dsd = load_from_disk(str(lb_root / subset))
    qs, gs = dsd["query"], dsd["gallery"]
    log.info("[%s] queries=%d gallery=%d  (encoding ...)", subset, len(qs), len(gs))

    qf = encode(model, preprocess, qs["image"], device, batch_size)
    gf_subset = encode(model, preprocess, gs["image"], device, batch_size)
    q_lab = labels(qs)
    g_lab_subset = labels(gs)

    if include_noise:
        # Shared noise gallery — cache once (under "_noise" subset key)
        noise_path = cache_path("_noise", "gallery")
        noise_lab_path = cache_path("_noise", "gallery_labels").with_suffix(".json")
        if noise_path.exists() and noise_lab_path.exists():
            nf = np.load(noise_path)
            n_lab = [tuple(x) for x in json.loads(noise_lab_path.read_text())]
        else:
            log.info("Encoding shared noise gallery (one-time) ...")
            noise_dsd = load_from_disk(str(lb_root / "noise"))
            noise_g = noise_dsd["gallery"] if "gallery" in noise_dsd else noise_dsd
            nf = encode(model, preprocess, noise_g["image"], device, batch_size)
            n_lab = labels(noise_g)
            np.save(noise_path, nf)
            noise_lab_path.write_text(json.dumps(n_lab))
        gf = np.concatenate([gf_subset, nf], axis=0)
        g_lab = g_lab_subset + n_lab
    else:
        gf = gf_subset
        g_lab = g_lab_subset

    np.save(q_path, qf)
    np.save(g_path, gf)
    ql_path.write_text(json.dumps(q_lab))
    gl_path.write_text(json.dumps(g_lab))
    log.info("[%s] cached: q=%s g=%s (gallery+noise=%d)",
             subset, qf.shape, gf.shape, gf.shape[0])
    return qf, gf, q_lab, g_lab


# ---------------------------------------------------------------------------
# Quantization primitives  (operate on raw fp32, return matrices ready for
# cosine via dot-product after L2-normalization)
# ---------------------------------------------------------------------------

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def quant_fp32(qf, gf):
    return l2norm(qf), l2norm(gf), {"q_bytes": qf.shape[1] * 4, "g_bytes": gf.shape[1] * 4}


def quant_fp16(qf, gf):
    qf16 = qf.astype(np.float16).astype(np.float32)
    gf16 = gf.astype(np.float16).astype(np.float32)
    return l2norm(qf16), l2norm(gf16), {"q_bytes": qf.shape[1] * 2, "g_bytes": gf.shape[1] * 2}


def quant_int8_per_dim(qf, gf, calib: np.ndarray | None = None):
    """Per-dimension asymmetric min/max int8.

    Calibrated on the GALLERY (database) only — queries are quantized using the
    same scale/zero-point at search time. Embeddings are L2-normalized AFTER
    dequantization so cosine = inner product on unit vectors.
    """
    src = calib if calib is not None else gf
    mn = src.min(axis=0)            # per-dim min
    mx = src.max(axis=0)            # per-dim max
    scale = (mx - mn) / 255.0
    scale = np.where(scale < 1e-12, 1e-12, scale)
    zp = (-mn / scale).round().clip(0, 255).astype(np.uint8)

    def q(x):
        q = ((x - mn) / scale).round().clip(0, 255).astype(np.uint8)
        return (q.astype(np.float32) - zp.astype(np.float32)) * scale  # dequant

    return l2norm(q(qf)), l2norm(q(gf)), {"q_bytes": qf.shape[1] * 1, "g_bytes": gf.shape[1] * 1}


def quant_binary(qf, gf):
    """sign() -> ±1. Cosine on these is equivalent (up to constant) to
    1 - 2*Hamming/d. Equivalent ranking, simpler implementation."""
    qb = np.where(qf >= 0, 1.0, -1.0).astype(np.float32)
    gb = np.where(gf >= 0, 1.0, -1.0).astype(np.float32)
    # length is sqrt(d); l2 normalize for cosine
    return l2norm(qb), l2norm(gb), {"q_bytes": qf.shape[1] // 8, "g_bytes": gf.shape[1] // 8}


# ---------------------------------------------------------------------------
# Two-stage cascade: binary top-K then fp16 rerank
# ---------------------------------------------------------------------------

def search_binary_then_rerank(qf, gf, top_k_binary=100, top_k_final=10):
    qb, gb, _ = quant_binary(qf, gf)
    sim_bin = qb @ gb.T                 # (Q, G)
    # top-K_binary by sign-cosine
    cand = np.argpartition(-sim_bin, top_k_binary, axis=1)[:, :top_k_binary]
    # rerank with fp16 cosine
    qf16 = l2norm(qf.astype(np.float16).astype(np.float32))
    gf16 = l2norm(gf.astype(np.float16).astype(np.float32))
    nq = qf.shape[0]
    final_scores = np.empty((nq, top_k_binary), dtype=np.float32)
    for i in range(nq):
        final_scores[i] = qf16[i] @ gf16[cand[i]].T
    rerank_order = np.argsort(-final_scores, axis=1)[:, :top_k_final]
    final_idx = np.take_along_axis(cand, rerank_order, axis=1)
    # Build a Q x top_k_final score matrix sentinel; not used for metrics directly
    return final_idx


# ---------------------------------------------------------------------------
# Metrics  (works on either similarity matrix or precomputed top-K indices)
# ---------------------------------------------------------------------------

def _metrics_from_topk(topk_indices: np.ndarray, q_lab, g_lab, ks=(1, 5, 10), ndcg_k=5):
    out = {"fine_recall": {}, "coarse_recall": {}, "id_recall": {}}
    n = len(q_lab)
    for k in ks:
        nf = nc = ni = 0
        for i in range(n):
            qc, qfine, qid = q_lab[i]
            top = [g_lab[j] for j in topk_indices[i, :k]]
            if any(t[1] == qfine for t in top):
                nf += 1
            if any(t[0] == qc for t in top):
                nc += 1
            if qid is not None and any(t[2] == qid for t in top):
                ni += 1
        out["fine_recall"][f"recall@{k}"] = round(100 * nf / n, 2)
        out["coarse_recall"][f"recall@{k}"] = round(100 * nc / n, 2)
        out["id_recall"][f"recall@{k}"] = round(100 * ni / n, 2)

    ndcg_total = 0.0
    for i in range(n):
        qfine = q_lab[i][1]
        rels = [1 if g_lab[j][1] == qfine else 0
                for j in topk_indices[i, :ndcg_k]]
        dcg = sum((2 ** r - 1) / np.log2(p + 2) for p, r in enumerate(rels))
        n_rel = sum(1 for x in g_lab if x[1] == qfine)
        ideal = (sum(1 / np.log2(p + 2) for p in range(min(ndcg_k, n_rel)))
                 if n_rel else 1.0)
        ndcg_total += (dcg / ideal) if ideal else 0.0
    out["ndcg@5"] = round(100 * ndcg_total / n, 2)
    return out


def metrics_from_sim(qf, gf, q_lab, g_lab, max_k=10):
    sim = qf @ gf.T
    topk = np.argpartition(-sim, max_k, axis=1)[:, :max_k]
    sorted_within = np.argsort(-np.take_along_axis(sim, topk, axis=1), axis=1)
    topk_sorted = np.take_along_axis(topk, sorted_within, axis=1)
    return _metrics_from_topk(topk_sorted, q_lab, g_lab)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/moda-siglip-matryoshka/best/model_state_dict.pt")
    ap.add_argument("--slices", type=int, nargs="+", default=[256])
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="CPU-friendly batch size to avoid contention with running training jobs")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, safe) | mps (faster but contends with MPS training)")
    ap.add_argument("--rerank-k-binary", type=int, default=100,
                    help="Top-K candidates from binary search before fp16 rerank")
    ap.add_argument("--no-noise", action="store_true",
                    help="Skip shared noise gallery (smoke-test only — inflated metrics)")
    ap.add_argument("--output", default="quantized_eval.json")
    args = ap.parse_args()

    device = args.device
    log.info("device=%s slices=%s variants=%s", device, args.slices, args.variants)

    log.info("Loading matryoshka student from %s", args.checkpoint)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    sd = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    lb_root = REPO / "data/raw/lookbench/datasets"

    out: dict = {
        "checkpoint": args.checkpoint,
        "slices": args.slices,
        "variants": args.variants,
        "subsets": {},
        "device": device,
        "batch_size": args.batch_size,
    }

    # Encode (or load cached) per subset, then sweep all (slice, variant)
    encoded = {}
    for subset in args.subsets:
        t0 = time.time()
        qf, gf, q_lab, g_lab = load_or_encode(
            model, preprocess, device, lb_root, subset, args.batch_size,
            include_noise=not args.no_noise)
        encoded[subset] = (qf, gf, q_lab, g_lab)
        log.info("[%s] ready in %.1fs (q=%s g=%s)", subset, time.time() - t0, qf.shape, gf.shape)

    # Quantization sweeps  (model is no longer needed -> free)
    del model

    for subset in args.subsets:
        out["subsets"][subset] = {}
        qf_full, gf_full, q_lab, g_lab = encoded[subset]
        for d in args.slices:
            qf = qf_full[:, :d]
            gf = gf_full[:, :d]
            out["subsets"][subset][d] = {}
            for variant in args.variants:
                t0 = time.time()
                if variant == "binary_rerank":
                    topk = search_binary_then_rerank(
                        qf, gf, top_k_binary=args.rerank_k_binary, top_k_final=10)
                    m = _metrics_from_topk(topk, q_lab, g_lab)
                    bytes_per_q = qf.shape[1] // 8 + qf.shape[1] * 2  # binary + fp16 for rerank context
                    bytes_per_g = qf.shape[1] // 8 + qf.shape[1] * 2
                    m["bytes"] = {"q": bytes_per_q, "g": bytes_per_g,
                                  "note": f"binary index + fp16 for top-{args.rerank_k_binary} rerank"}
                else:
                    qq, gg, sz = {
                        "fp32": quant_fp32,
                        "fp16": quant_fp16,
                        "int8": quant_int8_per_dim,
                        "binary": quant_binary,
                    }[variant](qf, gf)
                    m = metrics_from_sim(qq, gg, q_lab, g_lab)
                    m["bytes"] = sz
                out["subsets"][subset][d][variant] = m
                log.info("  [%s d=%d %s] Fine R@1=%.2f Coarse R@1=%.2f nDCG@5=%.2f  (%.2fs)",
                         subset, d, variant,
                         m["fine_recall"]["recall@1"],
                         m["coarse_recall"]["recall@1"],
                         m["ndcg@5"], time.time() - t0)

    # Query-weighted overall per (slice, variant)
    n_per = {s: len(load_from_disk(str(lb_root / s))["query"])
             for s in args.subsets}
    total_q = sum(n_per.values())
    out["overall"] = {}
    for d in args.slices:
        out["overall"][d] = {}
        for variant in args.variants:
            agg = {"fine_recall@1": 0.0, "coarse_recall@1": 0.0,
                   "ndcg@5": 0.0, "id_recall@1": 0.0}
            for s in args.subsets:
                sm = out["subsets"][s][d][variant]
                w = n_per[s] / total_q
                agg["fine_recall@1"] += w * sm["fine_recall"]["recall@1"]
                agg["coarse_recall@1"] += w * sm["coarse_recall"]["recall@1"]
                agg["ndcg@5"] += w * sm["ndcg@5"]
                agg["id_recall@1"] += w * sm["id_recall"]["recall@1"]
            agg = {k: round(v, 2) for k, v in agg.items()}
            out["overall"][d][variant] = agg
            log.info("OVERALL d=%d %-14s -> Fine R@1=%.2f  Coarse R@1=%.2f  nDCG@5=%.2f  ID R@1=%.2f",
                     d, variant, agg["fine_recall@1"], agg["coarse_recall@1"],
                     agg["ndcg@5"], agg["id_recall@1"])

    out_path = RESULTS_DIR / args.output
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Saved %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
