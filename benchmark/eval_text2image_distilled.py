"""
P0 sanity check: text->image retrieval with the distilled student.

Recipe A' only fine-tuned the *vision* tower of FashionSigLIP. The text tower
was inherited unchanged from Marqo/marqo-fashionSigLIP. This script verifies
that vision-tower distillation has NOT broken text->image alignment.

Setup:
- Eval set: Marqo/deepfashion-multimodal (42K images with one natural-language
  caption + item_ID per image). We sample a query subset and use the full
  remaining set + same items as the gallery, with item_ID as ground truth.
- Models compared:
    1. FashionSigLIP (Marqo, baseline) — vision + text both stock
    2. MODA-SigLIP-Distilled — Recipe A' vision tower + original Marqo text tower

Metric: Recall@K (text -> image) for K in {1, 5, 10}, item_ID-exact match.

If R@1 of (2) is within ~3% of (1), text alignment is preserved and we can
use the distilled vision encoder in any text-conditioned downstream pipeline
without retraining the text tower.
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
from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "lookbench"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def device_pick(arg):
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(distilled: bool, device: str):
    """Load FashionSigLIP. If distilled=True, swap visual weights with
    Recipe A' student checkpoint while keeping the original text tower."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    if distilled:
        ckpt = REPO / "models/moda-siglip-distilled/best/model_state_dict.pt"
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        log.info("  loaded distilled student weights from %s", ckpt)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, images, device: str, batch_size: int = 32):
    feats = []
    for s in tqdm(range(0, len(images), batch_size), desc="img-enc"):
        batch = images[s:s + batch_size]
        t = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        f = model.encode_image(t)
        if device == "mps":
            f = f.float()
        feats.append(F.normalize(f, p=2, dim=-1).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device: str, batch_size: int = 64):
    feats = []
    for s in tqdm(range(0, len(texts), batch_size), desc="txt-enc"):
        batch = texts[s:s + batch_size]
        toks = tokenizer(batch).to(device)
        f = model.encode_text(toks)
        if device == "mps":
            f = f.float()
        feats.append(F.normalize(f, p=2, dim=-1).cpu())
    return torch.cat(feats, 0)


def recall_at_k(text_feats, image_feats, q_ids, g_ids, ks=(1, 5, 10)):
    sims = text_feats @ image_feats.T
    out = {}
    g_id_arr = np.asarray(g_ids)
    for k in ks:
        topk = sims.topk(k, dim=1).indices.numpy()
        hits = 0
        for i, q in enumerate(q_ids):
            if q in g_id_arr[topk[i]]:
                hits += 1
        out[f"recall@{k}"] = round(100 * hits / len(q_ids), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=2000,
                    help="Number of text queries to sample (gallery = all 42K imgs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="text2image_distilled_eval.json")
    args = ap.parse_args()

    device = device_pick(args.device)
    log.info("device=%s", device)

    log.info("Loading Marqo/deepfashion-multimodal ...")
    ds = load_dataset(
        "Marqo/deepfashion-multimodal",
        cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
    log.info("  total images=%d", len(ds))

    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(len(ds), size=min(args.n_query, len(ds)),
                       replace=False).tolist()

    log.info("Materializing %d query texts + %d gallery images ...",
             len(q_idx), len(ds))
    texts = [ds[i]["text"] for i in q_idx]
    q_ids = [ds[i]["item_ID"] for i in q_idx]
    g_imgs = [ds[i]["image"] for i in range(len(ds))]
    g_ids = [ds[i]["item_ID"] for i in range(len(ds))]

    results = {}
    for label, distilled in [("fashionsiglip_baseline", False),
                              ("moda_siglip_distilled", True)]:
        log.info("=" * 60)
        log.info("Evaluating: %s (distilled_vision=%s)", label, distilled)
        log.info("=" * 60)
        model, preprocess, tokenizer = load_model(distilled, device)

        t0 = time.time()
        img_feats = encode_images(model, preprocess, g_imgs, device,
                                   batch_size=args.batch_size)
        txt_feats = encode_texts(model, tokenizer, texts, device,
                                  batch_size=max(args.batch_size, 64))
        t_enc = time.time() - t0

        r = recall_at_k(txt_feats, img_feats, q_ids, g_ids)
        log.info("  %s -> R@1=%.2f R@5=%.2f R@10=%.2f (enc %.1fs)",
                 label, r["recall@1"], r["recall@5"], r["recall@10"], t_enc)
        results[label] = {
            "n_query": len(q_ids),
            "n_gallery": len(g_ids),
            "metrics": r,
            "encode_seconds": round(t_enc, 1),
        }

        del model
        torch.mps.empty_cache() if device == "mps" else None

    base = results["fashionsiglip_baseline"]["metrics"]
    dist = results["moda_siglip_distilled"]["metrics"]
    delta = {k: round(dist[k] - base[k], 2) for k in base}
    results["delta_distilled_minus_baseline"] = delta
    results["verdict"] = ("PRESERVED" if delta["recall@1"] >= -3.0
                          else "REGRESSED")
    log.info("=" * 60)
    log.info("DELTA (distilled - baseline): %s", delta)
    log.info("Verdict: %s", results["verdict"])
    log.info("=" * 60)

    out = RESULTS_DIR / args.output
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
