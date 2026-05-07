"""
Recipe X: Linear-probe attribute extraction on frozen embeddings.

Goal: report per-attribute accuracy / macro-F1 of MODA-SigLIP-DeepFashion2
vs Marqo-FashionSigLIP using a simple linear classifier on frozen image
features. Tells us:
    1. Whether our DF2 fine-tuned model retains attribute information
       (or whether it lost some during contrastive training).
    2. Quantitative attribute-extraction performance comparable to the
       DeepFashion C&A benchmark protocol.

Datasets used (all already on disk, no leakage with LookBench — verified
in results/lookbench/data_leakage_check_v2.json):
    - Marqo/deepfashion-inshop      (52K images, category1/2/3 + color)
    - Marqo/deepfashion-multimodal  (42K images, category1/2/3)
    - H&M articles + images         (105K images, 6 structured attribute axes)

For each (model, dataset, attribute) cell:
    - Stratified 80/20 train/test split
    - Frozen 768-d image features
    - sklearn LogisticRegression (saga, multinomial, max_iter=200)
    - Metrics: top-1 accuracy, macro-F1

Outputs:
    - results/attributes/linear_probe_eval.json  (full per-cell results)
    - results/attributes/features_cache/         (cached features for re-runs)

Run:
    python benchmark/linear_probe_attributes.py [--limit N] [--model M ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results/attributes"
CACHE_DIR = RESULTS_DIR / "features_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "linear_probe_eval.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry (keep in sync with benchmark/eval_lookbench_baseline.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "fashionsiglip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "Marqo-FashionSigLIP",
        "checkpoint": None,
    },
    "siglip-deepfashion2": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-DeepFashion2",
        "checkpoint": "models/moda-siglip-deepfashion2/best/model_state_dict.pt",
    },
    "siglip-distilled": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Distilled (Recipe A')",
        "checkpoint": "models/moda-siglip-distilled/best/model_state_dict.pt",
    },
    "siglip-matryoshka": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Matryoshka (P1)",
        "checkpoint": "models/moda-siglip-matryoshka/best/model_state_dict.pt",
    },
    "siglip-text-vision": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "display": "MODA-SigLIP-Text+Vision (Recipe Y)",
        "checkpoint": "models/moda-siglip-text-vision/best/model_state_dict.pt",
    },
}

# ---------------------------------------------------------------------------
# Probe configuration
# ---------------------------------------------------------------------------
PROBE_MIN_PER_CLASS = 5
PROBE_MAX_CLASSES = 200
LR_KW = dict(
    solver="saga",
    max_iter=200,
    C=1.0,
    tol=1e-3,
)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_df_inshop(limit: int | None = None):
    log.info("Loading deepfashion-inshop ...")
    ds = load_dataset("Marqo/deepfashion-inshop",
                      cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    images, attrs = [], {"category1": [], "category2": [], "color": []}
    for row in tqdm(ds, desc="df_inshop rows"):
        images.append(row["image"])
        attrs["category1"].append(row.get("category1") or "")
        attrs["category2"].append(row.get("category2") or "")
        attrs["color"].append(row.get("color") or "")
    return images, attrs


def load_df_multimodal(limit: int | None = None):
    log.info("Loading deepfashion-multimodal ...")
    ds = load_dataset("Marqo/deepfashion-multimodal",
                      cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    images, attrs = [], {"category1": [], "category2": []}
    for row in tqdm(ds, desc="df_mm rows"):
        images.append(row["image"])
        attrs["category1"].append(row.get("category1") or "")
        attrs["category2"].append(row.get("category2") or "")
    return images, attrs


def load_hnm(limit: int | None = 30000):
    log.info("Loading H&M (limit=%s) ...", limit)
    df = pd.read_csv(REPO / "data/raw/hnm/articles.csv")
    img_root = REPO / "data/raw/hnm_images"

    def img_path(article_id) -> Path | None:
        s = f"{int(article_id):010d}"
        p = img_root / s[:3] / f"{s}.jpg"
        return p if p.exists() else None

    df["_path"] = df["article_id"].apply(img_path)
    df = df[df["_path"].notna()].reset_index(drop=True)
    log.info("  H&M articles with images on disk: %d / %d", len(df), 105_542)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        log.info("  Subsampled to %d rows", len(df))

    attr_cols = [
        "product_type_name",          # 131 classes  (fine-grained garment type)
        "product_group_name",         # 19 classes
        "garment_group_name",         # 21 classes
        "graphical_appearance_name",  # 30 classes  (pattern: solid, stripe, ...)
        "colour_group_name",          # 50 classes  (fine color)
        "perceived_colour_master_name",  # 20 classes (coarse color)
        "perceived_colour_value_name",   # 8 classes  (light/dark/dusty/...)
        "department_name",            # 250 classes  (long-tail dept names)
        "index_name",                 # 10 classes
        "index_group_name",           # 5 classes
        "section_name",               # 56 classes
    ]
    attrs = {c: df[c].fillna("").astype(str).tolist() for c in attr_cols}

    images = []
    valid_idx = []
    for i, p in enumerate(tqdm(df["_path"].tolist(), desc="hnm load images")):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_idx.append(i)
        except Exception:
            continue
    if len(valid_idx) != len(df):
        for k in attrs:
            attrs[k] = [attrs[k][i] for i in valid_idx]
    return images, attrs


# ---------------------------------------------------------------------------
# Model loading + feature extraction
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: str):
    cfg = MODEL_CONFIGS[model_key]
    log.info("Loading model: %s", cfg["display"])
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["hf_hub"])
    if cfg["checkpoint"]:
        ckpt_full = REPO / cfg["checkpoint"]
        log.info("  Applying fine-tuned checkpoint: %s", ckpt_full)
        sd = torch.load(ckpt_full, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def extract_features(images, model, preprocess, device: str,
                     batch_size: int = 32, desc: str = "encode") -> np.ndarray:
    feats = []
    for start in tqdm(range(0, len(images), batch_size), desc=desc):
        batch = images[start:start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        if device == "mps":
            f = model.encode_image(tensors).float()
        else:
            f = model.encode_image(tensors)
        f = F.normalize(f, p=2, dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def cache_path(dataset_key: str, model_key: str) -> Path:
    return CACHE_DIR / f"{dataset_key}__{model_key}.npy"


def get_or_extract_features(dataset_key: str, model_key: str, images,
                            device: str, batch_size: int = 64) -> np.ndarray:
    p = cache_path(dataset_key, model_key)
    if p.exists():
        feats = np.load(p)
        if len(feats) == len(images):
            log.info("  Loaded cached features: %s (shape=%s)", p, feats.shape)
            return feats
        log.warning("  Cache size mismatch (%d vs %d), re-extracting", len(feats), len(images))
    model, preprocess = load_model(model_key, device)
    feats = extract_features(images, model, preprocess, device,
                             batch_size=batch_size,
                             desc=f"{dataset_key}/{model_key}")
    np.save(p, feats)
    log.info("  Cached features to %s (shape=%s)", p, feats.shape)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return feats


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe_attribute(features: np.ndarray, labels: list[str],
                    attr_name: str) -> dict:
    """Train a single linear probe on (features, labels). Returns metrics."""
    counts = Counter(labels)
    valid_classes = {c for c, n in counts.items() if c and n >= PROBE_MIN_PER_CLASS}
    if len(valid_classes) > PROBE_MAX_CLASSES:
        top = sorted(valid_classes, key=lambda c: -counts[c])[:PROBE_MAX_CLASSES]
        valid_classes = set(top)

    keep = [i for i, l in enumerate(labels) if l in valid_classes]
    if len(keep) < 100 or len(valid_classes) < 2:
        return {"skipped": True, "reason": "too few samples or classes",
                "n_total": len(labels), "n_classes": len(valid_classes)}

    X = features[keep]
    y = np.array([labels[i] for i in keep])

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(**LR_KW)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    y_pred = clf.predict(X_te)

    return {
        "skipped": False,
        "attribute": attr_name,
        "n_total": int(len(keep)),
        "n_classes": int(len(valid_classes)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "top1_accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_te, y_pred, average="macro",
                                          zero_division=0)), 4),
        "majority_baseline": round(
            float(Counter(y_tr).most_common(1)[0][1]) / max(1, len(y_tr)), 4),
        "fit_seconds": round(fit_s, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "df_inshop": load_df_inshop,
    "df_multimodal": load_df_multimodal,
    "hnm": load_hnm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal", "hnm"],
                    choices=list(DATASET_LOADERS.keys()))
    ap.add_argument("--models", nargs="+",
                    default=["fashionsiglip", "siglip-deepfashion2"],
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit each dataset to N rows (for smoke testing)")
    ap.add_argument("--hnm-limit", type=int, default=30000,
                    help="Cap H&M to this many rows (default 30K for speed)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    t0 = time.time()
    results: dict = {
        "config": {
            "datasets": args.datasets,
            "models": args.models,
            "device": device,
            "logreg": LR_KW,
            "min_per_class": PROBE_MIN_PER_CLASS,
            "max_classes": PROBE_MAX_CLASSES,
            "hnm_limit": args.hnm_limit,
        },
        "datasets": {},
    }

    for ds_key in args.datasets:
        log.info("=" * 70)
        log.info("DATASET: %s", ds_key)
        log.info("=" * 70)
        loader = DATASET_LOADERS[ds_key]
        if ds_key == "hnm":
            images, attrs = loader(limit=args.hnm_limit)
        else:
            images, attrs = loader(limit=args.limit)
        log.info("  Loaded %d images, %d attribute axes", len(images), len(attrs))

        features_per_model: dict[str, np.ndarray] = {}
        for mk in args.models:
            features_per_model[mk] = get_or_extract_features(
                ds_key, mk, images, device, batch_size=args.batch_size)

        del images
        ds_results: dict = {"attributes": {}}
        for attr_name, labels in attrs.items():
            ds_results["attributes"][attr_name] = {}
            for mk in args.models:
                m = probe_attribute(features_per_model[mk], labels, attr_name)
                ds_results["attributes"][attr_name][mk] = m
                if not m.get("skipped"):
                    log.info("  [%s | %s | %s] acc=%.4f macro-F1=%.4f (n=%d, k=%d)",
                             ds_key, attr_name, mk,
                             m["top1_accuracy"], m["macro_f1"],
                             m["n_total"], m["n_classes"])
                else:
                    log.info("  [%s | %s | %s] SKIPPED: %s",
                             ds_key, attr_name, mk, m.get("reason"))
        results["datasets"][ds_key] = ds_results

    results["elapsed_seconds"] = round(time.time() - t0, 1)

    summary_rows = []
    for ds_key, ds in results["datasets"].items():
        for attr_name, attr in ds["attributes"].items():
            row = {"dataset": ds_key, "attribute": attr_name}
            for mk in args.models:
                m = attr.get(mk, {})
                if m.get("skipped"):
                    row[f"{mk}_acc"] = None
                    row[f"{mk}_f1"] = None
                else:
                    row[f"{mk}_acc"] = m.get("top1_accuracy")
                    row[f"{mk}_f1"] = m.get("macro_f1")
                    row["n_classes"] = m.get("n_classes")
                    row["n_eval"] = m.get("n_test")
            if (len(args.models) == 2
                    and row.get(f"{args.models[1]}_acc") is not None
                    and row.get(f"{args.models[0]}_acc") is not None):
                row["delta_acc"] = round(
                    row[f"{args.models[1]}_acc"] - row[f"{args.models[0]}_acc"], 4)
            summary_rows.append(row)
    results["summary_table"] = summary_rows

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 70)
    log.info("SUMMARY (acc / macro-F1) — saved to %s", OUT_JSON)
    log.info("=" * 70)
    for r in summary_rows:
        line = f"  [{r['dataset']:<14} | {r['attribute']:<30}] "
        for mk in args.models:
            a = r.get(f"{mk}_acc")
            f1v = r.get(f"{mk}_f1")
            line += f"{mk}: acc={a if a is None else f'{a:.3f}'} F1={f1v if f1v is None else f'{f1v:.3f}'}  "
        if "delta_acc" in r:
            line += f"Δacc={r['delta_acc']:+.3f}"
        log.info(line)
    log.info("Total elapsed: %.1fs", results["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
