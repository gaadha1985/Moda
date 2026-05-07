"""
Recipe A' - Step 1: Build the teacher-ensemble embedding cache.

Teacher = L2-normalize( concat[
    Marqo-FashionSigLIP(x)       # 768-d
    Marqo-FashionCLIP(x)         # 512-d
    MODA-SigLIP-DeepFashion2(x)  # 768-d
] )  -->  R^2048

The resulting 2048-d vector is what the 768-d student will try to match
(via relational distillation) during Recipe A' training.

Performance shortcut: Recipe X's linear_probe_attributes.py has already
cached the FashionSigLIP and MODA-SigLIP-DeepFashion2 features for
{df_inshop, df_multimodal, hnm} under:
    results/attributes/features_cache/{dataset}__{fashionsiglip|siglip-deepfashion2}.npy

This script:
    1. Reuses those caches wherever they exist
    2. Adds the missing FashionCLIP forward pass (~30 min on MPS)
    3. Concatenates + L2-normalizes per image
    4. Writes results/distillation/teacher_cache/{dataset}_teacher_2048.npy
       and a sibling meta.json describing image-ID alignment
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
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
PROBE_CACHE = REPO / "results/attributes/features_cache"
TEACHER_CACHE = REPO / "results/distillation/teacher_cache"
TEACHER_CACHE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TEACHERS = {
    "fashionsiglip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "dim": 768, "ckpt": None,
    },
    "fashionclip": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionCLIP",
        "dim": 512, "ckpt": None,
    },
    "siglip-deepfashion2": {
        "hf_hub": "hf-hub:Marqo/marqo-fashionSigLIP",
        "dim": 768, "ckpt": "models/moda-siglip-deepfashion2/best/model_state_dict.pt",
    },
}

TEACHER_ORDER = ["fashionsiglip", "fashionclip", "siglip-deepfashion2"]


def load_model(key: str, device: str):
    cfg = TEACHERS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["hf_hub"])
    if cfg["ckpt"]:
        sd = torch.load(REPO / cfg["ckpt"], map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), preprocess


def load_images(dataset_key: str, limit: int | None):
    if dataset_key == "df_inshop":
        ds = load_dataset("Marqo/deepfashion-inshop",
                          cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [r["image"] for r in tqdm(ds, desc="df_inshop load")], \
               [r["item_ID"] for r in ds]
    if dataset_key == "df_multimodal":
        ds = load_dataset("Marqo/deepfashion-multimodal",
                          cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [r["image"] for r in tqdm(ds, desc="df_mm load")], \
               [r["item_ID"] for r in ds]
    if dataset_key == "hnm":
        df = pd.read_csv(REPO / "data/raw/hnm/articles.csv")
        img_root = REPO / "data/raw/hnm_images"
        df["_path"] = df["article_id"].apply(
            lambda a: img_root / f"{int(a):010d}"[:3] / f"{int(a):010d}.jpg")
        df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        imgs, ids = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="hnm load"):
            try:
                imgs.append(Image.open(row["_path"]).convert("RGB"))
                ids.append(str(row["article_id"]))
            except Exception:
                continue
        return imgs, ids
    raise ValueError(dataset_key)


@torch.no_grad()
def encode(images, model, preprocess, device, batch_size=96, desc="encode"):
    out = []
    for s in tqdm(range(0, len(images), batch_size), desc=desc):
        batch = images[s:s + batch_size]
        t = torch.stack([preprocess(x) for x in batch]).to(device)
        f = model.encode_image(t).float() if device == "mps" else model.encode_image(t)
        f = F.normalize(f, p=2, dim=-1)
        out.append(f.cpu().numpy())
    return np.concatenate(out, 0)


def get_teacher_feats(dataset_key: str, teacher_key: str, images, device,
                      batch_size: int) -> np.ndarray:
    probe_path = PROBE_CACHE / f"{dataset_key}__{teacher_key}.npy"
    if probe_path.exists():
        f = np.load(probe_path)
        if len(f) == len(images):
            log.info("  Reusing Recipe-X cache: %s  shape=%s",
                     probe_path.name, f.shape)
            return f
        log.warning("  probe cache len=%d != images=%d, re-extracting",
                    len(f), len(images))
    log.info("  No usable probe cache for %s/%s — extracting fresh",
             dataset_key, teacher_key)
    model, preprocess = load_model(teacher_key, device)
    f = encode(images, model, preprocess, device, batch_size=batch_size,
               desc=f"{dataset_key}/{teacher_key}")
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal", "hnm"])
    ap.add_argument("--hnm-limit", type=int, default=30000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("Using device: %s", device)

    manifest = {"teacher_order": TEACHER_ORDER,
                "teacher_dim": 2048,
                "l2_normalize_after_concat": True,
                "datasets": {}}
    t0 = time.time()

    for ds in args.datasets:
        log.info("=" * 70)
        log.info("DATASET: %s", ds)
        log.info("=" * 70)
        images, ids = load_images(
            ds, limit=(args.hnm_limit if ds == "hnm" else args.limit))
        log.info("  %d images loaded", len(images))

        per_teacher = {}
        for tk in TEACHER_ORDER:
            f = get_teacher_feats(ds, tk, images, device, args.batch_size)
            per_teacher[tk] = f

        del images

        teacher = np.concatenate([per_teacher[tk] for tk in TEACHER_ORDER], axis=1)
        teacher = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)

        out_emb = TEACHER_CACHE / f"{ds}_teacher_2048.npy"
        out_ids = TEACHER_CACHE / f"{ds}_ids.json"
        np.save(out_emb, teacher.astype(np.float32))
        with open(out_ids, "w") as f:
            json.dump({"ids": ids}, f)
        manifest["datasets"][ds] = {
            "n_images": int(len(ids)),
            "teacher_emb": str(out_emb.relative_to(REPO)),
            "ids_file": str(out_ids.relative_to(REPO)),
            "per_teacher_dims": {tk: int(per_teacher[tk].shape[1]) for tk in TEACHER_ORDER},
        }
        log.info("  Wrote teacher embeddings: %s  shape=%s",
                 out_emb, teacher.shape)

    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(TEACHER_CACHE / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Done — manifest at %s", TEACHER_CACHE / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
