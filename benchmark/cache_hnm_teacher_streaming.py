"""
Streaming H&M teacher cache builder.

Why a separate script:
The original benchmark/cache_teacher_embeddings.py loads ALL images into a
Python list of PIL.Image objects before encoding. For H&M with 30K
~720x1080 RGB images that's >70 GB of resident memory and the process
gets OOM-killed (we observed silent death at ~13.4K/30K with a leaked
multiprocessing semaphore).

This script keeps the SAME output contract as cache_teacher_embeddings.py
for the H&M dataset:
    results/distillation/teacher_cache/hnm_teacher_2048.npy   (N, 2048) f32
    results/distillation/teacher_cache/hnm_ids.json
    + updates / creates results/distillation/teacher_cache/manifest.json

…but encodes each teacher in a streaming fashion: for one teacher at a
time we open + preprocess + encode in batches and discard PIL objects
immediately, so resident memory is bounded by `--batch-size` images.

Teacher order matches cache_teacher_embeddings.py:
    fashionsiglip (768) | fashionclip (512) | siglip-deepfashion2 (768)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
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
        "dim": 768,
        "ckpt": "models/moda-siglip-deepfashion2/best/model_state_dict.pt",
    },
}

TEACHER_ORDER = ["fashionsiglip", "fashionclip", "siglip-deepfashion2"]


def load_teacher(key: str, device: str):
    cfg = TEACHERS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["hf_hub"])
    if cfg["ckpt"]:
        sd = torch.load(REPO / cfg["ckpt"], map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), preprocess


def list_hnm_paths(limit: int | None) -> tuple[list[Path], list[str]]:
    df = pd.read_csv(REPO / "data/raw/hnm/articles.csv")
    img_root = REPO / "data/raw/hnm_images"
    df["_path"] = df["article_id"].apply(
        lambda a: img_root / f"{int(a):010d}"[:3] / f"{int(a):010d}.jpg")
    df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    paths = [Path(p) for p in df["_path"].tolist()]
    ids = [str(a) for a in df["article_id"].tolist()]
    return paths, ids


@torch.no_grad()
def stream_encode(paths: list[Path], model, preprocess, device: str,
                  batch_size: int, desc: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    bad = 0
    for s in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch_paths = paths[s:s + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
                tensors.append(preprocess(img))
            except Exception:
                bad += 1
                tensors.append(torch.zeros(3, 224, 224))
        t = torch.stack(tensors).to(device)
        f = model.encode_image(t).float() if device == "mps" else model.encode_image(t)
        f = F.normalize(f, p=2, dim=-1)
        chunks.append(f.cpu().numpy())
        del t, f
    if bad:
        log.warning("  %d image(s) failed to load (zero-padded so dims align)", bad)
    return np.concatenate(chunks, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hnm-limit", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-per-teacher", action="store_true",
                    help="also write per-teacher *.npy under teacher_cache/")
    args = ap.parse_args()

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("Device: %s", device)
    log.info("HNM limit: %d  batch-size: %d", args.hnm_limit, args.batch_size)

    t0 = time.time()
    log.info("Listing H&M image paths…")
    paths, ids = list_hnm_paths(args.hnm_limit)
    log.info("  %d images selected", len(paths))

    per_teacher: dict[str, np.ndarray] = {}
    for tk in TEACHER_ORDER:
        log.info("=" * 70)
        log.info("Encoding teacher: %s (dim=%d)", tk, TEACHERS[tk]["dim"])
        log.info("=" * 70)
        model, preprocess = load_teacher(tk, device)
        feats = stream_encode(paths, model, preprocess, device,
                              args.batch_size, desc=f"hnm/{tk}")
        log.info("  encoded shape=%s", feats.shape)
        if args.keep_per_teacher:
            out = TEACHER_CACHE / f"hnm__{tk}.npy"
            np.save(out, feats.astype(np.float32))
            log.info("  per-teacher cache: %s", out)
        per_teacher[tk] = feats
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    log.info("Concatenating to 2048-d teacher and L2-normalizing…")
    teacher = np.concatenate([per_teacher[tk] for tk in TEACHER_ORDER], axis=1)
    teacher = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)
    log.info("  final shape=%s", teacher.shape)

    out_emb = TEACHER_CACHE / "hnm_teacher_2048.npy"
    out_ids = TEACHER_CACHE / "hnm_ids.json"
    np.save(out_emb, teacher.astype(np.float32))
    with open(out_ids, "w") as f:
        json.dump({"ids": ids}, f)
    log.info("Wrote: %s  shape=%s", out_emb, teacher.shape)
    log.info("Wrote: %s", out_ids)

    manifest_path = TEACHER_CACHE / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "teacher_order": TEACHER_ORDER,
            "teacher_dim": 2048,
            "l2_normalize_after_concat": True,
            "datasets": {},
        }
    manifest.setdefault("datasets", {})
    manifest["datasets"]["hnm"] = {
        "n_images": int(len(ids)),
        "teacher_emb": str(out_emb.relative_to(REPO)),
        "ids_file": str(out_ids.relative_to(REPO)),
        "per_teacher_dims": {tk: int(per_teacher[tk].shape[1]) for tk in TEACHER_ORDER},
        "built_by": "cache_hnm_teacher_streaming.py",
    }
    manifest["elapsed_seconds_hnm"] = round(time.time() - t0, 1)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Updated manifest: %s", manifest_path)
    log.info("DONE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
