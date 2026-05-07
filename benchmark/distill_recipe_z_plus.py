"""
Recipe Z+ — close the text-to-image gap to FashionSigLIP.

What changed vs Recipe Z
------------------------
1. **Text corpus is ~5x bigger AND in natural product-title style.**
   Each H&M article and each DF-InShop row now has ~5 LLM-paraphrased
   retail-style titles cached in `data/processed/llm_captions/*.jsonl`
   (built by `benchmark/generate_llm_captions.py`). Approx pool sizes:

       df_multimodal :  41 K natural body-description sentences (as before)
       df_inshop     : ~52 K rows * 5 LLM paraphrases = ~260 K captions
       hnm           : ~105 K rows * 5 LLM paraphrases = ~525 K captions

   Per row we randomly draw ONE caption per __getitem__ call, giving the
   text tower fresh paraphrase exposure each epoch.

2. **Text-side teacher distillation.**
   To stop the text tower drifting away from FashionSigLIP's pretrained
   fashion-vocabulary knowledge (which is what hurt Recipe Y/Z on OOD
   queries), we keep a frozen copy of stock FashionSigLIP loaded
   alongside the student and add an MSE loss on the text embeddings:

       L_text_teacher = MSE( student_text_emb, frozen_fsl_text_emb )

   Teacher embeddings are computed inline (frozen, no_grad) so we don't
   need a separate caching step.

3. **Tightened text-drift regularization** (0.02 -> 0.10).
4. **Lower InfoNCE weight** (8.0 -> 4.0). The bigger natural-style corpus
   should drive the signal organically; we dial back the explicit
   contrastive pressure that previously over-fit the text tower.
5. **3 epochs** (vs Z's 2) — more steps become productive with the
   ~5x larger corpus.

Init
----
  models/moda-siglip-recipe-z/best/model_state_dict.pt   (Recipe Z best)
  → falls back to Recipe Y, then Matryoshka, then A', then stock.

Usage
-----
    python benchmark/distill_recipe_z_plus.py \
        --epochs 3 --batch-size 96 \
        --lr-text 6e-6 --lr-vision 2e-6 \
        --infonce-weight 4.0 \
        --text-teacher-weight 5.0 \
        --text-drift-weight 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))
from distill_ensemble_to_student import _pairwise_l2  # noqa: E402
from distill_recipe_z import (  # noqa: E402
    hnm_caption_set, df_inshop_caption_set,
    rkd_distance_loss_pair, sim_mimicry,
    matryoshka_vision_loss, cross_modal_infonce, drift_reg,
    quick_text2image_eval, quick_lookbench_at_dim,
)

TEACHER_CACHE = REPO / "results/distillation/teacher_cache"
LLM_CAPTIONS_DIR = REPO / "data/processed/llm_captions"
OUT_DIR = REPO / "models/moda-siglip-recipe-z-plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "distill_recipe_z_plus.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption pool loading
# ---------------------------------------------------------------------------

def load_llm_caption_pool(source: str) -> dict[str, list[str]]:
    """Return {key: [caption, ...]} loaded from llm_captions/{source}.jsonl.

    Returns empty dict if the file is missing — callers will fall back to
    templated captions.
    """
    path = LLM_CAPTIONS_DIR / f"{source}.jsonl"
    if not path.exists():
        log.warning("LLM caption pool missing: %s — using templated fallback",
                    path)
        return {}
    pool: dict[str, list[str]] = {}
    n_lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n_lines += 1
            caps = [c for c in obj.get("captions", [])
                    if isinstance(c, str) and c.strip()]
            if caps:
                pool[str(obj["key"])] = caps
    log.info("LLM caption pool [%s]: %d keys (%d lines)", source, len(pool),
              n_lines)
    return pool


# ---------------------------------------------------------------------------
# Datasets — same shape as Recipe Z but use LLM captions when available
# ---------------------------------------------------------------------------

class DistillTextVisionDatasetPlus(Dataset):
    """DF-InShop or DF-Multimodal w/ teacher emb + LLM captions when available."""

    def __init__(self, preprocess, key: str,
                 llm_pool: Optional[dict[str, list[str]]] = None):
        assert key in {"df_inshop", "df_multimodal"}
        self.key = key
        self.preprocess = preprocess
        self.llm_pool = llm_pool or {}

        emb = TEACHER_CACHE / f"{key}_teacher_2048.npy"
        ids = TEACHER_CACHE / f"{key}_ids.json"
        if not emb.exists() or not ids.exists():
            raise FileNotFoundError(
                f"Teacher cache missing: {emb}\n"
                f"Run benchmark/cache_teacher_embeddings.py first.")
        self.teacher = np.load(emb, mmap_mode="r")
        self.ids = json.load(open(ids))["ids"]

        if key == "df_multimodal":
            self.hf_ds = load_dataset(
                "Marqo/deepfashion-multimodal",
                cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
        else:
            self.hf_ds = load_dataset(
                "Marqo/deepfashion-inshop",
                cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]

        n = min(len(self.hf_ds), len(self.teacher))
        if n != len(self.teacher):
            log.warning("[%s] size mismatch hf=%d teacher=%d -> truncate to %d",
                        key, len(self.hf_ds), len(self.teacher), n)
            self.hf_ds = self.hf_ds.select(range(n))
            self.teacher = self.teacher[:n]
            self.ids = self.ids[:n]

        if self.llm_pool:
            llm_keys = set(self.llm_pool.keys())
            n_idx = sum(1 for i in range(n) if str(i) in llm_keys)
            log.info("[%s] LLM caption coverage: %d/%d rows", key, n_idx, n)

    def __len__(self):
        return len(self.ids)

    def _draw_text(self, idx: int, row: dict) -> str:
        if self.key == "df_multimodal":
            base_text = str(row.get("text") or "").strip() or "fashion item"
            llm = self.llm_pool.get(str(idx)) if self.llm_pool else None
            if llm and random.random() < 0.5:
                return random.choice(llm)
            return base_text
        else:  # df_inshop
            llm = self.llm_pool.get(str(idx)) if self.llm_pool else None
            if llm:
                return random.choice(llm)
            return random.choice(df_inshop_caption_set(row))

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.preprocess(img)
        text = self._draw_text(idx, row)
        teacher = torch.from_numpy(np.array(self.teacher[idx])).float()
        return tensor, text, teacher, True


class HnMTextDatasetPlus(Dataset):
    """H&M w/ LLM captions when available, templated fallback otherwise."""

    def __init__(self, preprocess, csv_path=None, img_root=None,
                 limit: Optional[int] = None,
                 llm_pool: Optional[dict[str, list[str]]] = None):
        self.preprocess = preprocess
        self.llm_pool = llm_pool or {}

        csv_path = Path(csv_path or REPO / "data/raw/hnm/articles.csv")
        if not csv_path.exists():
            csv_path = REPO / "data/raw/hnm_real/articles.csv"
        self.img_root = Path(img_root or REPO / "data/raw/hnm_images")
        df = pd.read_csv(csv_path)

        def _path(article_id):
            s = f"{int(article_id):010d}"
            return self.img_root / s[:3] / f"{s}.jpg"

        df["_path"] = df["article_id"].apply(_path)
        df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)

        if self.llm_pool:
            llm_keys = set(self.llm_pool.keys())
            df["_has_llm"] = df["article_id"].apply(
                lambda a: str(int(a)) in llm_keys)
            n_with = int(df["_has_llm"].sum())
            log.info("[hnm] LLM caption coverage: %d/%d rows", n_with, len(df))
            # Prefer rows with LLM captions; if limit < n_with, only use those
            if limit is not None and limit <= n_with:
                df = df[df["_has_llm"]].sample(
                    n=limit, random_state=42).reset_index(drop=True)
            elif limit is not None and limit < len(df):
                # mix: take all LLM rows + remainder of templated
                with_llm = df[df["_has_llm"]]
                without = df[~df["_has_llm"]].sample(
                    n=limit - len(with_llm), random_state=42)
                df = pd.concat([with_llm, without], ignore_index=True)
        elif limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)

        self.df = df
        self._zero_teacher = torch.zeros(2048)
        log.info("HnMTextDatasetPlus: %d rows ready", len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        try:
            img = Image.open(row["_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        tensor = self.preprocess(img)

        key = str(int(row["article_id"]))
        llm = self.llm_pool.get(key) if self.llm_pool else None
        if llm:
            text = random.choice(llm)
        else:
            text = random.choice(hnm_caption_set(row))
        return tensor, text, self._zero_teacher, False


def collate_fn(batch, tokenizer):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    teachers = torch.stack([b[2] for b in batch])
    has_teacher = torch.tensor([b[3] for b in batch], dtype=torch.bool)
    tokens = tokenizer(texts)
    return imgs, tokens, teachers, has_teacher


# ---------------------------------------------------------------------------
# New: text-side teacher distillation loss
# ---------------------------------------------------------------------------

def text_teacher_loss(student_text: torch.Tensor,
                       teacher_text: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized student & frozen teacher text embeddings."""
    s = F.normalize(student_text, p=2, dim=-1)
    t = F.normalize(teacher_text, p=2, dim=-1)
    return F.mse_loss(s, t)


# ---------------------------------------------------------------------------
# Init resolution
# ---------------------------------------------------------------------------

def _resolve_init(arg_path: str) -> Optional[Path]:
    candidates = [
        REPO / arg_path,
        REPO / "models/moda-siglip-recipe-z/best/model_state_dict.pt",
        REPO / "models/moda-siglip-text-vision/best/model_state_dict.pt",
        REPO / "models/moda-siglip-matryoshka/best/model_state_dict.pt",
        REPO / "models/moda-siglip-distilled/best/model_state_dict.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init",
                    default="models/moda-siglip-recipe-z/best/model_state_dict.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr-vision", type=float, default=2e-6,
                    help="Smaller than Z (3e-6): vision is already great.")
    ap.add_argument("--lr-text", type=float, default=6e-6,
                    help="Slightly smaller than Z (8e-6) to reduce overfitting "
                         "to a specific caption style now that the corpus "
                         "is much larger.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--infonce-weight", type=float, default=4.0,
                    help="Halved from Z (8.0): the bigger natural-caption "
                         "corpus is now the dominant signal, we don't want "
                         "InfoNCE to over-pressure the text tower.")
    ap.add_argument("--text-teacher-weight", type=float, default=5.0,
                    help="NEW. MSE pull toward frozen FashionSigLIP text "
                         "tower. Anchors the text tower to its pretrained "
                         "fashion-vocabulary knowledge.")
    ap.add_argument("--text-drift-weight", type=float, default=0.10,
                    help="5x tighter than Z (0.02). With text-teacher loss "
                         "doing the heavy anchoring, this is a backup.")
    ap.add_argument("--vis-drift-weight", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[64, 128, 256, 384, 512, 768])
    ap.add_argument("--slice-weights", type=float, nargs="+", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hnm-limit", type=int, default=80000,
                    help="Cap H&M rows. ~80K with images covers most of the "
                         "LLM-captioned set.")
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal", "hnm"])
    ap.add_argument("--no-llm-captions", action="store_true",
                    help="Skip LLM caption pools (fall back to templates).")
    args = ap.parse_args()

    if args.slice_weights is None:
        args.slice_weights = [1.0] * len(args.slices)

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("device=%s", device)
    log.info("slices=%s slice_weights=%s", args.slices, args.slice_weights)

    log.info("Loading FashionSigLIP arch ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    init_path = _resolve_init(args.init)
    if init_path is None:
        log.warning("No fine-tuned init found — starting from stock FashionSigLIP")
    else:
        log.info("init=%s", init_path)
        sd = torch.load(init_path, map_location="cpu", weights_only=True)
        miss, unexp = model.load_state_dict(sd, strict=False)
        log.info("loaded init: missing=%d unexpected=%d", len(miss), len(unexp))
    model = model.to(device)

    # Frozen text-side teacher = STOCK FashionSigLIP text tower.
    # We use stock here, NOT our Z init, because the entire goal of the text
    # teacher loss is to pull our drifted text tower BACK toward stock
    # behaviour on common fashion vocabulary.
    log.info("Loading frozen STOCK FashionSigLIP as text teacher ...")
    teacher_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    pretrained_vis = {k: v.detach().clone().to(device)
                      for k, v in model.named_parameters()
                      if k.startswith("visual.")}
    pretrained_text = {k: v.detach().clone().to(device)
                       for k, v in model.named_parameters()
                       if not k.startswith("visual.")}

    # LLM caption pools
    llm_pools = {}
    if not args.no_llm_captions:
        for src in ("hnm", "df_inshop", "df_multimodal"):
            pool = load_llm_caption_pool(src)
            if pool:
                llm_pools[src] = pool

    dsets = []
    for k in args.datasets:
        if k == "hnm":
            dsets.append(HnMTextDatasetPlus(
                preprocess, limit=args.hnm_limit,
                llm_pool=llm_pools.get("hnm")))
        else:
            dsets.append(DistillTextVisionDatasetPlus(
                preprocess, k, llm_pool=llm_pools.get(k)))
    full = dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)
    log.info("Total training pairs: %d (sources: %s)",
             len(full), ",".join(args.datasets))

    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    vis_params = [p for n, p in model.named_parameters()
                   if n.startswith("visual.") and p.requires_grad]
    text_params = [p for n, p in model.named_parameters()
                    if not n.startswith("visual.") and p.requires_grad]
    optim = torch.optim.AdamW([
        {"params": vis_params, "lr": args.lr_vision},
        {"params": text_params, "lr": args.lr_text},
    ], weight_decay=args.weight_decay)
    log.info("vis params=%d  text params=%d", len(vis_params), len(text_params))

    best_score = -1.0
    step = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        log.info("=" * 70)
        log.info("EPOCH %d/%d", epoch + 1, args.epochs)
        log.info("=" * 70)

        for images, tokens, teacher, has_teacher in tqdm(
                loader, desc=f"z+ ep{epoch+1}"):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher = F.normalize(teacher, p=2, dim=-1)
            has_teacher = has_teacher.to(device, non_blocking=True)

            s_full_img = model.encode_image(images)
            s_full_txt = model.encode_text(tokens)
            if device == "mps":
                s_full_img = s_full_img.float()
                s_full_txt = s_full_txt.float()

            # Text teacher embeddings (frozen, no grad)
            with torch.no_grad():
                t_text = teacher_model.encode_text(tokens)
                if device == "mps":
                    t_text = t_text.float()

            mask = has_teacher
            if mask.any():
                l_vis_mrl, _ = matryoshka_vision_loss(
                    s_full_img[mask], teacher[mask], args.slices,
                    args.slice_weights, args.rkd_weight, args.sim_weight)
            else:
                l_vis_mrl = torch.tensor(0.0, device=device)

            l_cross = cross_modal_infonce(s_full_img, s_full_txt,
                                           temperature=args.temperature)
            l_text_teacher = text_teacher_loss(s_full_txt, t_text)

            l_vis_drift = drift_reg(model, pretrained_vis, "visual.")
            l_text_drift = drift_reg(model, pretrained_text, "")
            l_text_drift = l_text_drift - l_vis_drift

            loss = (l_vis_mrl
                    + args.infonce_weight * l_cross
                    + args.text_teacher_weight * l_text_teacher
                    + args.vis_drift_weight * l_vis_drift
                    + args.text_drift_weight * l_text_drift)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vis_params + text_params, 1.0)
            optim.step()

            step += 1
            if step % 25 == 0:
                head = (f"  step={step} loss={loss.item():.4f} "
                        f"vis_mrl={l_vis_mrl.item():.4f} "
                        f"cross={l_cross.item():.4f} "
                        f"txt_tch={l_text_teacher.item():.4f} "
                        f"hT_frac={mask.float().mean().item():.2f} "
                        f"text_drift={l_text_drift.item():.3e}")
                log.info(head)

            if args.eval_every > 0 and step % args.eval_every == 0:
                model.eval()
                txt2img = quick_text2image_eval(model, preprocess, tokenizer,
                                                  device)
                lb_full = quick_lookbench_at_dim(model, preprocess, device,
                                                   args.slices[-1],
                                                   "real_studio_flat")
                lb_64 = quick_lookbench_at_dim(model, preprocess, device,
                                                 args.slices[0],
                                                 "real_studio_flat")
                model.train()
                composite = (0.6 * txt2img["recall@1"]
                             + 0.4 * lb_full["fine_r_at_1"])
                log.info("  [eval @ step %d] text2img=%s  lb_full=%s  lb_64=%s  "
                          "composite=%.2f",
                          step, txt2img, lb_full, lb_64, composite)
                history.append({"step": step,
                                "text2img": txt2img,
                                "lb_full": lb_full,
                                "lb_64": lb_64,
                                "composite": composite})
                if composite > best_score:
                    best_score = composite
                    best_path = OUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save(model.state_dict(),
                                best_path / "model_state_dict.pt")
                    with open(best_path / "meta.json", "w") as f:
                        json.dump({
                            "training": "Recipe Z+ — LLM caps + text teacher",
                            "init_from": str(init_path) if init_path else "stock",
                            "step": step,
                            "epoch": epoch,
                            "composite": composite,
                            "text2img_eval": txt2img,
                            "lookbench_full": lb_full,
                            "lookbench_64": lb_64,
                            "args": vars(args),
                        }, f, indent=2)
                    log.info("  *** new best composite=%.2f ***", composite)

        ckpt = OUT_DIR / f"epoch_{epoch + 1}"
        ckpt.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt / "model_state_dict.pt")

    elapsed = time.time() - t0
    log.info("Recipe Z+ complete in %.1f min. Best composite = %.2f",
             elapsed / 60, best_score)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "best_composite": best_score,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
