"""
Recipe Y - Joint text + vision distillation on top of the Matryoshka student.

Builds on Recipe A' (vision-only distill from ensemble teacher) and Recipe
Matryoshka (multi-resolution prefixes). Recipe Y now also trains the text
tower so that:

  (a) Vision tower keeps Matryoshka property + ensemble-grade quality
      (continuation of MRL distillation losses).
  (b) Text tower is no longer frozen-stock; it is fine-tuned with an
      InfoNCE contrastive objective against the (now better) vision tower
      on DeepFashion-MultiModal (image, caption) pairs.
  (c) Cross-modal alignment (text -> image retrieval) improves vs the
      stock FashionSigLIP text tower while vision quality is held by
      the per-slice distillation signal.

Init:
  Vision tower from models/moda-siglip-matryoshka/best/...
  Text tower from stock Marqo/marqo-fashionSigLIP (unchanged at start)

Losses:
  L_vis_mrl       : per-slice RKD-D + per-slice sim-mimicry on cached 2048-d
                    ensemble teacher (same as distill_matryoshka.py)
  L_cross_infonce : symmetric InfoNCE between batch image & text embeddings
                    (cross-modal contrastive on full 768-d)
  L_text_drift    : L2 drift on text-tower weights vs stock (prevents
                    text-tower collapse / catastrophic forgetting)
  L_vis_drift     : L2 drift on vision-tower weights vs MRL init

Usage:
    python benchmark/distill_text_vision.py \
        --epochs 2 --batch-size 96 --lr 5e-6 \
        --slices 64 128 256 384 512 768 \
        --infonce-weight 5.0 --temperature 0.07
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
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))
from distill_ensemble_to_student import _pairwise_l2  # noqa: E402

TEACHER_CACHE = REPO / "results/distillation/teacher_cache"
OUT_DIR = REPO / "models/moda-siglip-text-vision"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "distill_text_vision.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: paired (image_tensor, text_tokens, teacher_2048) for DF-MultiModal
# ---------------------------------------------------------------------------

class TextVisionDistillDataset(Dataset):
    """Yields (image_tensor, text_str, teacher_2048_emb) for every DF-MM row.

    The teacher embedding is reused from the cache built by
    cache_teacher_embeddings.py. Text is left as a string and tokenised
    in the collate fn.
    """

    def __init__(self, preprocess, dataset_key: str = "df_multimodal"):
        self.preprocess = preprocess
        emb = TEACHER_CACHE / f"{dataset_key}_teacher_2048.npy"
        ids = TEACHER_CACHE / f"{dataset_key}_ids.json"
        if not emb.exists() or not ids.exists():
            raise FileNotFoundError(
                f"Teacher cache missing: {emb}\n"
                f"Run benchmark/cache_teacher_embeddings.py first.")
        self.teacher = np.load(emb, mmap_mode="r")
        self.ids = json.load(open(ids))["ids"]

        if dataset_key == "df_multimodal":
            self.hf_ds = load_dataset(
                "Marqo/deepfashion-multimodal",
                cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
        elif dataset_key == "df_inshop":
            # InShop has no clean caption; we fall back to category-based text
            self.hf_ds = load_dataset(
                "Marqo/deepfashion-inshop",
                cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]
        else:
            raise ValueError(dataset_key)
        self.key = dataset_key

        n = min(len(self.hf_ds), len(self.teacher))
        if n != len(self.teacher):
            log.warning("size mismatch: hf=%d teacher=%d -> truncating to %d",
                        len(self.hf_ds), len(self.teacher), n)
            self.hf_ds = self.hf_ds.select(range(n))
            self.teacher = self.teacher[:n]
            self.ids = self.ids[:n]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.preprocess(img)
        if self.key == "df_multimodal":
            text = str(row.get("text", ""))
        else:
            text = (f"a photo of a {row.get('category2', '')} "
                    f"in {row.get('color', '')}").strip()
        teacher = torch.from_numpy(np.array(self.teacher[idx])).float()
        return tensor, text, teacher


def collate_fn(batch, tokenizer):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    teachers = torch.stack([b[2] for b in batch])
    tokens = tokenizer(texts)
    return imgs, tokens, teachers


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def rkd_distance_loss_pair(student: torch.Tensor, teacher: torch.Tensor):
    with torch.no_grad():
        d_t = _pairwise_l2(teacher)
        mean_t = d_t[d_t > 0].mean()
        d_t = d_t / (mean_t + 1e-12)
    d_s = _pairwise_l2(student)
    mean_s = d_s[d_s > 0].mean()
    d_s = d_s / (mean_s + 1e-12)
    return F.smooth_l1_loss(d_s, d_t)


def sim_mimicry(student, teacher):
    return F.mse_loss(student @ student.T, teacher @ teacher.T)


def matryoshka_vision_loss(s_full, teacher, slices, slice_weights,
                            rkd_weight, sim_weight):
    total = torch.zeros((), device=s_full.device)
    per = {}
    for d, w in zip(slices, slice_weights):
        s_slice = F.normalize(s_full[:, :d], p=2, dim=-1)
        l_rkd = rkd_distance_loss_pair(s_slice, teacher)
        l_sim = sim_mimicry(s_slice, teacher)
        total = total + w * (rkd_weight * l_rkd + sim_weight * l_sim)
        per[d] = {"rkd": float(l_rkd.detach()), "sim": float(l_sim.detach())}
    return total, per


def cross_modal_infonce(image_feats, text_feats, temperature=0.07):
    """Symmetric InfoNCE between paired (image_i, text_i)."""
    img = F.normalize(image_feats, p=2, dim=-1)
    txt = F.normalize(text_feats, p=2, dim=-1)
    logits_i2t = (img @ txt.T) / temperature
    targets = torch.arange(img.size(0), device=img.device)
    return 0.5 * (F.cross_entropy(logits_i2t, targets) +
                  F.cross_entropy(logits_i2t.T, targets))


def drift_reg(model, pretrained, prefix):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for name, p in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if name in pretrained and p.requires_grad:
            total = total + ((p - pretrained[name]) ** 2).sum()
            n += p.numel()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Quick eval helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_text2image_eval(model, preprocess, tokenizer, device,
                           n_query=500, batch_size=32):
    """Tiny text->image R@1 / R@5 / R@10 sanity on DF-MultiModal sample."""
    ds = load_dataset(
        "Marqo/deepfashion-multimodal",
        cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
    rng = np.random.default_rng(7)
    qi = rng.choice(len(ds), size=n_query, replace=False).tolist()
    texts = [ds[i]["text"] for i in qi]
    q_ids = [ds[i]["item_ID"] for i in qi]

    # Encode the full gallery (could be large; we trim to first 5K + queries
    # for speed during in-loop eval).
    gallery_idx = list(range(min(5000, len(ds))))
    for i in qi:
        if i not in gallery_idx:
            gallery_idx.append(i)
    g_imgs = [ds[i]["image"] for i in gallery_idx]
    g_ids = [ds[i]["item_ID"] for i in gallery_idx]

    model.eval()
    img_feats = []
    for s in range(0, len(g_imgs), batch_size):
        batch = g_imgs[s:s + batch_size]
        t = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        f = model.encode_image(t)
        if device == "mps":
            f = f.float()
        img_feats.append(F.normalize(f, p=2, dim=-1).cpu())
    img_feats = torch.cat(img_feats, 0)

    txt_feats = []
    for s in range(0, len(texts), batch_size * 2):
        batch = texts[s:s + batch_size * 2]
        toks = tokenizer(batch).to(device)
        f = model.encode_text(toks)
        if device == "mps":
            f = f.float()
        txt_feats.append(F.normalize(f, p=2, dim=-1).cpu())
    txt_feats = torch.cat(txt_feats, 0)

    sims = txt_feats @ img_feats.T
    g_id_arr = np.asarray(g_ids)
    out = {}
    for k in (1, 5, 10):
        topk = sims.topk(k, dim=1).indices.numpy()
        hits = sum(1 for i, q in enumerate(q_ids)
                    if q in g_id_arr[topk[i]])
        out[f"recall@{k}"] = round(100 * hits / len(q_ids), 2)
    model.train()
    return out


@torch.no_grad()
def quick_lookbench_at_dim(model, preprocess, device, dim, subset, max_query=300,
                            batch_size=32):
    lb_root = REPO / "data/raw/lookbench/datasets"
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
    idx = sims.argmax(1).tolist()
    q_lab = [(str(x.get("category", "")).lower().strip(),
              f"{str(x.get('category', '')).lower().strip()}|"
              f"{str(x.get('main_attribute', '')).lower().strip()}")
              for x in qs]
    g_lab = [(str(x.get("category", "")).lower().strip(),
              f"{str(x.get('category', '')).lower().strip()}|"
              f"{str(x.get('main_attribute', '')).lower().strip()}")
              for x in gs]
    fine = sum(1 for i, t in enumerate(idx) if q_lab[i][1] == g_lab[t][1])
    coarse = sum(1 for i, t in enumerate(idx) if q_lab[i][0] == g_lab[t][0])
    return {
        "fine_r_at_1": round(100 * fine / max(1, len(q_lab)), 2),
        "coarse_r_at_1": round(100 * coarse / max(1, len(q_lab)), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/moda-siglip-matryoshka/best/model_state_dict.pt",
                    help="Init checkpoint (relative to repo root). Falls back to "
                         "Recipe A' distilled if matryoshka not yet trained.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr-vision", type=float, default=5e-6)
    ap.add_argument("--lr-text", type=float, default=1e-5,
                    help="Text-tower LR (typically larger than vision since it's "
                         "starting from stock and we want it to actually move).")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--infonce-weight", type=float, default=5.0)
    ap.add_argument("--text-drift-weight", type=float, default=0.05)
    ap.add_argument("--vis-drift-weight", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[64, 128, 256, 384, 512, 768])
    ap.add_argument("--slice-weights", type=float, nargs="+", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="+", default=["df_multimodal"],
                    help="Only df_multimodal has natural captions. df_inshop "
                         "uses synthetic 'a photo of {cat} in {color}' if added.")
    args = ap.parse_args()

    if args.slice_weights is None:
        args.slice_weights = [1.0] * len(args.slices)

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("device=%s", device)
    log.info("slices=%s slice_weights=%s", args.slices, args.slice_weights)
    log.info("init=%s", args.init)

    log.info("Loading FashionSigLIP arch ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    init_path = REPO / args.init
    if not init_path.exists():
        fallback = REPO / "models/moda-siglip-distilled/best/model_state_dict.pt"
        log.warning("init checkpoint not found, falling back to %s", fallback)
        init_path = fallback
    sd = torch.load(init_path, map_location="cpu", weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    log.info("loaded init: missing=%d unexpected=%d", len(miss), len(unexp))
    model = model.to(device)

    pretrained_vis = {k: v.detach().clone().to(device)
                      for k, v in model.named_parameters()
                      if k.startswith("visual.")}
    pretrained_text = {k: v.detach().clone().to(device)
                       for k, v in model.named_parameters()
                       if not k.startswith("visual.")}

    dsets = [TextVisionDistillDataset(preprocess, k) for k in args.datasets]
    if len(dsets) == 1:
        full = dsets[0]
    else:
        from torch.utils.data import ConcatDataset
        full = ConcatDataset(dsets)
    log.info("Total training pairs: %d", len(full))

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

        for images, tokens, teacher in tqdm(loader, desc=f"y-ep{epoch+1}"):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher = F.normalize(teacher, p=2, dim=-1)

            s_full_img = model.encode_image(images)
            s_full_txt = model.encode_text(tokens)
            if device == "mps":
                s_full_img = s_full_img.float()
                s_full_txt = s_full_txt.float()

            l_vis_mrl, per_slice = matryoshka_vision_loss(
                s_full_img, teacher, args.slices, args.slice_weights,
                args.rkd_weight, args.sim_weight)
            l_cross = cross_modal_infonce(s_full_img, s_full_txt,
                                           temperature=args.temperature)
            l_vis_drift = drift_reg(model, pretrained_vis, "visual.")
            # Text drift: pretrained_text already excludes "visual.", so
            # any non-visual key matches; we identify "non-visual" by the
            # negation (handled by dict membership check).
            l_text_drift = drift_reg(model, pretrained_text, "")
            # The drift_reg with empty prefix counts BOTH towers — subtract
            # the visual portion to leave text-only.
            l_text_drift = l_text_drift - l_vis_drift

            loss = (l_vis_mrl
                    + args.infonce_weight * l_cross
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
                        f"vis_drift={l_vis_drift.item():.3e} "
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
                # Composite score: mean of (text->image R@1, vision Fine R@1 at full dim)
                composite = (txt2img["recall@1"] + lb_full["fine_r_at_1"]) / 2
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
                            "training": "Recipe Y joint text+vision distillation",
                            "init_from": args.init,
                            "step": step,
                            "epoch": epoch,
                            "composite": composite,
                            "text2img_eval": txt2img,
                            "lookbench_full": lb_full,
                            "lookbench_64": lb_64,
                        }, f, indent=2)
                    log.info("  *** new best composite=%.2f ***", composite)

        ckpt = OUT_DIR / f"epoch_{epoch + 1}"
        ckpt.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt / "model_state_dict.pt")

    elapsed = time.time() - t0
    log.info("Recipe Y complete in %.1f min. Best composite = %.2f",
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
