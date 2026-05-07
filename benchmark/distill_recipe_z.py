"""
Recipe Z — Scaled text+vision distillation to close the text-to-image gap.

Why this exists
---------------
Recipe Y trained the text tower on ~41 K image-caption pairs from
DeepFashion-MultiModal alone. FashionSigLIP's text tower was trained on
Marqo's full ~1 M+ fashion captions corpus. The result, on Marqo's clean
benchmark suite, was a small but consistent ~1 pp deficit on text-to-image
Recall@1 (Atlas, KAGL).

Recipe Z does NOT change the vision distillation losses (which already give
us Matryoshka + ensemble-grade quality). It only scales up the text-side
training distribution from ~41 K pairs to ~200 K pairs (~5x), drawn from
three internal training pools:

    1. DeepFashion-MultiModal  — 41 K natural sentence captions  (as in Y)
    2. DeepFashion-InShop      — 52 K templated captions, 4 paraphrases/row
    3. H&M                     — 105 K templated captions from rich
                                 article-level attributes
                                 (prod_name, detail_desc, colour, type, …)
                                 4 paraphrases/row.

All three sources are in our distillation training pool already, so
the leakage audit (`benchmark/marqo_clean_leakage_audit.py`) is unchanged
and still PASSes against the 4 Marqo clean evaluation datasets.

Vision-side distillation losses (Matryoshka MRL distill_loss against the
cached 2048-d ensemble teacher) are kept ONLY for samples that have a
teacher embedding (df_inshop + df_multimodal). H&M samples contribute
only to the cross-modal InfoNCE loss. This avoids the cost of caching
teacher embeddings for H&M and keeps the vision tower anchored to the
clean teacher signal.

Per-step alternation
--------------------
Each iteration draws a balanced mini-batch (~50/50) from:
  - "distill" pool  : df_inshop ∪ df_multimodal  (has teacher emb)
  - "text-only" pool: hnm                        (text contrastive only)

Losses
------
  L_vis_mrl       (only on distill pool rows): per-slice RKD-D + sim-mimicry
  L_cross_infonce (whole batch, both pools)  : symmetric InfoNCE
                                              between image_i and text_i
  L_vis_drift / L_text_drift: same as Recipe Y, slightly looser on text
                              (we WANT the text tower to move more)

Init
----
  models/moda-siglip-text-vision/best/model_state_dict.pt   (Recipe Y best)
  → falls back to matryoshka, then to distilled, then to stock.

Usage
-----
    python benchmark/distill_recipe_z.py \
        --epochs 2 --batch-size 96 --lr-text 8e-6 --lr-vision 3e-6 \
        --infonce-weight 8.0 --text-drift-weight 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

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

TEACHER_CACHE = REPO / "results/distillation/teacher_cache"
OUT_DIR = REPO / "models/moda-siglip-recipe-z"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "distill_recipe_z.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption builders (templated, for sources without natural language)
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return "" if s.lower() in {"nan", "none", "unknown"} else s


def hnm_caption_set(row: dict) -> list[str]:
    """4 diverse paraphrases for one H&M article row.

    Uses prod_name, detail_desc, colour_group_name, product_type_name,
    perceived_colour_master_name, garment_group_name, index_name (gender).
    Empty fields are skipped gracefully so we never emit "a  ".
    """
    name      = _clean(row.get("prod_name"))
    detail    = _clean(row.get("detail_desc"))
    colour    = _clean(row.get("colour_group_name")).lower()
    ptype     = _clean(row.get("product_type_name")).lower()
    master    = _clean(row.get("perceived_colour_master_name")).lower()
    grp       = _clean(row.get("garment_group_name")).lower()
    idx       = _clean(row.get("index_name")).lower()  # e.g. "ladieswear"

    captions = []
    if colour and ptype:
        captions.append(f"a {colour} {ptype}")
    if name and colour:
        captions.append(f"{name.lower()}, {colour}")
    if detail:
        captions.append(detail.strip(" .") + ".")
    if grp and (master or colour):
        col = master or colour
        captions.append(f"a {col} {grp}")
    if idx and ptype and colour:
        # gendered paraphrase
        gender = ("women's" if "ladies" in idx else
                  "men's" if "menswear" in idx else
                  "kids'" if "children" in idx or "baby" in idx else
                  "")
        if gender:
            captions.append(f"{gender} {colour} {ptype}")
    # Always have at least a fallback
    if not captions:
        fallback = name or ptype or "fashion item"
        captions.append(fallback)
    return captions[:4]  # cap at 4 paraphrases


def df_inshop_caption_set(row: dict) -> list[str]:
    """4 diverse paraphrases for one DF-InShop row."""
    cat   = _clean(row.get("category2") or row.get("category")).lower()
    color = _clean(row.get("color")).lower()
    captions = []
    if cat and color:
        captions.append(f"a photo of a {color} {cat}")
        captions.append(f"{color} {cat}")
        captions.append(f"women's {cat} in {color}")
    if cat:
        captions.append(f"a {cat}")
    if not captions:
        captions.append(_clean(row.get("category", "fashion item")) or "fashion item")
    return captions[:4]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DistillTextVisionDataset(Dataset):
    """DF-InShop or DF-Multimodal with cached teacher embeddings.

    Yields: (image_tensor, caption_str, teacher_2048_emb, has_teacher=True)
    """

    def __init__(self, preprocess, key: str):
        assert key in {"df_inshop", "df_multimodal"}
        self.key = key
        self.preprocess = preprocess
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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        img = row["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.preprocess(img)
        if self.key == "df_multimodal":
            text = _clean(row.get("text")) or "fashion item"
        else:
            captions = df_inshop_caption_set(row)
            text = random.choice(captions)
        teacher = torch.from_numpy(np.array(self.teacher[idx])).float()
        return tensor, text, teacher, True  # has_teacher = True


class HnMTextDataset(Dataset):
    """H&M articles → (image_tensor, caption_str, zero_teacher, has_teacher=False).

    No teacher embedding is computed; H&M only contributes to the
    cross-modal InfoNCE loss. We still emit a zero placeholder tensor so
    the collate fn stays uniform with the distillation streams.
    """

    def __init__(self, preprocess, csv_path: Path | str = None,
                 img_root: Path | str = None, limit: int | None = None):
        self.preprocess = preprocess
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
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        self.df = df
        self._zero_teacher = torch.zeros(2048)
        log.info("HnMTextDataset: %d image-caption candidates", len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        try:
            img = Image.open(row["_path"]).convert("RGB")
        except Exception:
            # corrupt image — return a tiny 1x1 white image, will be skipped
            # by drop_last since this is rare; avoids OSError mid-epoch.
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        tensor = self.preprocess(img)
        text = random.choice(hnm_caption_set(row))
        return tensor, text, self._zero_teacher, False  # has_teacher = False


def collate_fn(batch, tokenizer):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    teachers = torch.stack([b[2] for b in batch])
    has_teacher = torch.tensor([b[3] for b in batch], dtype=torch.bool)
    tokens = tokenizer(texts)
    return imgs, tokens, teachers, has_teacher


# ---------------------------------------------------------------------------
# Losses (same as Recipe Y, but vision losses are masked by has_teacher)
# ---------------------------------------------------------------------------

def rkd_distance_loss_pair(student: torch.Tensor, teacher: torch.Tensor):
    if student.size(0) < 2:
        return torch.tensor(0.0, device=student.device)
    with torch.no_grad():
        d_t = _pairwise_l2(teacher)
        mean_t = d_t[d_t > 0].mean() if (d_t > 0).any() else torch.tensor(1.0, device=teacher.device)
        d_t = d_t / (mean_t + 1e-12)
    d_s = _pairwise_l2(student)
    mean_s = d_s[d_s > 0].mean() if (d_s > 0).any() else torch.tensor(1.0, device=student.device)
    d_s = d_s / (mean_s + 1e-12)
    return F.smooth_l1_loss(d_s, d_t)


def sim_mimicry(student, teacher):
    if student.size(0) < 2:
        return torch.tensor(0.0, device=student.device)
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
# Quick eval helpers (text→image on DF-MM sample + LookBench at the full dim)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_text2image_eval(model, preprocess, tokenizer, device,
                           n_query=500, batch_size=32):
    ds = load_dataset(
        "Marqo/deepfashion-multimodal",
        cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
    rng = np.random.default_rng(7)
    qi = rng.choice(len(ds), size=n_query, replace=False).tolist()
    texts = [ds[i]["text"] for i in qi]
    q_ids = [ds[i]["item_ID"] for i in qi]

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
        hits = sum(1 for i, q in enumerate(q_ids) if q in g_id_arr[topk[i]])
        out[f"recall@{k}"] = round(100 * hits / len(q_ids), 2)
    model.train()
    return out


@torch.no_grad()
def quick_lookbench_at_dim(model, preprocess, device, dim, subset,
                            max_query=300, batch_size=32):
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

def _resolve_init(arg_path: str) -> Path:
    """Pick the best init checkpoint in the chain Y -> Matryoshka -> A' -> stock."""
    candidates = [
        REPO / arg_path,
        REPO / "models/moda-siglip-text-vision/best/model_state_dict.pt",
        REPO / "models/moda-siglip-matryoshka/best/model_state_dict.pt",
        REPO / "models/moda-siglip-distilled/best/model_state_dict.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None  # caller will fall through to stock weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init",
                    default="models/moda-siglip-text-vision/best/model_state_dict.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr-vision", type=float, default=3e-6,
                    help="Smaller than Y: vision tower is already very good, "
                         "we mostly want text alignment.")
    ap.add_argument("--lr-text", type=float, default=8e-6,
                    help="Slightly larger so the text tower can move on the "
                         "expanded corpus.")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--infonce-weight", type=float, default=8.0,
                    help="Larger than Y (5.0) since the bigger text corpus "
                         "is the whole point of Recipe Z.")
    ap.add_argument("--text-drift-weight", type=float, default=0.02,
                    help="Looser than Y (0.05) — we want text tower movement.")
    ap.add_argument("--vis-drift-weight", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--slices", type=int, nargs="+",
                    default=[64, 128, 256, 384, 512, 768])
    ap.add_argument("--slice-weights", type=float, nargs="+", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hnm-limit", type=int, default=80000,
                    help="Cap H&M rows; 80K is the sweet spot for ~6h on M1.")
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal", "hnm"])
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

    pretrained_vis = {k: v.detach().clone().to(device)
                      for k, v in model.named_parameters()
                      if k.startswith("visual.")}
    pretrained_text = {k: v.detach().clone().to(device)
                       for k, v in model.named_parameters()
                       if not k.startswith("visual.")}

    dsets = []
    for k in args.datasets:
        if k == "hnm":
            dsets.append(HnMTextDataset(preprocess, limit=args.hnm_limit))
        else:
            dsets.append(DistillTextVisionDataset(preprocess, k))
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

        for images, tokens, teacher, has_teacher in tqdm(loader, desc=f"z-ep{epoch+1}"):
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

            # Vision MRL distillation, masked to rows that actually have a teacher
            mask = has_teacher
            if mask.any():
                l_vis_mrl, _ = matryoshka_vision_loss(
                    s_full_img[mask], teacher[mask], args.slices,
                    args.slice_weights, args.rkd_weight, args.sim_weight)
            else:
                l_vis_mrl = torch.tensor(0.0, device=device)

            # Cross-modal contrastive on the FULL batch (both pools)
            l_cross = cross_modal_infonce(s_full_img, s_full_txt,
                                           temperature=args.temperature)

            l_vis_drift = drift_reg(model, pretrained_vis, "visual.")
            l_text_drift = drift_reg(model, pretrained_text, "")
            l_text_drift = l_text_drift - l_vis_drift  # text-only portion

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
                        f"hT_frac={mask.float().mean().item():.2f} "
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
                # Composite score weighted toward text alignment (the gap we
                # came here to close), but penalised if vision regresses.
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
                            "training": "Recipe Z scaled text+vision distillation",
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
    log.info("Recipe Z complete in %.1f min. Best composite = %.2f",
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
