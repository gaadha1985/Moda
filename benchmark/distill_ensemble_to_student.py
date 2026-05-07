"""
Recipe A' - Step 2: Distill a frozen ensemble teacher (2048-d) into a
single 768-d student by Relational Knowledge Distillation (RKD-D).

Target: a single 768-d embedding model that matches or beats the
ensemble on LookBench Fine R@1.

Key design choices:
---------------------------------------------------------------------------
Student:   CustomTextCLIP @ 768-d image features.
           Initialized from MODA-SigLIP-DeepFashion2 (our best single model).
Teacher:   Frozen 2048-d ensemble (FashionSigLIP + FashionCLIP + MODA-DF2),
           pre-computed by cache_teacher_embeddings.py.
Loss:      RKD-Distance (Park et al. 2019) on batch pairwise distance
           matrices. Since teacher and student dims differ, we compare
           *distance matrices* (dim-invariant) rather than embeddings
           directly. Optional add-ons:
             - Cosine-of-cosine loss (preserve pairwise similarity ordering)
             - Weight drift regularization vs pretrained student
             - Small InfoNCE term on DF2 cross-domain pairs (preserve
               retrieval discriminability)
---------------------------------------------------------------------------

Usage (typical run):
    python benchmark/distill_ensemble_to_student.py \
        --epochs 3 --batch-size 128 --lr 1e-5 \
        --datasets df_inshop df_multimodal hnm \
        --drift-weight 0.01 --rkd-weight 25.0 --sim-weight 10.0

Periodically evaluates the student against LookBench using the same
protocol as benchmark/eval_lookbench_baseline.py (Fine R@1 overall).
"""

from __future__ import annotations

import argparse
import copy
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
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
TEACHER_CACHE = REPO / "results/distillation/teacher_cache"
OUT_DIR = REPO / "models/moda-siglip-distilled"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "distill.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training dataset (image + cached teacher embedding)
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """Yields (student-preprocessed image tensor, teacher 2048-d embedding).

    Memory-efficient: HuggingFace Datasets are kept mmap-backed; images are
    decoded on-demand in __getitem__. H&M rows are stored as lightweight
    path strings, not decoded PIL objects.
    """

    def __init__(self, dataset_key: str, preprocess, hnm_limit: int = 30000):
        self.key = dataset_key
        self.preprocess = preprocess

        emb_path = TEACHER_CACHE / f"{dataset_key}_teacher_2048.npy"
        ids_path = TEACHER_CACHE / f"{dataset_key}_ids.json"
        self.teacher = np.load(emb_path, mmap_mode="r")
        self.ids = json.load(open(ids_path))["ids"]
        assert len(self.teacher) == len(self.ids), (len(self.teacher), len(self.ids))

        self._hf_ds = None
        self._hnm_paths = None
        self._setup_source(dataset_key, hnm_limit)

    def _setup_source(self, key: str, hnm_limit: int):
        if key == "df_inshop":
            self._hf_ds = load_dataset(
                "Marqo/deepfashion-inshop",
                cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]
        elif key == "df_multimodal":
            self._hf_ds = load_dataset(
                "Marqo/deepfashion-multimodal",
                cache_dir=str(REPO / "data/raw/deepfashion_multimodal"))["data"]
        elif key == "hnm":
            df = pd.read_csv(REPO / "data/raw/hnm/articles.csv")
            img_root = REPO / "data/raw/hnm_images"
            df["_path"] = df["article_id"].apply(
                lambda a: img_root / f"{int(a):010d}"[:3] / f"{int(a):010d}.jpg")
            df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
            if hnm_limit and len(df) > hnm_limit:
                df = df.sample(n=hnm_limit, random_state=42).reset_index(drop=True)
            self._hnm_paths = [str(p) for p in df["_path"].tolist()]
        else:
            raise ValueError(key)

        n_src = len(self._hnm_paths) if self._hnm_paths is not None else len(self._hf_ds)
        if n_src != len(self.teacher):
            n = min(n_src, len(self.teacher))
            log.warning("  %s: size mismatch source=%d teacher=%d -> truncating to %d",
                        key, n_src, len(self.teacher), n)
            if self._hnm_paths is not None:
                self._hnm_paths = self._hnm_paths[:n]
            else:
                self._hf_ds = self._hf_ds.select(range(n))
            self.teacher = self.teacher[:n]
            self.ids = self.ids[:n]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.key == "hnm":
            img = Image.open(self._hnm_paths[idx]).convert("RGB")
        else:
            img = self._hf_ds[idx]["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
        tensor = self.preprocess(img)
        teacher = torch.from_numpy(np.array(self.teacher[idx])).float()
        return tensor, teacher


class MultiDistillDataset(Dataset):
    """Concatenation of per-dataset DistillDatasets."""
    def __init__(self, datasets: list[DistillDataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum = np.cumsum(self.lengths)
        self.total = int(self.cum[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        d_idx = int(np.searchsorted(self.cum, idx, side="right"))
        offset = idx - (self.cum[d_idx - 1] if d_idx > 0 else 0)
        return self.datasets[d_idx][offset]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    """Pairwise L2 distance matrix without torch.cdist.

    Avoids `aten::_cdist_backward` which is not implemented for MPS
    in recent PyTorch versions. Mathematically identical to
    torch.cdist(x, x, p=2).
    """
    sq = (x * x).sum(dim=-1, keepdim=True)
    d2 = sq + sq.T - 2.0 * (x @ x.T)
    return d2.clamp_min(0.0).add(1e-12).sqrt()


def rkd_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """RKD-D: preserve *relative* pairwise distances between samples in a batch.

    Normalizes both distance matrices by their own mean (so scale/dim
    differences between 768-d student and 2048-d teacher don't matter),
    then Huber-loss between them.
    """
    with torch.no_grad():
        d_t = _pairwise_l2(teacher)
        mean_t = d_t[d_t > 0].mean()
        d_t = d_t / (mean_t + 1e-12)

    d_s = _pairwise_l2(student)
    mean_s = d_s[d_s > 0].mean()
    d_s = d_s / (mean_s + 1e-12)

    return F.smooth_l1_loss(d_s, d_t)


def similarity_mimicry_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between pairwise cosine similarity matrices (both L2-normalized).

    Student and teacher are L2-normalized before this is called, so
    similarity = dot product. Scale-invariant; works across dims.
    """
    s_sim = student @ student.T
    t_sim = teacher @ teacher.T
    return F.mse_loss(s_sim, t_sim)


def drift_reg(model: torch.nn.Module, pretrained: dict[str, torch.Tensor]) -> torch.Tensor:
    """L2 drift from pretrained visual weights (prevents catastrophic forgetting)."""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for name, p in model.named_parameters():
        if not name.startswith("visual."):
            continue
        if name in pretrained and p.requires_grad:
            total = total + ((p - pretrained[name]) ** 2).sum()
            n += p.numel()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# LookBench evaluation (subsampled, for fast progress tracking during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_lookbench_eval(model, preprocess, device: str,
                         subsets=("real_studio_flat",),
                         max_query: int = 500,
                         batch_size: int = 64) -> dict:
    """Lightweight eval: Fine R@1 on a handful of LookBench subsets."""
    model.eval()
    results = {}
    lb_root = REPO / "data/raw/lookbench/datasets"
    for subset in subsets:
        dsd = load_from_disk(str(lb_root / subset))
        qs = dsd["query"].select(range(min(max_query, len(dsd["query"]))))
        gs = dsd["gallery"]

        def enc(split):
            feats = []
            for s in range(0, len(split), batch_size):
                batch = split[s:s + batch_size]
                t = torch.stack([preprocess(im.convert("RGB"))
                                 for im in batch["image"]]).to(device)
                f = model.encode_image(t).float() if device == "mps" \
                    else model.encode_image(t)
                feats.append(F.normalize(f, p=2, dim=-1).cpu())
            return torch.cat(feats, 0)

        qf = enc(qs)
        gf = enc(gs)

        def labels(split):
            return [(str(x.get("category", "")).lower().strip(),
                     f"{str(x.get('category', '')).lower().strip()}|"
                     f"{str(x.get('main_attribute', '')).lower().strip()}")
                    for x in split]

        q_lab = labels(qs)
        g_lab = labels(gs)
        sims = qf @ gf.T
        idx = sims.argmax(dim=1).tolist()
        n_hit_fine = sum(1 for i, top in enumerate(idx) if q_lab[i][1] == g_lab[top][1])
        n_hit_coarse = sum(1 for i, top in enumerate(idx) if q_lab[i][0] == g_lab[top][0])
        results[subset] = {
            "n_query": len(q_lab),
            "fine_r_at_1": round(100 * n_hit_fine / max(1, len(q_lab)), 2),
            "coarse_r_at_1": round(100 * n_hit_coarse / max(1, len(q_lab)), 2),
        }
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["df_inshop", "df_multimodal", "hnm"])
    ap.add_argument("--hnm-limit", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rkd-weight", type=float, default=25.0)
    ap.add_argument("--sim-weight", type=float, default=10.0)
    ap.add_argument("--drift-weight", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="Run quick LookBench eval every N batches (0 to disable)")
    ap.add_argument("--eval-subsets", nargs="+",
                    default=["real_studio_flat", "real_streetlook"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", default="models/moda-siglip-deepfashion2/best/model_state_dict.pt",
                    help="Student init checkpoint (relative to repo)")
    args = ap.parse_args()

    device = (args.device
              or ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu")))
    log.info("Using device: %s", device)

    log.info("Loading student from SigLIP arch + init=%s", args.init)
    student, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP")
    init_path = REPO / args.init
    if init_path.exists():
        sd = torch.load(init_path, map_location="cpu", weights_only=True)
        student.load_state_dict(sd, strict=False)
        log.info("  Loaded init weights (%d keys)", len(sd))
    else:
        log.warning("  init checkpoint not found, starting from HF weights")
    student = student.to(device)

    pretrained_vis = {k: v.detach().clone().to(device)
                      for k, v in student.named_parameters() if k.startswith("visual.")}

    dsets = [DistillDataset(k, preprocess, hnm_limit=args.hnm_limit)
             for k in args.datasets]
    full = MultiDistillDataset(dsets)
    log.info("Total training images across %d datasets: %d",
             len(dsets), len(full))

    loader = DataLoader(full, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"),
                        drop_last=True)

    optim_params = [p for n, p in student.named_parameters() if n.startswith("visual.")]
    optim = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = None
    if device == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_fine_r1 = 0.0
    step = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        log.info("=" * 70)
        log.info("EPOCH %d/%d", epoch + 1, args.epochs)
        log.info("=" * 70)

        for images, teacher in tqdm(loader, desc=f"epoch{epoch+1}"):
            images = images.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            teacher = F.normalize(teacher, p=2, dim=-1)

            if device == "cuda":
                with torch.cuda.amp.autocast():
                    s = student.encode_image(images)
                    s = F.normalize(s, p=2, dim=-1)
                    l_rkd = rkd_distance_loss(s, teacher)
                    l_sim = similarity_mimicry_loss(s, teacher)
                    l_drift = drift_reg(student, pretrained_vis)
                    loss = (args.rkd_weight * l_rkd +
                            args.sim_weight * l_sim +
                            args.drift_weight * l_drift)
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                scaler.step(optim); scaler.update()
            else:
                s = student.encode_image(images)
                if device == "mps":
                    s = s.float()
                s = F.normalize(s, p=2, dim=-1)
                l_rkd = rkd_distance_loss(s, teacher)
                l_sim = similarity_mimicry_loss(s, teacher)
                l_drift = drift_reg(student, pretrained_vis)
                loss = (args.rkd_weight * l_rkd +
                        args.sim_weight * l_sim +
                        args.drift_weight * l_drift)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                optim.step()

            step += 1
            if step % 50 == 0:
                log.info("  step=%d loss=%.4f (rkd=%.4f sim=%.4f drift=%.4e)",
                         step, loss.item(), l_rkd.item(), l_sim.item(), l_drift.item())

            if args.eval_every > 0 and step % args.eval_every == 0:
                quick = quick_lookbench_eval(student, preprocess, device,
                                              subsets=args.eval_subsets)
                overall_fine = np.mean([v["fine_r_at_1"] for v in quick.values()])
                log.info("  [eval @ step %d] %s  mean_fine_r@1=%.2f",
                         step, quick, overall_fine)
                history.append({"step": step, "eval": quick, "mean_fine_r1": overall_fine})
                if overall_fine > best_fine_r1:
                    best_fine_r1 = overall_fine
                    best_path = OUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save(student.state_dict(),
                               best_path / "model_state_dict.pt")
                    with open(best_path / "meta.json", "w") as f:
                        json.dump({
                            "base_model": "Marqo/marqo-fashionSigLIP",
                            "init_from": args.init,
                            "training": ("Recipe A' RKD-D + sim-mimicry + drift-reg "
                                         "on ensemble teacher (FashionSigLIP + "
                                         "FashionCLIP + MODA-SigLIP-DF2)"),
                            "step": step,
                            "epoch": epoch,
                            "mean_fine_r1_subset_eval": overall_fine,
                            "per_subset": quick,
                        }, f, indent=2)
                    log.info("  Saved new best to %s", best_path)

        ckpt = OUT_DIR / f"epoch_{epoch + 1}"
        ckpt.mkdir(exist_ok=True)
        torch.save(student.state_dict(), ckpt / "model_state_dict.pt")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("Training complete in %.1f min. Best subset mean Fine R@1 = %.2f",
             elapsed / 60, best_fine_r1)
    log.info("Best checkpoint: %s/best/model_state_dict.pt", OUT_DIR)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "best_mean_fine_r1": best_fine_r1,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
