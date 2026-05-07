"""
MODA Phase 6.Z+ — LLM-paraphrased fashion captions for text-tower scaling.

Why this exists
---------------
Recipe Y (41 K natural sentences) and Recipe Z (175 K mostly templated)
both narrowed but did not close the text-to-image gap to FashionSigLIP on
the Marqo clean benchmarks. Diagnosis (see EXPERIMENT_LOG.md §6.Z):

    * Volume: FashionSigLIP saw ~1 M+ proprietary natural fashion
      captions during pretraining; we have a fraction of that.
    * Style:  our DF-InShop / H&M captions are short noun-phrase
      templates ("a black vest top"), while the actual eval queries
      (Atlas / KAGL / Polyvore) are retail product titles
      ("ZARA ribbed knit midi dress in slate blue").

This script fixes the *style* axis and dramatically scales *volume*: for
each H&M article and each DF-InShop row, we ask an LLM (gpt-4o-mini via
PaleblueDot) to generate N retail-style paraphrases conditioned on the
structured attributes we already have. Output is a JSONL file per
source, consumable by `benchmark/distill_recipe_z_plus.py`.

Usage
-----
  export PALEBLUEDOT_API_KEY=<key>     # also auto-loaded from .env
  python benchmark/generate_llm_captions.py --source hnm
  python benchmark/generate_llm_captions.py --source df_inshop
  python benchmark/generate_llm_captions.py --source hnm --max-rows 500    # smoke

Output
------
  data/processed/llm_captions/hnm.jsonl
  data/processed/llm_captions/df_inshop.jsonl

Schema (one JSON object per line):
  {
    "source":   "hnm" | "df_inshop",
    "key":      "<article_id>"  for hnm
                "<row_index>"   for df_inshop  (matches HF dataset order)
    "captions": ["..", "..", ...],            # N retail-style paraphrases
    "model":    "openai/gpt-4o-mini",
    "ts":       <unix>,
  }

Resumable: re-running skips keys already present in the output file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import AsyncOpenAI

OUT_DIR = REPO / "data/processed/llm_captions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "generate_llm_captions.log")],
)
log = logging.getLogger(__name__)

BASE_URL = "https://open.palebluedot.ai/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_CONCURRENT = 30
MAX_RETRIES = 4

ENV_PATH = REPO / ".env"


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    key = os.environ.get("PALEBLUEDOT_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("PALEBLUEDOT_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise ValueError(
        "PALEBLUEDOT_API_KEY not found. Set env var or add to .env. "
        "Get from https://palebluedot.ai")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert e-commerce copywriter creating short, retail-style "
    "fashion product titles for a search index. You always reply with VALID "
    "JSON only — no commentary, no markdown, no code fences."
)

USER_TEMPLATE = """Generate {n} DISTINCT retail product titles for the fashion item described by these attributes:

{attrs_block}

STRICT RULES:
1. Each title must be 4-15 words long.
2. Each title must be in the style of a real e-commerce product page (think Zara, ASOS, H&M, Net-a-Porter, Nordstrom, Asos, Shopbop), e.g.
     "ribbed-knit midi dress in stone"
     "high-waisted wide-leg cropped trousers"
     "men's cotton-jersey crew-neck t-shirt, navy"
     "floral-print wrap mini dress with tie waist"
3. Vary syntax across the {n} titles: some must lead with the colour, some
   with the material, some with the silhouette/cut, some with the gender.
4. Keep ALL the attributes that are factually given (colour, type, gender,
   pattern, material). NEVER invent attributes that are not in the input
   (no fake brand names, no fake fabrics, no fake silhouettes).
5. NO duplicates. NO hashtags. NO emoji. NO sales copy ("perfect for...",
   "must-have"). Just the bare product title.
6. Lower-case unless the attribute is clearly a proper noun.

Reply with JSON only:
{{"titles": ["title 1", "title 2", ...]}}
"""


def _attrs_to_block(attrs: dict) -> str:
    lines = []
    for k, v in attrs.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "unknown"}:
            continue
        lines.append(f"  - {k}: {s}")
    return "\n".join(lines) if lines else "  - (no attributes given)"


def _extract_titles(text: str) -> list[str]:
    """Robust JSON / list extraction from LLM reply."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return []
    titles = obj.get("titles") or obj.get("captions") or obj.get("list") or []
    if not isinstance(titles, list):
        return []
    out = []
    seen = set()
    for t in titles:
        if not isinstance(t, str):
            continue
        t = t.strip().strip('"').strip("'")
        t = re.sub(r"\s+", " ", t)
        if 3 <= len(t.split()) <= 25 and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Source-specific attribute extractors
# ---------------------------------------------------------------------------

def _safe_str(v) -> str:
    """NaN-safe string coercion (pandas dumps NaN as float)."""
    if v is None:
        return ""
    try:
        if isinstance(v, float):
            import math
            if math.isnan(v):
                return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "unknown"} else s


def hnm_row_to_attrs(row: dict) -> dict:
    return {
        "product_name":      _safe_str(row.get("prod_name")),
        "product_type":      _safe_str(row.get("product_type_name")),
        "garment_group":     _safe_str(row.get("garment_group_name")),
        "colour":            _safe_str(row.get("colour_group_name")),
        "perceived_colour":  _safe_str(row.get("perceived_colour_master_name")),
        "graphical_pattern": _safe_str(row.get("graphical_appearance_name")),
        "section":           _safe_str(row.get("section_name")),
        "department":        _safe_str(row.get("department_name")),
        "index":             _safe_str(row.get("index_name")),
        "detail_desc":       _safe_str(row.get("detail_desc"))[:280],
    }


def df_inshop_row_to_attrs(row: dict) -> dict:
    return {
        "category":       _safe_str(row.get("category")),
        "category_fine":  _safe_str(row.get("category2")),
        "color":          _safe_str(row.get("color")),
        "gender":         _safe_str(row.get("gender")) or "women",
        "item_id":        _safe_str(row.get("item_ID")),
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def caption_one(client: AsyncOpenAI, source: str, key: str,
                       attrs: dict, n: int, model: str,
                       semaphore: asyncio.Semaphore) -> Optional[dict]:
    prompt = USER_TEMPLATE.format(n=n, attrs_block=_attrs_to_block(attrs))
    use_json_mode = any(k in model for k in ("gpt-4o", "gpt-5"))

    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.8,
                max_tokens=400,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            async with semaphore:
                resp = await client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content
            titles = _extract_titles(content or "")
            if len(titles) < max(2, n // 2):
                raise ValueError(
                    f"Only got {len(titles)} usable titles (need >= {max(2, n//2)})")
            return {
                "source":   source,
                "key":      str(key),
                "captions": titles[:n],
                "model":    model,
                "ts":       int(time.time()),
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                log.warning("Failed after %d retries: %s/%s — %s",
                            MAX_RETRIES, source, key, e)
                return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_done_keys(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    keys = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys.add(str(obj["key"]))
            except Exception:
                continue
    return keys


async def run(source: str, items: list[tuple[str, dict]], n: int, model: str,
                concurrency: int, out_path: Path, chunk_size: int = 200):
    api_key = _load_api_key()
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    done = _load_done_keys(out_path)
    todo = [(k, a) for k, a in items if str(k) not in done]
    log.info("[%s] total=%d  done=%d  todo=%d  -> %s",
             source, len(items), len(done), len(todo), out_path)

    if not todo:
        log.info("[%s] all keys already captioned", source)
        return

    t0 = time.time()
    completed = 0
    failed = 0

    for cs in range(0, len(todo), chunk_size):
        chunk = todo[cs:cs + chunk_size]
        tasks = [caption_one(client, source, k, a, n, model, semaphore)
                 for k, a in chunk]
        results = await asyncio.gather(*tasks)

        with open(out_path, "a") as f:
            for r in results:
                if r is None:
                    failed += 1
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                completed += 1

        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (len(todo) - completed - failed) / rate if rate > 0 else 0
        log.info(
            "[%s] %d/%d done (%.1f/s, ~%.0f min remaining, %d failed)",
            source, completed, len(todo), rate, remaining / 60, failed,
        )

    elapsed = time.time() - t0
    log.info("[%s] DONE: %d captioned, %d failed in %.1f min (%.2f/s)",
             source, completed, failed, elapsed / 60,
             completed / max(1e-6, elapsed))


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_hnm_items(max_rows: Optional[int]) -> list[tuple[str, dict]]:
    csv_path = REPO / "data/raw/hnm/articles.csv"
    if not csv_path.exists():
        csv_path = REPO / "data/raw/hnm_real/articles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"H&M articles.csv not found at {csv_path}")
    img_root = REPO / "data/raw/hnm_images"

    df = pd.read_csv(csv_path)

    def _path(article_id) -> Path:
        s = f"{int(article_id):010d}"
        return img_root / s[:3] / f"{s}.jpg"

    df["_path"] = df["article_id"].apply(_path)
    df = df[df["_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    log.info("H&M: %d articles with images on disk", len(df))

    if max_rows is not None and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        log.info("H&M: subsampled to %d", len(df))

    items = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        items.append((str(int(rd["article_id"])), hnm_row_to_attrs(rd)))
    return items


def load_df_inshop_items(max_rows: Optional[int]) -> list[tuple[str, dict]]:
    """Loads DF-InShop dataset directly via HF datasets API.

    Uses the row INDEX as the stable key (matches the order
    used by `DistillTextVisionDataset` in distill_recipe_z*.py).
    """
    from datasets import load_dataset
    ds = load_dataset(
        "Marqo/deepfashion-inshop",
        cache_dir=str(REPO / "data/raw/deepfashion_inshop"))["data"]
    n = len(ds)
    log.info("DF-InShop: %d rows in dataset", n)
    indices = list(range(n))
    if max_rows is not None and max_rows < n:
        random.Random(42).shuffle(indices)
        indices = sorted(indices[:max_rows])
        log.info("DF-InShop: subsampled to %d", len(indices))

    items = []
    cols_we_need = {"category", "category2", "color", "item_ID", "gender"}
    available = set(ds.column_names) & cols_we_need
    for idx in indices:
        row = {c: ds[idx][c] for c in available}
        items.append((str(idx), df_inshop_row_to_attrs(row)))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["hnm", "df_inshop"])
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap number of items to caption (default: all).")
    ap.add_argument("--n-paraphrases", type=int, default=5,
                    help="LLM paraphrases per item (default 5).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"PaleblueDot model id (default {DEFAULT_MODEL}).")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="Write batch size (also flush cadence).")
    ap.add_argument("--output", default=None,
                    help="Override output path. "
                         "Default: data/processed/llm_captions/<source>.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else \
                OUT_DIR / f"{args.source}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("MODA — LLM caption generation")
    log.info("source=%s  model=%s  n_paraphrases=%d  concurrency=%d",
             args.source, args.model, args.n_paraphrases, args.concurrency)
    log.info("output=%s", out_path)
    log.info("=" * 60)

    if args.source == "hnm":
        items = load_hnm_items(args.max_rows)
    elif args.source == "df_inshop":
        items = load_df_inshop_items(args.max_rows)
    else:
        raise ValueError(args.source)
    log.info("loaded %d items for source=%s", len(items), args.source)

    asyncio.run(run(
        source=args.source,
        items=items,
        n=args.n_paraphrases,
        model=args.model,
        concurrency=args.concurrency,
        out_path=out_path,
        chunk_size=args.chunk_size,
    ))
    log.info("Wrote captions to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
