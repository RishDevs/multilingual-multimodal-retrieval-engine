#!/usr/bin/env python3
"""
End-to-end pipeline: WIT dataset → CLIP fine-tuning → Retrieval evaluation.
Downloads real WIT images, trains/evaluates CLIP, reports R@K, mAP, FAISS latency.
Uses huggingface transformers to avoid open_clip safetensor segfaults on Mac.
"""

import argparse, gc, hashlib, io, json, os, re, sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, pandas as pd, requests, torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "processed" / "image_cache"
RESULTS = ROOT / "reports" / "tables"

# ═══════════════════════════ Wikimedia download session ═══════════════════════════

DL_SESSION = requests.Session()
DL_SESSION.headers.update({
    "User-Agent": "WITRetrievalBot/1.0 (https://github.com/google-research-datasets/wit; krrish@research.edu) python-requests/2.31",
    "Accept": "image/*,*/*",
})

def download_one(url, cache_dir, timeout=20):
    """Download a single Wikipedia image, with http→https fix and thumbnail fallback."""
    url_norm = url.replace("http://", "https://", 1) if url.startswith("http://") else url
    h = hashlib.md5(url.encode()).hexdigest()
    p = cache_dir / f"{h}.jpg"
    if p.exists() and p.stat().st_size > 1000:
        return str(p)

    for try_url in _url_variants(url_norm):
        try:
            r = DL_SESSION.get(try_url, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(str(p), "JPEG", quality=85)
                return str(p)
        except Exception:
            pass
    return None

def _url_variants(url):
    """Return direct + thumbnail URL variants."""
    yield url
    if "upload.wikimedia.org" in url and "/thumb/" not in url:
        base = url.replace("https://upload.wikimedia.org/wikipedia/", "")
        parts = base.split("/")
        if len(parts) >= 3:
            project, rest, filename = parts[0], "/".join(parts[1:]), parts[-1]
            yield f"https://upload.wikimedia.org/wikipedia/{project}/thumb/{rest}/300px-{filename}"

def download_batch(urls, cache_dir, workers=4):
    """Download images in parallel."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, u, cache_dir): u for u in urls}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading images"):
            results[futs[f]] = f.result()
    ok = sum(1 for v in results.values() if v)
    print(f"[DL] {ok}/{len(urls)} images downloaded")
    return results

# ═══════════════════════════ WIT Data Prep ═══════════════════════════

def clean(t):
    if not isinstance(t, str): return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"\[.*?\]", " ", t))).strip()

def load_and_prepare_wit(tsv_path, languages, num_per_lang, nrows=50000):
    """Load WIT TSV and prepare balanced manifest."""
    df = pd.read_csv(str(tsv_path), sep="\t", compression="gzip", on_bad_lines="skip", nrows=nrows)
    print(f"[DATA] Loaded {len(df)} raw rows")

    fields = ["caption_reference_description", "caption_attribution_description",
              "caption_alt_text_description", "context_page_description"]
    def best(row):
        for f in fields:
            v = row.get(f)
            if pd.notna(v) and isinstance(v, str) and len(v.strip()) >= 10:
                return clean(v)
        return ""

    df["text"] = df.apply(best, axis=1)
    df = df[df["text"].str.len() >= 10]
    df = df[df["image_url"].notna()]
    df = df[df["language"].isin(languages)]
    df["img_hash"] = df["image_url"].apply(lambda u: hashlib.md5(str(u).encode()).hexdigest()[:10])
    df = df.drop_duplicates("img_hash")

    parts = []
    for lang in languages:
        sub = df[df["language"] == lang]
        parts.append(sub.sample(n=min(num_per_lang, len(sub)), random_state=42))
    
    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[DATA] Manifest: {len(out)} samples | {dict(out['language'].value_counts())}")
    return out

# ═══════════════════════════ Evaluation ═══════════════════════════

def compute_metrics(img_emb, txt_emb, ks=[1, 5, 10]):
    """Compute R@K, mAP, MRR from paired embeddings."""
    sim = img_emb @ txt_emb.T
    n = sim.shape[0]
    tgt = np.arange(n)
    m = {}

    for k in ks:
        topk = np.argsort(-sim.T, axis=1)[:, :k]
        m[f"t2i_r@{k}"] = float(np.mean(np.any(topk == tgt[:, None], axis=1)))
        topk = np.argsort(-sim, axis=1)[:, :k]
        m[f"i2t_r@{k}"] = float(np.mean(np.any(topk == tgt[:, None], axis=1)))

    rrs = [1.0 / (np.where(np.argsort(-sim.T[i]) == i)[0][0] + 1) for i in range(n)]
    m["map"] = m["mrr"] = float(np.mean(rrs))
    return m

def compute_per_lang(img_emb, txt_emb, langs):
    """Per-language R@10 and mAP."""
    langs = np.array(langs)
    res = {}
    for lang in sorted(set(langs)):
        idx = np.where(langs == lang)[0]
        if len(idx) < 3: continue
        sub_m = compute_metrics(img_emb[idx], txt_emb[idx], ks=[1, 5, 10])
        sub_m["n"] = int(len(idx))
        res[lang] = sub_m
    return res

def bench_faiss(emb, num_q=500):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import faiss
    
    dim = emb.shape[1]
    res = {}
    for t in ["IndexFlatIP", "HNSW"]:
        ix = faiss.IndexFlatIP(dim) if t == "IndexFlatIP" else faiss.IndexHNSWFlat(dim, 32)
        if t == "HNSW": ix.hnsw.efSearch = 128
        ix.add(emb.astype(np.float32))
        for _ in range(50):
            ix.search(emb[np.random.randint(len(emb))].reshape(1, -1).astype(np.float32), 10)
        lats = []
        for _ in range(min(num_q, len(emb) * 3)):
            s = time.perf_counter()
            ix.search(emb[np.random.randint(len(emb))].reshape(1, -1).astype(np.float32), 10)
            lats.append((time.perf_counter() - s) * 1000)
        res[t] = {"p50": np.percentile(lats, 50), "p95": np.percentile(lats, 95),
                   "p99": np.percentile(lats, 99), "qps": 1000 / np.mean(lats), "n": ix.ntotal}
    return res

# ═══════════════════════════ Main Pipeline ═══════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-lang", type=int, default=200, help="Samples per language")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nrows", type=int, default=80000)
    args = parser.parse_args()

    print("=" * 70)
    print("MULTILINGUAL MULTIMODAL RETRIEVAL ENGINE")
    print("Transformers CLIP Dual Encoder · Hard Negative Mining · FAISS")
    print("=" * 70)
    device = torch.device("cpu")
    print(f"[DEVICE] {device}")

    # ── 1. Load CLIP model FIRST ──
    print("\n[MODEL] Loading Transformers CLIP ViT-B-32 (OpenAI pretrained)...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    clip_model.eval()
    clip_model.to(device)
    print(f"[MODEL] {sum(p.numel() for p in clip_model.parameters()):,} parameters")

    # ── 2. Load and prepare WIT ──
    tsv = ROOT / "data" / "raw" / "wit_v1.train.all-1percent_sample.tsv.gz"
    languages = ["en", "de", "fr", "es"]
    manifest = load_and_prepare_wit(tsv, languages, args.per_lang, nrows=args.nrows)
    gc.collect()

    # ── 3. Download images ──
    urls = manifest["image_url"].tolist()
    paths = download_batch(urls, CACHE, workers=args.workers)

    mask = [paths.get(u) is not None for u in manifest["image_url"]]
    manifest = manifest[mask].reset_index(drop=True)
    print(f"[DATA] After download filter: {len(manifest)} paired samples")

    if len(manifest) < 20:
        print("[ERROR] Too few images downloaded. Try again later or increase --nrows.")
        sys.exit(1)

    # ── 4. Build paired tensors ──
    print("\n[PREP] Processing image-text pairs...")
    images_list = []
    texts_list = []
    langs_list = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing pairs"):
        img_path = paths[row["image_url"]]
        try:
            img = Image.open(img_path).convert("RGB")
            # We delay converting to tensor to save RAM, save as paths instead
            images_list.append(img_path)
            texts_list.append(row["text"][:77])
            langs_list.append(row["language"])
        except Exception:
            pass

    print(f"[PREP] {len(images_list)} pairs ready")

    # ── 5. Split ──
    n = len(images_list)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    print(f"[SPLIT] Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

    test_imgs_paths = [images_list[i] for i in test_idx]
    test_texts = [texts_list[i] for i in test_idx]
    test_langs = [langs_list[i] for i in test_idx]

    # Batch processing function to avoid memory issues
    def get_embeddings(img_paths, texts, b_size=32):
        clip_model.eval()
        all_ie, all_te = [], []
        with torch.no_grad():
            for start in range(0, len(img_paths), b_size):
                b_img = [Image.open(p).convert("RGB") for p in img_paths[start:start+b_size]]
                b_txt = texts[start:start+b_size]
                
                inputs_img = processor(images=b_img, return_tensors="pt")["pixel_values"].to(device)
                inputs_txt = processor(text=b_txt, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs_txt["input_ids"].to(device)
                attn_mask = inputs_txt["attention_mask"].to(device)
                
                ie = clip_model.get_image_features(pixel_values=inputs_img)
                if hasattr(ie, "pooler_output"): ie = ie.pooler_output
                elif not isinstance(ie, torch.Tensor): ie = ie[1]

                te = clip_model.get_text_features(input_ids=input_ids, attention_mask=attn_mask)
                if hasattr(te, "pooler_output"): te = te.pooler_output
                elif not isinstance(te, torch.Tensor): te = te[1]
                
                all_ie.append(F.normalize(ie, dim=-1).cpu())
                all_te.append(F.normalize(te, dim=-1).cpu())
        return torch.cat(all_ie).numpy(), torch.cat(all_te).numpy()

    # ── 6. Evaluate pretrained (zero-shot) ──
    print("\n" + "=" * 70)
    print("PRETRAINED CLIP EVALUATION (zero-shot)")
    print("=" * 70)

    test_img_emb, test_txt_emb = get_embeddings(test_imgs_paths, test_texts)
    pre = compute_metrics(test_img_emb, test_txt_emb)
    print(f"  T→I  R@1={pre['t2i_r@1']*100:.1f}%  R@5={pre['t2i_r@5']*100:.1f}%  R@10={pre['t2i_r@10']*100:.1f}%")
    print(f"  I→T  R@1={pre['i2t_r@1']*100:.1f}%  R@5={pre['i2t_r@5']*100:.1f}%  R@10={pre['i2t_r@10']*100:.1f}%")
    print(f"  mAP={pre['map']:.4f}  MRR={pre['mrr']:.4f}")

    # ── 7. Fine-tune ──
    if not args.eval_only and n_train >= 10:
        print("\n" + "=" * 70)
        print("FINE-TUNING ON WIT DATA")
        print("=" * 70)

        optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.lr, weight_decay=0.01)

        train_imgs_paths = [images_list[i] for i in train_idx]
        train_texts = [texts_list[i] for i in train_idx]
        val_imgs_paths = [images_list[i] for i in val_idx]
        val_texts = [texts_list[i] for i in val_idx]

        bs = min(args.batch_size, n_train)
        total_steps = (n_train // bs) * args.epochs
        import math
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

        best_r10 = 0
        best_state = None

        for epoch in range(args.epochs):
            clip_model.train()
            perm_e = np.random.RandomState(epoch).permutation(n_train)
            losses = []
            t0 = time.time()

            for start in tqdm(range(0, n_train - bs + 1, bs), desc=f"Epoch {epoch+1}/{args.epochs}"):
                batch_idx = perm_e[start:start + bs]
                b_img = [Image.open(train_imgs_paths[i]).convert("RGB") for i in batch_idx]
                b_txt = [train_texts[i] for i in batch_idx]

                b_img_tensor = processor(images=b_img, return_tensors="pt")["pixel_values"].to(device)
                b_txt_res = processor(text=b_txt, return_tensors="pt", padding=True, truncation=True)
                b_txt_id = b_txt_res["input_ids"].to(device)
                b_txt_mask = b_txt_res["attention_mask"].to(device)

                outputs = clip_model(input_ids=b_txt_id, pixel_values=b_img_tensor, attention_mask=b_txt_mask, return_loss=True)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            # Validate
            v_ie, v_te = get_embeddings(val_imgs_paths, val_texts)
            vm = compute_metrics(v_ie, v_te)

            print(f"\n[E{epoch+1}] loss={np.mean(losses):.4f} | "
                  f"T→I R@1={vm['t2i_r@1']*100:.1f}% R@5={vm['t2i_r@5']*100:.1f}% R@10={vm['t2i_r@10']*100:.1f}% | "
                  f"I→T R@10={vm['i2t_r@10']*100:.1f}% | mAP={vm['map']:.4f} | {time.time()-t0:.0f}s")

            if vm["t2i_r@10"] > best_r10:
                best_r10 = vm["t2i_r@10"]
                best_state = {k: v.cpu().clone() for k, v in clip_model.state_dict().items()}
                print(f"  ✓ New best T→I R@10: {best_r10*100:.1f}%")

        if best_state:
            clip_model.load_state_dict(best_state)

    # ── 8. Final evaluation ──
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    test_img_emb, test_txt_emb = get_embeddings(test_imgs_paths, test_texts)
    fin = compute_metrics(test_img_emb, test_txt_emb)
    
    print(f"\n  TEXT → IMAGE:")
    print(f"    R@1:  {fin['t2i_r@1']*100:.1f}%")
    print(f"    R@5:  {fin['t2i_r@5']*100:.1f}%")
    print(f"    R@10: {fin['t2i_r@10']*100:.1f}%")
    print(f"\n  IMAGE → TEXT:")
    print(f"    R@1:  {fin['i2t_r@1']*100:.1f}%")
    print(f"    R@5:  {fin['i2t_r@5']*100:.1f}%")
    print(f"    R@10: {fin['i2t_r@10']*100:.1f}%")
    print(f"\n  mAP: {fin['map']:.4f}  MRR: {fin['mrr']:.4f}")

    # ── 9. Per-language ──
    print("\n" + "=" * 70)
    print("PER-LANGUAGE BREAKDOWN")
    print("=" * 70)
    pl = compute_per_lang(test_img_emb, test_txt_emb, test_langs)
    hdr = f"{'Lang':>8} | {'N':>5} | {'T→I R@1':>9} | {'T→I R@5':>9} | {'T→I R@10':>10} | {'I→T R@10':>10} | {'mAP':>8}"
    print(hdr); print("-" * len(hdr))
    for l, m in sorted(pl.items()):
        print(f"{l:>8} | {m['n']:>5} | {m['t2i_r@1']*100:>8.1f}% | {m['t2i_r@5']*100:>8.1f}% | {m['t2i_r@10']*100:>9.1f}% | {m['i2t_r@10']*100:>9.1f}% | {m['map']:>8.4f}")
    if pl:
        mac_r10 = np.mean([m["t2i_r@10"] for m in pl.values()])
        print(f"{'MACRO':>8} |       |           |           | {mac_r10*100:>9.1f}% |")

    # ── 10. FAISS ──
    print("\n" + "=" * 70)
    print("FAISS LATENCY BENCHMARK")
    print("=" * 70)
    for t, s in bench_faiss(test_img_emb).items():
        print(f"  {t}: P50={s['p50']:.3f}ms P95={s['p95']:.3f}ms P99={s['p99']:.3f}ms QPS={s['qps']:.0f}")

    # ── 11. Save ──
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_res = {"pretrained": pre, "final": fin, "per_language": pl, "config": vars(args)}
    with open(RESULTS / "final_results.json", "w") as f:
        json.dump(all_res, f, indent=2, default=str)

    ckpt_dir = ROOT / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": clip_model.state_dict(), "metrics": fin}, str(ckpt_dir / "best.ckpt"))

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
