import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time
import pathlib
import pickle
import os

SEED = 42
BATCH_SIZE = 256
MAX_TIER = 1_000_000
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"
DATA_DIR.mkdir(parents=True, exist_ok=True)
POOL_PATH = DATA_DIR / "master_pool.pkl"
# os.makedirs(DATA_DIR, exist_ok=True)

def build_master_pool(pool_size):
    if POOL_PATH.exists():
        print(f"Loading cached pool from {POOL_PATH}")
        with open(POOL_PATH, "rb") as f:
            return pickle.load(f)

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    sentences = []
    for row in ds:
        clean_text = " ".join(row["text"].split())  # collapses all whitespace, including \n\n, to single spaces
        for s in clean_text.split(". "):
            s = s.strip()
            if 20 < len(s) < 300:
                sentences.append(s)
        if len(sentences) >= pool_size:
            break
    pool = sentences[:pool_size]

    with open(POOL_PATH, "wb") as f:
        pickle.dump(pool, f)
    print(f"Saved pool ({len(pool)} sentences) to {POOL_PATH}")

    return pool

def get_nested_permutation(pool_len):
    """One fixed permutation reused across all tiers to guarantee nesting."""
    rng = np.random.default_rng(SEED)
    return rng.permutation(pool_len)

def generate_tier(n, model, pool, permutation):
    print(f"--- Generating tier: {n} vectors ---")
    idx = np.sort(permutation[:n])  # sorted slice of the SAME permutation for every tier
    sentences = [pool[i] for i in idx]

    pd.DataFrame({"id": range(n), "text": sentences}).to_csv(
        f"{DATA_DIR}/sentences_{n}.csv", index=False
    )

    out_path = f"{DATA_DIR}/embeddings_{n}.npy"
    all_vecs = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n, 768)
    )

    start = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs[i:i + len(batch)] = vecs
        if i % (BATCH_SIZE * 20) == 0:
            elapsed = time.time() - start
            rate = (i + BATCH_SIZE) / elapsed if elapsed > 0 else 0
            print(f"  {i+BATCH_SIZE}/{n} done, {rate:.1f} sent/sec")

    all_vecs.flush()
    assert not np.isnan(all_vecs).any(), "NaN values detected in embeddings!"
    print(f"Tier {n} complete in {time.time()-start:.1f}s -> {out_path}")

if __name__ == "__main__":
    model = SentenceTransformer("all-mpnet-base-v2")

    pool_size = int(MAX_TIER * 1.1)  # small buffer for filtering safety
    pool = build_master_pool(pool_size)
    permutation = get_nested_permutation(len(pool))

    # generate_tier(100_000, model, pool, permutation)
    # generate_tier(500_000, model, pool, permutation)
    generate_tier(1_000_000, model, pool, permutation)