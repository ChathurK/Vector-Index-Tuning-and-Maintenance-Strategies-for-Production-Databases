import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle
import pathlib
import time

SEED = 42
EXISTING_POOL_SIZE = 1_100_000  # already consumed by the 3 tiers + query set + buffer
INSERT_POOL_SIZE = 550_000      # covers up to 50% of the 1M tier with margin
BATCH_SIZE = 256

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"
INSERT_POOL_PATH = DATA_DIR / "insert_pool.pkl"
INSERT_EMB_PATH = DATA_DIR / "insert_embeddings.npy"
INSERT_CSV_PATH = DATA_DIR / "insert_sentences.csv"

def build_insert_pool():
    if INSERT_POOL_PATH.exists():
        print("Loading cached insert pool...")
        with open(INSERT_POOL_PATH, "rb") as f:
            return pickle.load(f)

    print(f"Streaming past the first {EXISTING_POOL_SIZE} filtered sentences "
          f"to build a pool disjoint from all existing tiers/queries...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    skipped = 0
    collected = []
    for row in ds:
        clean_text = " ".join(row["text"].split())
        for s in clean_text.split(". "):
            s = s.strip()
            if 20 < len(s) < 300:
                if skipped < EXISTING_POOL_SIZE:
                    skipped += 1
                    continue
                collected.append(s)
        if len(collected) >= INSERT_POOL_SIZE:
            break

    pool = collected[:INSERT_POOL_SIZE]
    with open(INSERT_POOL_PATH, "wb") as f:
        pickle.dump(pool, f)
    print(f"Saved disjoint insert pool ({len(pool)} sentences) -> {INSERT_POOL_PATH}")
    return pool

def embed_insert_pool(pool, model):
    n = len(pool)
    pd.DataFrame({"id": range(n), "text": pool}).to_csv(INSERT_CSV_PATH, index=False)

    all_vecs = np.lib.format.open_memmap(
        INSERT_EMB_PATH, mode="w+", dtype=np.float32, shape=(n, 768)
    )
    start = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch = pool[i:i + BATCH_SIZE]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs[i:i + len(batch)] = vecs
        if i % (BATCH_SIZE * 40) == 0:
            elapsed = time.time() - start
            rate = (i + BATCH_SIZE) / elapsed if elapsed > 0 else 0
            print(f"  {i+BATCH_SIZE}/{n} done, {rate:.1f} sent/sec")
    all_vecs.flush()
    assert not np.isnan(all_vecs).any(), "NaN detected in insert-pool embeddings!"
    print(f"Insert pool embeddings complete in {time.time()-start:.1f}s -> {INSERT_EMB_PATH}")

if __name__ == "__main__":
    pool = build_insert_pool()
    model = SentenceTransformer("all-mpnet-base-v2")
    embed_insert_pool(pool, model)