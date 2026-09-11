import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time
import pathlib
import os

SEED = 42
BATCH_SIZE = 256
MAX_TIER = 1_000_000
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# os.makedirs(DATA_DIR, exist_ok=True)

def load_source_sentences(n):
    """Load and deterministically sample n sentences from Wikipedia."""
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    rng = np.random.default_rng(SEED)
    sentences = []
    for row in ds:
        text = row["text"]
        # crude split into sentence-like chunks; refine as needed
        for s in text.split(". "):
            s = s.strip()
            if 20 < len(s) < 300:
                sentences.append(s)
            if len(sentences) >= n * 2:  # oversample, then trim deterministically
                break
        if len(sentences) >= n * 2:
            break
    idx = rng.choice(len(sentences), size=n, replace=False)
    idx.sort()  # preserve nesting property across tiers
    return [sentences[i] for i in idx]

def generate_tier(n, model):
    print(f"--- Generating tier: {n} vectors ---")
    sentences = load_source_sentences(n)

    # save the text index for traceability
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

    # Step 1: small validation slice first
    generate_tier(500, model)

    # Step 2: uncomment once validated
    # generate_tier(100_000, model)
    # generate_tier(500_000, model)
    # generate_tier(1_000_000, model)