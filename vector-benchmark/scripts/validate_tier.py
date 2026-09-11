import numpy as np
import pandas as pd
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

# Build file paths using DATA_DIR
emb_path = DATA_DIR / f"embeddings_{n}.npy"
sent_path = DATA_DIR / f"sentences_{n}.csv"

# Load
vecs = np.load(emb_path)
df = pd.read_csv(sent_path)

print(f"Shape: {vecs.shape}")
assert vecs.shape == (n, 768), "Shape mismatch!"
assert not np.isnan(vecs).any(), "NaN found!"
assert len(df) == n, "Sentence count mismatch!"

norms = np.linalg.norm(vecs, axis=1)
print(f"Vector norm range: {norms.min():.3f} - {norms.max():.3f}")
print(f"Sample sentence [0]: {df.iloc[0]['text'][:80]}...")
print("Validation passed.")