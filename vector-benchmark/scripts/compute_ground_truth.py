import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import pathlib

SEED = 42
MAX_TIER = 1_000_000
N_QUERIES = 500
TOP_K = 100  # ground truth depth; supports recall@1 through recall@100 later

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"
POOL_PATH = DATA_DIR / "master_pool.pkl"

def load_pool_and_permutation():
    with open(POOL_PATH, "rb") as f:
        pool = pickle.load(f)
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(pool))
    return pool, permutation

def build_query_set(pool, permutation, model):
    # queries come from AFTER the max indexed tier, so they never appear in any corpus
    query_indices = permutation[MAX_TIER:MAX_TIER + N_QUERIES]
    query_sentences = [pool[i] for i in query_indices]

    pd.DataFrame({"id": range(N_QUERIES), "text": query_sentences}).to_csv(
        DATA_DIR / "query_sentences.csv", index=False
    )

    print(f"Embedding {N_QUERIES} held-out queries...")
    query_vecs = model.encode(query_sentences, convert_to_numpy=True, show_progress_bar=True)
    np.save(DATA_DIR / "query_embeddings.npy", query_vecs)
    return query_vecs

def compute_ground_truth_for_tier(n, query_vecs):
    print(f"Computing brute-force ground truth for tier {n}...")
    corpus = np.load(DATA_DIR / f"embeddings_{n}.npy", mmap_mode="r")

    # normalize for cosine similarity (matches vector_cosine_ops in pgvector)
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    q_normalized = query_vecs / query_norms

    ground_truth = np.zeros((len(query_vecs), TOP_K), dtype=np.int64)

    # batch over queries, not corpus, to keep memory bounded
    QUERY_BATCH = 50
    for qi in range(0, len(query_vecs), QUERY_BATCH):
        q_batch = q_normalized[qi:qi + QUERY_BATCH]
        corpus_batch_normalized = corpus / corpus_norms  # (n, 768), ~n*768*4 bytes
        sims = q_batch @ corpus_batch_normalized.T  # (batch, n)
        top_k_idx = np.argpartition(-sims, TOP_K, axis=1)[:, :TOP_K]
        # sort within the top-k for correct ranking
        for row in range(top_k_idx.shape[0]):
            row_idx = top_k_idx[row]
            order = np.argsort(-sims[row, row_idx])
            ground_truth[qi + row] = row_idx[order]
        print(f"  {qi + QUERY_BATCH}/{len(query_vecs)} queries done")

    out_path = DATA_DIR / f"ground_truth_{n}.npy"
    np.save(out_path, ground_truth)
    print(f"Saved ground truth for tier {n} -> {out_path}")

if __name__ == "__main__":
    pool, permutation = load_pool_and_permutation()
    model = SentenceTransformer("all-mpnet-base-v2")

    query_vecs = build_query_set(pool, permutation, model)

    for n in [100_000, 500_000, 1_000_000]:
        compute_ground_truth_for_tier(n, query_vecs)

    print("Ground truth generation complete for all tiers.")