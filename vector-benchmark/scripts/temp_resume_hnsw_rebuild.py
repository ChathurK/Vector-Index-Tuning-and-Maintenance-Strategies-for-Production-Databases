import numpy as np
import psycopg2
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from maintenance_degradation import (
    build_hnsw_index, compute_growing_ground_truth, run_checkpoint_benchmark,
    cleanup_maintenance_rows, HNSW_PARAMS, DATA_DIR, DB_CONFIG, TABLE
)

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    # sanity check before doing anything expensive
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE};")
        row_count = cur.fetchone()[0]
    print(f"{TABLE} currently has {row_count} rows")
    assert row_count == 1_500_000, f"Expected 1,500,000 rows, found {row_count} — stop and investigate."

    query_vecs = np.load(DATA_DIR / "query_embeddings.npy")
    insert_vecs = np.load(DATA_DIR / "insert_embeddings.npy", mmap_mode="r")

    print("Loading and normalizing the original 1M corpus...")
    orig_corpus = np.load(DATA_DIR / "embeddings_1000000.npy")
    orig_norms = orig_corpus / np.linalg.norm(orig_corpus, axis=1, keepdims=True)

    print("Recomputing ground truth for the 500K-insert (50%) state...")
    ground_truth_50pct = compute_growing_ground_truth(query_vecs, orig_norms, insert_vecs, 500_000)

    print("Rebuilding HNSW index on the grown 1.5M-row table...")
    build_hnsw_index(conn, TABLE, m=HNSW_PARAMS["m"], ef_construction=HNSW_PARAMS["ef_construction"])

    run_checkpoint_benchmark(conn, "hnsw", HNSW_PARAMS["ef_search"], query_vecs, ground_truth_50pct,
                              checkpoint_pct=50, cumulative_count=500_000, phase_label="post_rebuild")

    print("Cleaning up maintenance rows...")
    cleanup_maintenance_rows(conn)

    conn.close()
    print("\nHNSW post-rebuild benchmark complete. items_1m reset to 1,000,000 rows.")