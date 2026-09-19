import numpy as np
import pandas as pd
import psycopg2
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from maintenance_degradation import (
    build_hnsw_index, build_ivfflat_index, run_checkpoint_benchmark,
    bulk_insert_chunk, cleanup_maintenance_rows,
    HNSW_PARAMS, IVFFLAT_PARAMS, DATA_DIR, DB_CONFIG, TABLE, K, LOG_PATH
)

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE};")
        print(f"{TABLE} currently has {cur.fetchone()[0]} rows")

    print("Cleaning up items_1m back to the true 0% baseline (1,000,000 rows)...")
    cleanup_maintenance_rows(conn)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE};")
        count = cur.fetchone()[0]
    assert count == 1_000_000, f"Expected 1,000,000 rows after cleanup, got {count}"

    query_vecs = np.load(DATA_DIR / "query_embeddings.npy")
    orig_ground_truth = np.load(DATA_DIR / "ground_truth_1000000.npy")[:, :K]  # fixed: top-10, not top-100

    # remove stale checkpoint-0 rows before adding freshly measured ones
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        df = df[~((df.checkpoint_pct == 0) & (df.phase == "no_rebuild"))]
        df.to_csv(LOG_PATH, index=False)
        print("Removed stale checkpoint-0 rows from the log.")

    print("\nBuilding fresh IVFFlat baseline on the clean 1,000,000-row table...")
    build_ivfflat_index(conn, TABLE, lists=IVFFLAT_PARAMS["lists"])
    run_checkpoint_benchmark(conn, "ivfflat", IVFFLAT_PARAMS["nprobe"], query_vecs,
                              orig_ground_truth, checkpoint_pct=0, cumulative_count=0,
                              phase_label="no_rebuild")

    print("\nBuilding fresh HNSW baseline on the clean 1,000,000-row table...")
    print("(expect a few hours here, matching the original ~12,200s build time)")
    build_hnsw_index(conn, TABLE, m=HNSW_PARAMS["m"], ef_construction=HNSW_PARAMS["ef_construction"])
    run_checkpoint_benchmark(conn, "hnsw", HNSW_PARAMS["ef_search"], query_vecs,
                              orig_ground_truth, checkpoint_pct=0, cumulative_count=0,
                              phase_label="no_rebuild")

    print("\nRestoring the 500K maintenance rows so items_1m is back at 1,500,000 rows "
          "for resume_hnsw_rebuild.py...")
    insert_vecs = np.load(DATA_DIR / "insert_embeddings.npy", mmap_mode="r")
    bulk_insert_chunk(conn, insert_vecs, 0, 500_000)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE};")
        final_count = cur.fetchone()[0]
    print(f"{TABLE} now has {final_count} rows (expected 1,500,000)")

    conn.close()
    print("\nCheckpoint-0 recomputation complete for both algorithms.")