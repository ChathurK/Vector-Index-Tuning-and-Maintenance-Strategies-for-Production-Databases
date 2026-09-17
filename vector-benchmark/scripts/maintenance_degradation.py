import psycopg2
import numpy as np
import io
import time
import csv
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_index import build_hnsw_index, build_ivfflat_index

DB_CONFIG = dict(
    host="localhost", port=5433,
    dbname="vectorbench", user="postgres", password="research"
)

DATA_DIR = SCRIPT_DIR.parent / "embeddings"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "maintenance_degradation_log.csv"

TABLE = "items_1m"
K = 10
MAINTENANCE_ID_OFFSET = 800_000_000  # distinct from workload's 900,000,000
CHECKPOINTS = [100_000, 200_000, 300_000, 500_000]  # cumulative inserts = 10/20/30/50% of 1M
QUERY_BATCH = 50

HNSW_PARAMS = dict(m=32, ef_construction=128, ef_search=40)
IVFFLAT_PARAMS = dict(lists=1000, nprobe=20)  # nlist matches scalability's sqrt(1M)

def vector_to_pg_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def bulk_insert_chunk(conn, insert_vecs, start_idx, end_idx):
    print(f"  Inserting rows {start_idx}-{end_idx} ({end_idx - start_idx} vectors)...")
    with conn.cursor() as cur:
        CHUNK = 20_000
        for cs in range(start_idx, end_idx, CHUNK):
            ce = min(cs + CHUNK, end_idx)
            buf = io.StringIO()
            for i in range(cs, ce):
                buf.write(f"{MAINTENANCE_ID_OFFSET + i}\t{vector_to_pg_literal(insert_vecs[i])}\n")
            buf.seek(0)
            cur.copy_expert(
                f"COPY {TABLE} (id, embedding) FROM STDIN WITH (FORMAT text)", buf
            )
    conn.commit()

def compute_growing_ground_truth(query_vecs, orig_norms, insert_vecs, cumulative_count):
    q_norm = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    ground_truth_ids = np.zeros((len(query_vecs), K), dtype=np.int64)

    orig_norms_T = np.ascontiguousarray(orig_norms.T)  # fix: contiguous, computed once
    insert_norms_T = None
    if cumulative_count > 0:
        insert_slice = insert_vecs[:cumulative_count]
        insert_norms = insert_slice / np.linalg.norm(insert_slice, axis=1, keepdims=True)
        insert_norms_T = np.ascontiguousarray(insert_norms.T)

    for qi in range(0, len(query_vecs), QUERY_BATCH):
        q_batch = q_norm[qi:qi + QUERY_BATCH]
        sims_orig = q_batch @ orig_norms_T
        if cumulative_count > 0:
            sims_ins = q_batch @ insert_norms_T
            sims_full = np.concatenate([sims_orig, sims_ins], axis=1)
        else:
            sims_full = sims_orig

        top_k_idx = np.argpartition(-sims_full, K, axis=1)[:, :K]
        for row in range(top_k_idx.shape[0]):
            row_idx = top_k_idx[row]
            order = np.argsort(-sims_full[row, row_idx])
            sorted_idx = row_idx[order]
            ids = np.where(sorted_idx < 1_000_000, sorted_idx,
                            MAINTENANCE_ID_OFFSET + (sorted_idx - 1_000_000))
            ground_truth_ids[qi + row] = ids
    assert not np.isnan(ground_truth_ids).any(), "NaN in ground truth indices!"
    assert (ground_truth_ids >= 0).all(), "Negative index in ground truth — matmul corruption!"
    assert (ground_truth_ids < MAINTENANCE_ID_OFFSET + cumulative_count).all() or cumulative_count == 0, \
        "Ground truth ID out of valid range!"
    return ground_truth_ids

def set_search_param(conn, index_type, param_value):
    with conn.cursor() as cur:
        if index_type == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {param_value};")
        else:
            cur.execute(f"SET ivfflat.probes = {param_value};")
    conn.commit()

def run_checkpoint_benchmark(conn, index_type, search_param, query_vecs, ground_truth_ids,
                              checkpoint_pct, cumulative_count, phase_label):
    set_search_param(conn, index_type, search_param)
    latencies, recalls = [], []

    for i in range(len(query_vecs)):
        literal = vector_to_pg_literal(query_vecs[i])
        with conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT {K};", (literal,))
            result_ids = [row[0] for row in cur.fetchall()]
            latencies.append(time.perf_counter() - start)
        overlap = len(set(result_ids) & set(ground_truth_ids[i].tolist()))
        recalls.append(overlap / K)

    latencies = np.array(latencies)
    metrics = {
        "index_type": index_type, "checkpoint_pct": checkpoint_pct,
        "cumulative_inserts": cumulative_count, "total_rows": 1_000_000 + cumulative_count,
        "phase": phase_label, "mean_recall": np.mean(recalls),
        "qps": len(query_vecs) / latencies.sum(),
        "mean_latency_ms": latencies.mean() * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
    }
    log_result(metrics)
    print(f"  [{phase_label}] {index_type} @ {checkpoint_pct}% ({metrics['total_rows']} rows) | "
          f"recall={metrics['mean_recall']:.4f} | QPS={metrics['qps']:.1f} | "
          f"p95={metrics['p95_latency_ms']:.2f}ms")

def log_result(metrics):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def cleanup_maintenance_rows(conn):
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE} WHERE id >= %s;", (MAINTENANCE_ID_OFFSET,))
        cur.execute(f"VACUUM ANALYZE {TABLE};")
    conn.autocommit = False

def run_degradation_test(conn, index_type, params, orig_norms, insert_vecs, query_vecs,
                          orig_ground_truth):
    print(f"\n=== Maintenance test: {index_type} on {TABLE} ===")

    print(f"Building fresh baseline {index_type} index...")
    if index_type == "hnsw":
        build_hnsw_index(conn, TABLE, m=params["m"], ef_construction=params["ef_construction"])
        search_param = params["ef_search"]
    else:
        build_ivfflat_index(conn, TABLE, lists=params["lists"])
        search_param = params["nprobe"]

    # checkpoint 0: baseline, reuse existing precomputed ground truth
    run_checkpoint_benchmark(conn, index_type, search_param, query_vecs, orig_ground_truth,
                              checkpoint_pct=0, cumulative_count=0, phase_label="no_rebuild")

    prev_count = 0
    final_ground_truth = None
    for cumulative_count in CHECKPOINTS:
        pct = round(cumulative_count / 1_000_000 * 100)
        bulk_insert_chunk(conn, insert_vecs, prev_count, cumulative_count)
        prev_count = cumulative_count

        print(f"  Computing ground truth for {cumulative_count} cumulative inserts...")
        gt = compute_growing_ground_truth(query_vecs, orig_norms, insert_vecs, cumulative_count)
        run_checkpoint_benchmark(conn, index_type, search_param, query_vecs, gt,
                                  checkpoint_pct=pct, cumulative_count=cumulative_count,
                                  phase_label="no_rebuild")
        final_ground_truth = gt

    # RQ5: full rebuild after growth, compare against the pre-rebuild 50% checkpoint
    print(f"  Performing full rebuild on grown table ({1_000_000 + prev_count} rows)...")
    if index_type == "hnsw":
        build_hnsw_index(conn, TABLE, m=params["m"], ef_construction=params["ef_construction"])
    else:
        build_ivfflat_index(conn, TABLE, lists=params["lists"])

    run_checkpoint_benchmark(conn, index_type, search_param, query_vecs, final_ground_truth,
                              checkpoint_pct=50, cumulative_count=prev_count,
                              phase_label="post_rebuild")

    cleanup_maintenance_rows(conn)
    print(f"Cleaned up maintenance rows, {TABLE} reset to 1,000,000 rows.")

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    query_vecs = np.load(DATA_DIR / "query_embeddings.npy")
    insert_vecs = np.load(DATA_DIR / "insert_embeddings.npy", mmap_mode="r")
    orig_ground_truth = np.load(DATA_DIR / "ground_truth_1000000.npy")

    print("Loading and normalizing the original 1M corpus (cached across both algorithms)...")
    orig_corpus = np.load(DATA_DIR / "embeddings_1000000.npy")
    orig_norms = orig_corpus / np.linalg.norm(orig_corpus, axis=1, keepdims=True)

    # IVFFlat first: fast, gives a complete result quickly
    run_degradation_test(conn, "ivfflat", IVFFLAT_PARAMS, orig_norms, insert_vecs, query_vecs,
                          orig_ground_truth)

    # HNSW last: the long one (~7-9 hours combined for both its builds)
    run_degradation_test(conn, "hnsw", HNSW_PARAMS, orig_norms, insert_vecs, query_vecs,
                          orig_ground_truth)

    conn.close()
    print("\nMaintenance & degradation testing complete.")