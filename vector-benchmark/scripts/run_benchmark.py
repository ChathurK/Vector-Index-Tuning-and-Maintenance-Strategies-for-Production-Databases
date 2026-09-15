import psycopg2
import numpy as np
import time
import csv
import pathlib

DB_CONFIG = dict(
    host="localhost", port=5433,
    dbname="vectorbench", user="postgres", password="research"
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "query_benchmark_log.csv"

WARMUP_QUERIES = 20  # excluded from timing, lets Postgres warm its page cache

def vector_to_pg_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def load_queries_and_ground_truth(n_tier):
    queries = np.load(DATA_DIR / "query_embeddings.npy")
    ground_truth = np.load(DATA_DIR / f"ground_truth_{n_tier}.npy")
    return queries, ground_truth

def set_search_param(conn, index_type, param_value):
    with conn.cursor() as cur:
        if index_type == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {param_value};")
        elif index_type == "ivfflat":
            cur.execute(f"SET ivfflat.probes = {param_value};")
    conn.commit()

def run_single_query(conn, table_name, query_vec, k):
    literal = vector_to_pg_literal(query_vec)
    with conn.cursor() as cur:
        start = time.perf_counter()
        cur.execute(f"""
            SELECT id FROM {table_name}
            ORDER BY embedding <=> %s
            LIMIT {k};
        """, (literal,))
        result_ids = [row[0] for row in cur.fetchall()]
        elapsed = time.perf_counter() - start
    return result_ids, elapsed

def compute_recall(retrieved_ids, true_ids_row, k):
    retrieved_set = set(retrieved_ids[:k])
    true_set = set(true_ids_row[:k].tolist())
    return len(retrieved_set & true_set) / k

def run_benchmark(conn, table_name, tier_n, index_type, param_value, k=10):
    queries, ground_truth = load_queries_and_ground_truth(tier_n)
    set_search_param(conn, index_type, param_value)

    # warmup: run a few queries untimed so page cache and query planner settle
    for i in range(WARMUP_QUERIES):
        run_single_query(conn, table_name, queries[i], k)

    latencies = []
    recalls = []
    total_start = time.perf_counter()

    for i in range(len(queries)):
        result_ids, elapsed = run_single_query(conn, table_name, queries[i], k)
        latencies.append(elapsed)
        recalls.append(compute_recall(result_ids, ground_truth[i], k))

    total_elapsed = time.perf_counter() - total_start
    latencies = np.array(latencies)

    metrics = {
        "table": table_name,
        "index_type": index_type,
        "param": param_value,
        "k": k,
        "n_queries": len(queries),
        "qps": len(queries) / total_elapsed,
        "mean_latency_ms": latencies.mean() * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "mean_recall": np.mean(recalls),
    }
    log_result(metrics)
    print(f"{table_name} | {index_type} param={param_value} | k={k} | "
          f"QPS={metrics['qps']:.1f} | recall={metrics['mean_recall']:.4f} | "
          f"p95={metrics['p95_latency_ms']:.2f}ms")
    return metrics

def log_result(metrics):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    # smoke test: default ef_search on the 100K HNSW index you just built
    run_benchmark(conn, "items_100k", tier_n=100_000,
                  index_type="hnsw", param_value=40, k=10)

    conn.close()