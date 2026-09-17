import psycopg2
import numpy as np
import threading
import time
import random
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
LOG_PATH = RESULTS_DIR / "workload_profile_log.csv"

TABLE = "items_500k"
DURATION_SEC = 60
N_WORKERS = 8
K = 10
WRITE_ID_OFFSET = 900_000_000  # far outside any real tier's id range

HNSW_PARAMS = dict(m=32, ef_construction=128, ef_search=40)
IVFFLAT_PARAMS = dict(lists=707, nprobe=20)

WORKLOAD_PROFILES = [
    ("read_heavy", 0.9, 0.1),
    ("balanced", 0.5, 0.5),
    ("write_heavy", 0.3, 0.7),
]

def vector_to_pg_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def set_search_param(conn, index_type, param_value):
    with conn.cursor() as cur:
        if index_type == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {param_value};")
        else:
            cur.execute(f"SET ivfflat.probes = {param_value};")
    conn.commit()

def worker_loop(worker_id, read_ratio, index_type, search_param,
                 query_vecs, insert_vecs, id_counter, id_lock,
                 stop_event, results):
    conn = psycopg2.connect(**DB_CONFIG)
    set_search_param(conn, index_type, search_param)
    read_latencies = []
    write_count = 0

    while not stop_event.is_set():
        if random.random() < read_ratio:
            qv = query_vecs[random.randrange(len(query_vecs))]
            literal = vector_to_pg_literal(qv)
            with conn.cursor() as cur:
                start = time.perf_counter()
                cur.execute(f"""
                    SELECT id FROM {TABLE}
                    ORDER BY embedding <=> %s
                    LIMIT {K};
                """, (literal,))
                cur.fetchall()
                read_latencies.append(time.perf_counter() - start)
        else:
            with id_lock:
                idx = id_counter[0] % len(insert_vecs)
                new_id = WRITE_ID_OFFSET + id_counter[0]
                id_counter[0] += 1
            vec = insert_vecs[idx]
            literal = vector_to_pg_literal(vec)
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {TABLE} (id, embedding) VALUES (%s, %s);",
                            (new_id, literal))
            conn.commit()
            write_count += 1

    conn.close()
    results[worker_id] = (read_latencies, write_count)

def run_workload(index_type, search_param, profile_name, read_ratio, write_ratio,
                  query_vecs, insert_vecs):
    print(f"\n--- Workload: {index_type} | {profile_name} "
          f"({int(read_ratio*100)}/{int(write_ratio*100)}) ---")

    stop_event = threading.Event()
    id_counter = [0]
    id_lock = threading.Lock()
    results = {}

    threads = []
    for i in range(N_WORKERS):
        t = threading.Thread(target=worker_loop, args=(
            i, read_ratio, index_type, search_param,
            query_vecs, insert_vecs, id_counter, id_lock,
            stop_event, results
        ))
        threads.append(t)
        t.start()

    time.sleep(DURATION_SEC)
    stop_event.set()
    for t in threads:
        t.join()

    all_read_latencies = []
    total_writes = 0
    for read_lat, writes in results.values():
        all_read_latencies.extend(read_lat)
        total_writes += writes

    all_read_latencies = np.array(all_read_latencies) if all_read_latencies else np.array([0])
    read_qps = len(all_read_latencies) / DURATION_SEC
    write_throughput = total_writes / DURATION_SEC

    metrics = {
        "table": TABLE, "index_type": index_type, "profile": profile_name,
        "read_ratio": read_ratio, "write_ratio": write_ratio,
        "duration_sec": DURATION_SEC, "n_workers": N_WORKERS,
        "read_qps": read_qps,
        "mean_read_latency_ms": all_read_latencies.mean() * 1000,
        "p95_read_latency_ms": np.percentile(all_read_latencies, 95) * 1000,
        "p99_read_latency_ms": np.percentile(all_read_latencies, 99) * 1000,
        "write_throughput_per_sec": write_throughput,
        "total_writes": total_writes,
    }
    log_result(metrics)
    print(f"Read QPS={read_qps:.1f} | mean_lat={metrics['mean_read_latency_ms']:.2f}ms | "
          f"p95={metrics['p95_read_latency_ms']:.2f}ms | "
          f"write_throughput={write_throughput:.1f}/s | total_writes={total_writes}")

    # cleanup: remove inserted rows, then VACUUM to prevent dead-tuple bloat
    # from accumulating across the 6 runs in this script - unlike the earlier
    # index build/drop cycles, this phase actually churns table rows via
    # real INSERT+DELETE, which is exactly what causes bloat.
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE} WHERE id >= %s;", (WRITE_ID_OFFSET,))
        cur.execute(f"VACUUM ANALYZE {TABLE};")
    conn.close()

def log_result(metrics):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    query_vecs = np.load(DATA_DIR / "query_embeddings.npy")
    insert_vecs = np.load(DATA_DIR / "insert_embeddings.npy", mmap_mode="r")

    print("Building HNSW baseline on items_500k...")
    build_hnsw_index(conn, TABLE, m=HNSW_PARAMS["m"], ef_construction=HNSW_PARAMS["ef_construction"])
    for name, r, w in WORKLOAD_PROFILES:
        run_workload("hnsw", HNSW_PARAMS["ef_search"], name, r, w, query_vecs, insert_vecs)

    print("\nBuilding IVFFlat baseline on items_500k...")
    build_ivfflat_index(conn, TABLE, lists=IVFFLAT_PARAMS["lists"])
    for name, r, w in WORKLOAD_PROFILES:
        run_workload("ivfflat", IVFFLAT_PARAMS["nprobe"], name, r, w, query_vecs, insert_vecs)

    conn.close()
    print("\nWorkload profile evaluation complete.")