import psycopg2
import time
import csv
import pathlib

DB_CONFIG = dict(
    host="localhost", port=5433,
    dbname="vectorbench", user="postgres", password="research"
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "index_build_log.csv"

def get_index_size(conn, index_name):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size(%s);", (index_name,))
        return cur.fetchone()[0]

def drop_existing_indexes(conn, table_name):
    # only one index at a time per table, so each build measures in isolation
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s AND indexname != %s;
        """, (table_name, f"{table_name}_pkey"))
        for (idx_name,) in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
    conn.commit()

def build_hnsw_index(conn, table_name, m, ef_construction):
    index_name = f"{table_name}_hnsw_m{m}_ef{ef_construction}"
    drop_existing_indexes(conn, table_name)

    with conn.cursor() as cur:
        # disable parallel workers for build-time consistency across runs
        cur.execute("SET max_parallel_maintenance_workers = 0;")
        start = time.time()
        cur.execute(f"""
            CREATE INDEX {index_name} ON {table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """)
        build_time = time.time() - start
    conn.commit()

    size_bytes = get_index_size(conn, index_name)
    log_result(table_name, "hnsw", f"m={m},ef_construction={ef_construction}",
                build_time, size_bytes)
    print(f"HNSW built on {table_name} (m={m}, ef_construction={ef_construction}): "
          f"{build_time:.1f}s, {size_bytes/1e6:.1f}MB")
    return index_name, build_time, size_bytes

def build_ivfflat_index(conn, table_name, lists):
    index_name = f"{table_name}_ivfflat_lists{lists}"
    drop_existing_indexes(conn, table_name)

    with conn.cursor() as cur:
        cur.execute("SET max_parallel_maintenance_workers = 0;")
        start = time.time()
        cur.execute(f"""
            CREATE INDEX {index_name} ON {table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)
        build_time = time.time() - start
    conn.commit()

    size_bytes = get_index_size(conn, index_name)
    log_result(table_name, "ivfflat", f"lists={lists}", build_time, size_bytes)
    print(f"IVFFlat built on {table_name} (lists={lists}): "
          f"{build_time:.1f}s, {size_bytes/1e6:.1f}MB")
    return index_name, build_time, size_bytes

def log_result(table_name, index_type, params, build_time, size_bytes):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["table", "index_type", "params", "build_time_sec", "size_bytes"])
        writer.writerow([table_name, index_type, params, f"{build_time:.2f}", size_bytes])

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    # smoke test: default-ish parameters on the smallest tier only
    build_hnsw_index(conn, "items_100k", m=16, ef_construction=64)

    conn.close()