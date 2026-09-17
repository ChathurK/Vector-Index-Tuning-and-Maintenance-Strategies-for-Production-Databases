import psycopg2
import math
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_index import build_hnsw_index, build_ivfflat_index
from run_benchmark import run_benchmark

DB_CONFIG = dict(
    host="localhost", port=5433,
    dbname="vectorbench", user="postgres", password="research"
)

# Optimised baseline - fixed across Scalability, Workload, Maintenance phases
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 40
IVFFLAT_NPROBE = 20
K = 10

TIERS = [100_000, 500_000, 1_000_000]  # all three, sequential on this machine

def table_name_for(n):
    return f"items_{n // 1000}k" if n < 1_000_000 else "items_1m"

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    for n in TIERS:
        table = table_name_for(n)
        nlist = round(math.sqrt(n))

        print(f"\n=== Scalability: {table} ({n} vectors) - HNSW baseline ===")
        build_hnsw_index(conn, table, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)
        run_benchmark(conn, table, tier_n=n, index_type="hnsw",
                       param_value=HNSW_EF_SEARCH, k=K)

        print(f"\n=== Scalability: {table} ({n} vectors) - IVFFlat baseline ===")
        build_ivfflat_index(conn, table, lists=nlist)
        run_benchmark(conn, table, tier_n=n, index_type="ivfflat",
                       param_value=IVFFLAT_NPROBE, k=K)

    conn.close()
    print("\nScalability profiling complete (100K, 500K, 1M).")