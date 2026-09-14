import psycopg2
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_index import build_hnsw_index, build_ivfflat_index
from run_benchmark import run_benchmark

DB_CONFIG = dict(
    host="localhost", port=5432,
    dbname="vectorbench", user="postgres", password="research"
)

TABLE = "items_500k"
TIER_N = 500_000
K = 10

# per Chapter 3, Table 5
HNSW_M_VALUES = [16, 32, 64]
HNSW_EF_CONSTRUCTION_VALUES = [64, 128, 200]
HNSW_EF_SEARCH_SWEEP = [10, 20, 40, 80, 160]  # runtime sweep, not fixed in Ch3 - adjust as needed

# nlist per Ch3: "sqrt(N), and values above/below"
import math
SQRT_N = round(math.sqrt(TIER_N))
IVFFLAT_NLIST_VALUES = [SQRT_N // 2, SQRT_N, SQRT_N * 2]
IVFFLAT_NPROBE_SWEEP = [1, 5, 10, 20]  # per Ch3, Table 5


def run_hnsw_sweep(conn):
    for m in HNSW_M_VALUES:
        for ef_c in HNSW_EF_CONSTRUCTION_VALUES:
            print(f"\n=== Building HNSW: M={m}, ef_construction={ef_c} ===")
            build_hnsw_index(conn, TABLE, m=m, ef_construction=ef_c)

            for ef_s in HNSW_EF_SEARCH_SWEEP:
                run_benchmark(conn, TABLE, tier_n=TIER_N,
                               index_type="hnsw", param_value=ef_s, k=K)


def run_ivfflat_sweep(conn):
    for nlist in IVFFLAT_NLIST_VALUES:
        print(f"\n=== Building IVFFlat: nlist={nlist} ===")
        build_ivfflat_index(conn, TABLE, lists=nlist)

        for nprobe in IVFFLAT_NPROBE_SWEEP:
            run_benchmark(conn, TABLE, tier_n=TIER_N,
                           index_type="ivfflat", param_value=nprobe, k=K)


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    print(f"Starting Phase 3 tuning sweep on {TABLE} ({TIER_N} vectors)")
    print(f"HNSW builds: {len(HNSW_M_VALUES) * len(HNSW_EF_CONSTRUCTION_VALUES)}, "
          f"benchmark runs: {len(HNSW_M_VALUES) * len(HNSW_EF_CONSTRUCTION_VALUES) * len(HNSW_EF_SEARCH_SWEEP)}")
    print(f"IVFFlat builds: {len(IVFFLAT_NLIST_VALUES)}, "
          f"benchmark runs: {len(IVFFLAT_NLIST_VALUES) * len(IVFFLAT_NPROBE_SWEEP)}")

    run_hnsw_sweep(conn)
    run_ivfflat_sweep(conn)

    conn.close()
    print("\nPhase 3 tuning sweep complete.")