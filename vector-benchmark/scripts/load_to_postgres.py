import numpy as np
import psycopg2
import io
import time
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "embeddings"

DB_CONFIG = dict(
    host="localhost", port=5432,
    dbname="vectorbench", user="postgres", password="research"
)

CHUNK_SIZE = 20_000  # rows per COPY batch, keeps in-memory text buffer small

def vector_to_pg_literal(vec):
    # pgvector text format: [0.123,0.456,...]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def create_table(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        cur.execute(f"""
            CREATE UNLOGGED TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                embedding VECTOR(768)
            );
        """)
        # UNLOGGED skips WAL writes during bulk load, roughly 2x faster on
        # a single-disk laptop setup; safe here since this is regenerable
        # benchmark data, not data you need crash-durability for.
    conn.commit()

def load_tier(conn, n):
    table_name = f"items_{n // 1000}k" if n < 1_000_000 else "items_1m"
    print(f"--- Loading tier {n} into {table_name} ---")

    create_table(conn, table_name)

    vecs = np.load(DATA_DIR / f"embeddings_{n}.npy", mmap_mode="r")

    start = time.time()
    with conn.cursor() as cur:
        for chunk_start in range(0, n, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            buf = io.StringIO()
            for i in range(chunk_start, chunk_end):
                buf.write(f"{i}\t{vector_to_pg_literal(vecs[i])}\n")
            buf.seek(0)
            cur.copy_expert(
                f"COPY {table_name} (id, embedding) FROM STDIN WITH (FORMAT text)",
                buf
            )
            if chunk_start % (CHUNK_SIZE * 5) == 0:
                elapsed = time.time() - start
                print(f"  {chunk_end}/{n} rows loaded, {elapsed:.1f}s elapsed")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cur.fetchone()[0]
    elapsed = time.time() - start
    print(f"Tier {n} loaded: {count} rows in {elapsed:.1f}s -> {table_name}")
    assert count == n, f"Row count mismatch! Expected {n}, got {count}"

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    for n in [1_000_000]:
        load_tier(conn, n)
    conn.close()
    print("All tiers loaded successfully.")