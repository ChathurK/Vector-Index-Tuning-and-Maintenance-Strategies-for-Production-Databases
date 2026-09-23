import numpy as np
import psycopg2
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from maintenance_degradation import bulk_insert_chunk, DATA_DIR, DB_CONFIG, TABLE, MAINTENANCE_ID_OFFSET

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE id >= %s;", (MAINTENANCE_ID_OFFSET,))
        already_inserted = cur.fetchone()[0]

    print(f"{already_inserted} of 500,000 maintenance rows already present")

    if already_inserted < 500_000:
        insert_vecs = np.load(DATA_DIR / "insert_embeddings.npy", mmap_mode="r")
        print(f"Resuming insert from row {already_inserted}...")
        bulk_insert_chunk(conn, insert_vecs, already_inserted, 500_000)
    else:
        print("Already at 500,000 maintenance rows - nothing to do.")

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE};")
        final_count = cur.fetchone()[0]
    print(f"{TABLE} now has {final_count} rows")
    conn.close()