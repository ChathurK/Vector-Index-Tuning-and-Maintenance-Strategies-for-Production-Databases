import psycopg2

DB_CONFIG = dict(
    host="localhost", port=5432,
    dbname="vectorbench", user="postgres", password="research"
)

# TABLE = "items_100k"
# TABLE = "items_500k"
TABLE = "items_1m"

def main():
    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cur:
        # Fetch one sample vector (id = 0)
        cur.execute(f"SELECT embedding FROM {TABLE} WHERE id = 0;")
        row = cur.fetchone()

        if row is None:
            print("No vector found for id=0")
            return

        vec_text = row[0]              # pgvector returns text like "[0.12,0.34,...]"
        dims = len(vec_text.split(","))  # count number of comma-separated values

        print(f"Sample vector from {TABLE}:")
        print(f"  Raw text: {vec_text[:60]}...")   # preview first part
        print(f"  Dimension: {dims}")

    conn.close()

if __name__ == "__main__":
    main()
