import psycopg2
from sentence_transformers import SentenceTransformer

# Test 1: database connection
conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="vectorbench", user="postgres", password="research"
)
cur = conn.cursor()
cur.execute("SELECT 1;")
print("Postgres OK:", cur.fetchone())
conn.close()

# Test 2: embedding generation
model = SentenceTransformer("all-mpnet-base-v2")
vec = model.encode("test sentence for dimension check")
print("Embedding dimension:", len(vec))  # should print 768