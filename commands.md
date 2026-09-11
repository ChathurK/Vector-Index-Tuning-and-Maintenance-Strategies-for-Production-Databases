- [PostgreSQL + pgvector in Docker](#postgresql--pgvector-in-docker)
- [Set up the Python environment](#set-up-the-python-environment)
- [Dataset generation pipeline](#dataset-generation-pipeline)

# PostgreSQL + pgvector in Docker
```
docker --version
```
```
docker compose up -d
```
```
docker compose logs -f
```
```
docker compose ps
```
```
docker exec -it <container_name> psql -U postgres
```
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
```sql
-- Check the version of the vector extension
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```
```sql
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(3)
);
```
```sql
INSERT INTO items (embedding) VALUES
  ('[4,5,6]'),
  ('[1,2,3]'),
  ('[28,29,30]'),
  ('[7,8,9]'),
  ('[22,23,24]'),
  ('[16,17,18]'),
  ('[19,20,21]'),
  ('[10,11,12]'),
  ('[25,26,27]'),
  ('[13,14,15]');
```
Performs exact nearest neighbor search
```sql
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```
Performs approximate nearest neighbor search using HNSW index
```sql
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
```
```sql
CREATE INDEX ON items USING hnsw (embedding vector_l1_ops);
```
```sql
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
```
```sql
CREATE INDEX items_hnsw_idx ON items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX items_ivfflat_idx ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
```
- View indexes for a specific table
```sql
\d table_name
```
- List all indexes in the current schema
```sql
\di
```
- List all indexes with extra details (like size and description)
```sql
\di+
```
```sql
DROP INDEX items_hnsw_idx;
```
```sql
DROP INDEX items_ivfflat_idx;
SELECT id, embedding, embedding <-> '[1,2,3]' AS distance
FROM items
ORDER BY embedding <-> '[1,2,3]'
LIMIT 3;
```
```sql
DROP TABLE items;
```
Confirm the data volume survives a restart
```
docker compose down
```
```
docker compose up -d
```
# Set up the Python environment

```
python --version
```
```
cd .\research\vector-benchmark\
```
```
python -m venv venv
```
```
.\venv\Scripts\activate
```
```
python -m pip install --upgrade pip
```
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```
pip install -r requirements.txt
```
```
python smoke_test_env.py
```
```
pip freeze > requirements-lock.txt
```
```
deactivate
```
# Dataset generation pipeline
```
mkdir data, scripts, embeddings
```
```
python scripts\generate_embeddings.py
```
```
python -c "import numpy as np; a = np.load('embeddings/embeddings_500.npy'); print(a.shape, a.dtype)"
```
```
python .\scripts\validate_tier.py 100000
```
```
python .\sanity_check.py
```