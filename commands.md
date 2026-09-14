- [PostgreSQL + pgvector in Docker](#postgresql--pgvector-in-docker)
- [Set up the Python environment](#set-up-the-python-environment)
- [Dataset generation pipeline](#dataset-generation-pipeline)
- [The benchmark harness](#the-benchmark-harness)

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

> <container_name> is the name of the container running PostgreSQL. You can find it by running `docker compose ps`.
> "pgvector-research" is the name of the container running PostgreSQL.
```
docker exec -it <container_name> psql -U postgres
```
> Here -d vectorbench is the database name, and if you don't mention it, it will default to the user name (postgres) as the database name.
```
docker exec -it <container_name> psql -U postgres -d vectorbench -c "CREATE EXTENSION IF NOT EXISTS vector;"
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
> Stops the main processes without deleting anything; `start` wakes them right back up.
```
docker stop -t 60 pgvector-research
```
```
docker start pgvector-research
```
```
docker logs -f pgvector-research
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
python .\scripts\generate_embeddings.py *> logs\gen_1m.log
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
# The benchmark harness
```
python .\scripts\compute_ground_truth.py
```
```
python .\scripts\load_to_postgres.py
```
```sql
ALTER TABLE items_100k SET LOGGED;
ALTER TABLE items_500k SET LOGGED;
ALTER TABLE items_1m SET LOGGED;

-- Re-verify counts
SELECT count(*) FROM items_100k;
SELECT count(*) FROM items_500k;
SELECT count(*) FROM items_1m;
```
```
docker exec pgvector-research pg_dump -U postgres -d vectorbench -F c -f /tmp/vectorbench_backup.dump
docker cp pgvector-research:/tmp/vectorbench_backup.dump C:\research\vector-benchmark\backups\vectorbench_backup.dump
```
```
python .\scripts\fetch_sample.py
```
```
python .\scripts\build_index.py
```
```
python .\scripts\run_benchmark.py
```
```
python .\scripts\sweep_phase3_tuning.py *> logs\phase3_tuning.log
```