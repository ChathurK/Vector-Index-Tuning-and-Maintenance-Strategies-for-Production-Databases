import pandas as pd
for n in [100_000, 500_000]:
    df = pd.read_csv(f"../embeddings/sentences_{n}.csv")
    lengths = df["text"].astype(str).str.len()
    print(n, "min:", lengths.min(), "max:", lengths.max(),
          "violations:", (lengths < 20).sum())