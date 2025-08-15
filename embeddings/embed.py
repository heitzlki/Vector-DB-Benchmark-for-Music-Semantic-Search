import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Choose either sentence-transformers or OpenAI
USE_OPENAI = False

def build_text_row(row: pd.Series) -> str:
    seeds = row.get("seeds", "")
    return f"{row['track']} by {row['artist']}. Genre: {row.get('genre','')}. Tags: {seeds}"

def embed_st(df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = df["text"].tolist()
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vectors

def embed_openai(df: pd.DataFrame, model_name: str = "text-embedding-3-large") -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    vectors = []
    texts = df["text"].tolist()
    batch = 256
    for i in tqdm(range(0, len(texts), batch), desc="OpenAI embeddings"):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(model=model_name, input=chunk)
        for item in resp.data:
            vectors.append(item.embedding)
    return np.array(vectors, dtype=np.float32)

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with the Kaggle dataset")
    ap.add_argument("--out", required=True, help="Where to store embeddings parquet")
    ap.add_argument("--use_openai", action="store_true", help="Use OpenAI embeddings")
    ap.add_argument("--model", default=None, help="Embedding model name")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["text"] = df.apply(build_text_row, axis=1)

    if args.use_openai or USE_OPENAI:
        model_name = args.model or "text-embedding-3-large"
        vecs = embed_openai(df, model_name=model_name)
    else:
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        vecs = embed_st(df, model_name=model_name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet with vectors as list column
    df_out = df.copy()
    df_out["embedding"] = list(map(lambda v: v.tolist(), vecs))
    df_out.to_parquet(out_path, index=False)
    print(f"Wrote {len(df_out)} rows to {out_path}")

if __name__ == "__main__":
    main()
