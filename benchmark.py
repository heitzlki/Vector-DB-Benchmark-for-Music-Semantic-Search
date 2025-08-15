import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.metrics import hits_at_k
from databases.qdrant_client import Qdrant
from databases.milvus_client import Milvus
from databases.weaviate_client import WeaviateDB


def load_embeddings(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    vectors = np.array(df["embedding"].tolist(), dtype=np.float32)
    payloads = df[["track", "artist", "genre", "seeds", "text"]].to_dict(
        orient="records"
    )
    return vectors, payloads


def embed_query(q: str, model) -> List[float]:
    v = model.encode([q], normalize_embeddings=True)[0]
    return v.tolist()


def get_db(name: str, args) -> Any:
    name = name.lower()
    if name == "qdrant":
        return Qdrant(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    if name == "milvus":
        return Milvus(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
    if name == "weaviate":
        return WeaviateDB(url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Original CSV path")
    ap.add_argument("--embeddings", required=True, help="Parquet with embeddings")
    ap.add_argument(
        "--dbs", nargs="+", default=["qdrant"], help="Which DBs to benchmark"
    )
    ap.add_argument("--queries", default="queries.yaml", help="YAML file with queries")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--warmup", type=int, default=1, help="Warm-up passes per DB (not timed)"
    )
    ap.add_argument("--query_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    vectors, payloads = load_embeddings(args.embeddings)
    dim = vectors.shape[1]
    # Check normalization: all vectors should have norm ~1
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(
            "Warning: Not all vectors are normalized! Min norm:",
            norms.min(),
            "Max norm:",
            norms.max(),
        )
    else:
        print("All vectors are normalized (L2 norm ~1)")

    with open(args.queries, "r") as f:
        cfg = yaml.safe_load(f)
    queries = cfg["queries"]

    # Preload the query embedding model once to avoid repeated loads
    from sentence_transformers import SentenceTransformer

    query_model = SentenceTransformer(args.query_model)

    # Exact baseline recall helper (cosine on normalized vectors)
    def exact_topk_indices(qv: np.ndarray, mat: np.ndarray, k: int) -> np.ndarray:
        # qv: shape (D,), mat: shape (N, D)
        sims = mat @ qv  # dot product = cosine since normalized
        if k >= len(sims):
            return np.argsort(-sims)
        idx = np.argpartition(-sims, k)[:k]
        # sort top-k for stable order
        return idx[np.argsort(-sims[idx])]

    results = {}
    # Add config metadata
    results["_config"] = {
        "batch_size": 2000,
        "hnsw_params": {"M": 16, "efConstruction": 128, "ef": 128},
        "metric": "COSINE",
        "model": args.query_model,
        "dataset_size": len(vectors),
        "repetitions": args.repetitions,
    }
    for db_name in args.dbs:
        print(f"Setting up {db_name}")
        db = get_db(db_name, args)
        t0 = time.time()
        db.setup(dim=dim)
        t1 = time.time()
        db.upsert(vectors=vectors.tolist(), payloads=payloads)
        ingest_time = time.time() - t1
        setup_time = t1 - t0

        # Optional warm-up passes (not timed)
        for _ in range(max(0, args.warmup)):
            for q in cfg["queries"]:
                q_vec = embed_query(q["text"], query_model)
                _ = db.search(q_vec, top_k=args.topk)

        latencies = []
        hits = []
        recalls = []
        import random

        for rep in range(args.repetitions):
            # randomize query order each rep to avoid order effects
            order = list(range(len(queries)))
            random.shuffle(order)
            for idx in order:
                q = queries[idx]
                q_vec = embed_query(q["text"], query_model)
                s0 = time.time()
                res = db.search(q_vec, top_k=args.topk)
                latency = time.time() - s0
                latencies.append(latency)
                res_payloads = [r["payload"] for r in res]
                hitk = hits_at_k(res_payloads, q["expected"])
                hits.append(hitk)

                # Compute exact recall@k using baseline
                true_idx = exact_topk_indices(
                    np.array(q_vec, dtype=np.float32), vectors, args.topk
                )
                true_set = set(int(i) for i in true_idx.tolist())
                # Map result ids to row indices
                res_ids = []
                for r in res:
                    # Always try to use row_id from payload for recall matching
                    pid = r.get("payload", {}).get("row_id")
                    if isinstance(pid, (int, np.integer)):
                        res_ids.append(int(pid))
                    else:
                        rid = r.get("id")
                        if isinstance(rid, (int, np.integer)):
                            res_ids.append(int(rid))
                inter = len(true_set.intersection(set(res_ids)))
                recalls.append(inter / float(args.topk) if args.topk > 0 else 0.0)

        avg_latency = float(np.mean(latencies)) if latencies else None
        p50_latency = float(np.percentile(latencies, 50)) if latencies else None
        p95_latency = float(np.percentile(latencies, 95)) if latencies else None
        avg_hitk = float(np.mean(hits)) if hits else None
        avg_recall = float(np.mean(recalls)) if recalls else None

        results[db_name] = {
            "setup_time_sec": setup_time,
            "ingest_time_sec": ingest_time,
            "avg_query_latency_sec": avg_latency,
            "p50_query_latency_sec": p50_latency,
            "p95_query_latency_sec": p95_latency,
            f"avg_hits_at_{args.topk}": avg_hitk,
            f"avg_recall_at_{args.topk}": avg_recall,
        }

        db.teardown()

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        import json

        json.dump(results, f, indent=2)
    print("Saved metrics.json")

    # --- All plotting code (original and new) ---
    # --- Original bar charts (restored) ---
    labels = [k for k in results.keys() if k != "_config"]
    ingest = [results[k]["ingest_time_sec"] for k in labels]
    latency = [results[k]["avg_query_latency_sec"] for k in labels]
    hitk = [results[k][f"avg_hits_at_{args.topk}"] for k in labels]
    recallk = [results[k][f"avg_recall_at_{args.topk}"] for k in labels]

    plt.figure()
    plt.bar(labels, ingest)
    plt.title("Ingest time (sec)")
    plt.ylabel("seconds")
    plt.savefig(out_dir / "ingest_time.png", bbox_inches="tight")

    plt.figure()
    plt.bar(labels, latency)
    plt.title("Average query latency (sec)")
    plt.ylabel("seconds")
    plt.savefig(out_dir / "avg_latency.png", bbox_inches="tight")

    plt.figure()
    plt.bar(labels, hitk)
    plt.title(f"Average hits in top {args.topk} (weak labels)")
    plt.ylabel("count")
    plt.savefig(out_dir / "hits_at_k.png", bbox_inches="tight")

    plt.figure()
    plt.bar(labels, recallk)
    plt.title(f"Average recall@{args.topk} vs exact baseline")
    plt.ylim(0, 1)
    plt.ylabel("recall")
    plt.savefig(out_dir / "recall_at_k.png", bbox_inches="tight")

    # Summary image as a combined simple layout
    plt.figure(figsize=(8, 12))
    plt.subplot(4, 1, 1)
    plt.bar(labels, ingest)
    plt.title("Ingest time")

    plt.subplot(4, 1, 2)
    plt.bar(labels, latency)
    plt.title("Avg query latency")

    plt.subplot(4, 1, 3)
    plt.bar(labels, hitk)
    plt.title(f"Avg hits in top {args.topk}")

    plt.subplot(4, 1, 4)
    plt.bar(labels, recallk)
    plt.title(f"Avg recall@{args.topk}")
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", bbox_inches="tight")
    print("Saved original bar charts in results/")

    # --- New metrics table and labeled bar charts ---
    metrics = [
        ("Ingest Time (sec)", ingest),
        ("Avg Query Latency (sec)", latency),
        (f"Avg Hits in Top {args.topk}", hitk),
        (f"Avg Recall@{args.topk}", recallk),
    ]

    # Create a table visualization using matplotlib
    cell_text = []
    for metric_name, values in metrics:
        row = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 2 + len(metrics)))
    table = ax.table(
        cellText=cell_text,
        rowLabels=[m[0] for m in metrics],
        colLabels=labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.axis("off")
    plt.title("Benchmark Metrics Table", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_table.png", bbox_inches="tight")
    print("Saved metrics_table.png in results/")


    # Grouped bar charts for each metric with value labels
    for metric_name, values in metrics:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
        bars = ax.bar(labels, values, color="skyblue")
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        plt.tight_layout()
        fname = (
            metric_name.lower()
            .replace(" ", "_")
            .replace("@", "at")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            + ".png"
        )
        plt.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname} in results/")

    # --- Summary image with all four bar charts (2x2 grid) ---
    fig, axs = plt.subplots(2, 2, figsize=(max(10, len(labels)*2.5), 10))
    chart_data = [
        ("Ingest Time (sec)", ingest),
        ("Avg Query Latency (sec)", latency),
        (f"Avg Hits in Top {args.topk}", hitk),
        (f"Avg Recall@{args.topk}", recallk),
    ]
    for ax, (title, values) in zip(axs.flat, chart_data):
        bars = ax.bar(labels, values, color="skyblue")
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Benchmark Summary: All Metrics", fontsize=18)
    plt.savefig(out_dir / "summary_all.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved summary_all.png in results/")

    # Exclude '_config' from plotting/metrics
    labels = [k for k in results.keys() if k != "_config"]
    metrics = [
        ("Ingest Time (sec)", [results[k]["ingest_time_sec"] for k in labels]),
        (
            "Avg Query Latency (sec)",
            [results[k]["avg_query_latency_sec"] for k in labels],
        ),
        (
            f"Avg Hits in Top {args.topk}",
            [results[k][f"avg_hits_at_{args.topk}"] for k in labels],
        ),
        (
            f"Avg Recall@{args.topk}",
            [results[k][f"avg_recall_at_{args.topk}"] for k in labels],
        ),
    ]

    # Create a table visualization using matplotlib

    cell_text = []
    for metric_name, values in metrics:
        row = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 2 + len(metrics)))
    table = ax.table(
        cellText=cell_text,
        rowLabels=[m[0] for m in metrics],
        colLabels=labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.axis("off")
    plt.title("Benchmark Metrics Table", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_table.png", bbox_inches="tight")
    print("Saved metrics_table.png in results/")


if __name__ == "__main__":
    main()
