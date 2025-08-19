import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mplcyberpunk

plt.style.use("cyberpunk")


def load_metrics(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Infer DB names (top-level keys excluding _config)
    db_names = [k for k in data.keys() if k != "_config"]
    # Extract k values (assume consistent labels like "k=5")
    # Use the first DB to read available ks
    first_db = data[db_names[0]]
    k_values = sorted([int(k.split("=")[1]) for k in first_db.keys()])
    # Build structures
    ingest = {}
    qps_list = {}
    recall50 = {}
    latency = {db: [] for db in db_names}
    for db in db_names:
        db_block = data[db]
        # Ingest/setup stored redundantly per k; read once from the first k
        first_k_label = f"k={k_values[0]}"
        ingest[db] = float(db_block[first_k_label]["ingest_time_sec"])
        # Aggregate QPS over ks (average)
        qps_vals = []
        for k in k_values:
            k_label = f"k={k}"
            qps_vals.append(float(db_block[k_label]["avg_qps"]))
        qps_list[db] = float(np.mean(qps_vals))
        # Recall@50 from k=50 if present; else best effort
        if 50 in k_values:
            recall50[db] = float(db_block["k=50"]["avg_recall_at_50"])
        else:
            # Fallback: use largest available k's recall if exact 50 not present
            max_k = max(k_values)
            recall_key = f"k={max_k}"
            # pick any recall field in that block
            rec = None
            for key in db_block[recall_key].keys():
                if "recall" in key:
                    rec = float(db_block[recall_key][key])
            recall50[db] = rec if rec is not None else float("nan")
        # Latency series per k
        for k in k_values:
            k_label = f"k={k}"
            latency[db].append(float(db_block[k_label]["avg_query_latency_sec"]))
    return db_names, k_values, ingest, qps_list, recall50, latency


def add_value_labels_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_value_labels_points(ax, x, y, fmt="{:.4f}"):
    for xi, yi in zip(x, y):
        ax.annotate(
            fmt.format(yi),
            xy=(xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_grouped_bars(db_names, ingest, qps_list, recall50, out_path: Path):
    # Prepare data in consistent order
    x = np.arange(len(db_names))
    width = 0.25

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    bars1 = ax.bar(
        x - width, [ingest[d] for d in db_names], width, label="Ingest Time (s)"
    )
    bars2 = ax.bar(x, [qps_list[d] for d in db_names], width, label="QPS (avg)")
    bars3 = ax.bar(x + width, [recall50[d] for d in db_names], width, label="Recall@50")

    mplcyberpunk.add_bar_gradient(bars=bars1)
    mplcyberpunk.add_bar_gradient(bars=bars2)
    mplcyberpunk.add_bar_gradient(bars=bars3)

    ax.set_xticks(x)
    ax.set_xticklabels(db_names)
    ax.set_title("Ingest Time, QPS (avg), and Recall@50")
    ax.legend()

    # Add labels
    add_value_labels_bars(ax, bars1, fmt="{:.2f}")
    add_value_labels_bars(ax, bars2, fmt="{:.1f}")
    add_value_labels_bars(ax, bars3, fmt="{:.3f}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_lines(db_names, k_values, latency, out_path: Path):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    lines = []
    for db in db_names:
        y = latency[db]
        (line,) = ax.plot(k_values, y, marker="o", label=db)
        lines.append(line)
        add_value_labels_points(ax, k_values, y, fmt="{:.4f}")

    mplcyberpunk.make_lines_glow(lines)

    ax.set_xticks(k_values)
    ax.set_xlabel("k")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs. k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def stack_images_vertically(img_paths, out_path: Path):
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    canvas = Image.new("RGB", (max(widths), sum(heights)), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_benchmarks.py <metrics.json> <output_prefix>")
        sys.exit(1)
    json_path = Path(sys.argv[1])
    out_prefix = Path(sys.argv[2])

    db_names, k_values, ingest, qps_list, recall50, latency = load_metrics(json_path)

    bars_path = out_prefix.with_name(out_prefix.name + "_bars.png")
    latency_path = out_prefix.with_name(out_prefix.name + "_latency.png")
    combined_path = out_prefix.with_suffix(".png")

    plot_grouped_bars(db_names, ingest, qps_list, recall50, bars_path)
    plot_latency_lines(db_names, k_values, latency, latency_path)
    stack_images_vertically([bars_path, latency_path], combined_path)

    print(f"Saved:\n- {bars_path}\n- {latency_path}\n- {combined_path}")


if __name__ == "__main__":
    main()
