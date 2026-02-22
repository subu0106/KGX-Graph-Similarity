#!/usr/bin/env python3
"""
LLM Benchmark Similarity Evaluation

For each dataset × LLM CSV file in this directory, computes AA-KEA similarity
between gold_kg and llm_kg, then produces a cross-model comparison summary.

Parallel processing notes:
  - Uses ProcessPoolExecutor with spawn start method (required for CUDA on Linux)
  - Each worker loads SBERT once into its own GPU context, then handles its batch
  - AA-KEA is imported lazily inside the worker to avoid fork+CUDA deadlocks

Output layout (relative to this script's directory):
  results/<dataset>/<file_stem>_scored.csv   -- per-row scores
  results/summary.csv                        -- mean/median/std per model × dataset
  results/comparison.png                     -- bar + box plots
"""

import os
import ast
import csv
import glob
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

# ── config ────────────────────────────────────────────────────────────────────
INPUT_GLOB        = os.path.join(SCRIPT_DIR, "**", "*.csv")
RESULTS_DIR       = os.path.join(SCRIPT_DIR, "results")
DEFAULT_WORKERS   = 8
DEFAULT_BATCH_SIZE = 10

MODEL_LABELS = {
    "Llama-2-7b-chat-hf":       "Llama-2-7B",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "gemma-7b-it":              "Gemma-7B",
}

DATASET_LABELS = {
    "messaqa": "MeSS-AQA",
    "pubmed":  "PubMedQA",
}

MODEL_COLORS = {
    "Llama-2-7B": "#2E86AB",
    "Mistral-7B": "#A23B72",
    "Gemma-7B":   "#6A994E",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_kg(kg_str):
    """Parse a KG string into a list of [s, p, o] triples."""
    try:
        parsed = ast.literal_eval(kg_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                return [t for t in parsed if len(t) == 3]
            elif len(parsed) == 3 and not isinstance(parsed[0], list):
                return [parsed]
        return []
    except Exception:
        return []


def infer_names(filepath):
    """Return (dataset_label, model_label, dataset_raw) from file path."""
    parts       = filepath.replace("\\", "/").split("/")
    dataset_raw = parts[-2]
    stem        = os.path.splitext(parts[-1])[0]

    dataset = DATASET_LABELS.get(dataset_raw, dataset_raw)

    model = stem
    for key, label in MODEL_LABELS.items():
        if key in stem:
            model = label
            break

    return dataset, model, dataset_raw


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ── worker (top-level required for ProcessPoolExecutor pickling) ───────────────

def _process_batch(batch_data):
    """
    Worker: receives list of (row_id, question, gold_kg_str, llm_kg_str).
    SBERT loads on first call, stays cached for the whole batch.
    """
    from Methods.aa_kea import calculate_aa_kea_similarity  # lazy import: avoids fork+CUDA deadlock

    results = []
    for row_id, question, gold_str, llm_str in batch_data:
        gold_triples = parse_kg(gold_str)
        llm_triples  = parse_kg(llm_str)

        if not gold_triples or not llm_triples:
            score = None
        else:
            try:
                score = calculate_aa_kea_similarity(gold_triples, llm_triples)
            except Exception:
                score = None

        results.append((row_id, question, score))

    return results


# ── per-file scoring ──────────────────────────────────────────────────────────

def score_file(filepath, workers=DEFAULT_WORKERS, batch_size=DEFAULT_BATCH_SIZE):
    """
    Compute AA-KEA similarity for every row in one CSV using parallel batches.
    Returns list of dicts with row_id, question, aa_kea_similarity, dataset, model.
    """
    dataset, model, _ = infer_names(filepath)
    print(f"\n{'─'*60}")
    print(f"Dataset : {dataset}  |  Model: {model}")
    print(f"File    : {os.path.basename(filepath)}")

    with open(filepath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    batches = [
        [(r.get("row_id", i), r.get("question", ""),
          r.get("gold_kg", ""), r.get("llm_kg", ""))
         for i, r in enumerate(chunk)]
        for chunk in _chunk(rows, batch_size)
    ]
    n_batches = len(batches)
    print(f"Rows: {total}  |  Workers: {workers}  |  Batches: {n_batches}")

    ordered   = [None] * n_batches
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_process_batch, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                ordered[idx] = future.result()
            except Exception as e:
                print(f"\n  Batch {idx} failed: {e}")
                ordered[idx] = []
            completed += 1
            done = sum(len(b) for b in ordered if b is not None)
            print(f"  Batches done: {completed}/{n_batches}  ({done}/{total} rows)", flush=True)

    records = []
    for batch_results in ordered:
        if batch_results:
            for row_id, question, score in batch_results:
                records.append({
                    "row_id":            row_id,
                    "question":          question,
                    "aa_kea_similarity": score,
                    "dataset":           dataset,
                    "model":             model,
                })

    valid = [r["aa_kea_similarity"] for r in records if r["aa_kea_similarity"] is not None]
    if valid:
        print(f"  Done — {len(valid)}/{total} scored | "
              f"mean={np.mean(valid):.4f}  median={np.median(valid):.4f}  std={np.std(valid):.4f}")
    else:
        print(f"  Done — 0/{total} scored (all failed)")

    return records


def save_scored_csv(records, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["row_id", "question", "dataset", "model", "aa_kea_similarity"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ── aggregation & plotting ────────────────────────────────────────────────────

def build_summary(all_records):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_records:
        if r["aa_kea_similarity"] is not None:
            groups[(r["dataset"], r["model"])].append(r["aa_kea_similarity"])

    summary = []
    for (dataset, model), scores in sorted(groups.items()):
        summary.append({
            "dataset": dataset,
            "model":   model,
            "n":       len(scores),
            "mean":    np.mean(scores),
            "median":  np.median(scores),
            "std":     np.std(scores),
            "min":     np.min(scores),
            "max":     np.max(scores),
        })
    return summary


def save_summary_csv(summary, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["dataset", "model", "n", "mean", "median", "std", "min", "max"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def print_summary_table(summary):
    print(f"\n{'='*72}")
    print("LLM BENCHMARK — AA-KEA SIMILARITY SUMMARY")
    print(f"{'='*72}")
    print(f"{'Dataset':<14}  {'Model':<14}  {'N':>5}  {'Mean':>8}  "
          f"{'Median':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"{'─'*72}")
    for row in summary:
        print(f"{row['dataset']:<14}  {row['model']:<14}  {row['n']:>5}  "
              f"{row['mean']:>8.4f}  {row['median']:>8.4f}  {row['std']:>8.4f}  "
              f"{row['min']:>8.4f}  {row['max']:>8.4f}")
    print(f"{'='*72}\n")


def plot_comparison(all_records, out_path):
    from collections import defaultdict

    datasets = sorted({r["dataset"] for r in all_records})
    models   = sorted({r["model"]   for r in all_records})

    groups = defaultdict(list)
    for r in all_records:
        if r["aa_kea_similarity"] is not None:
            groups[(r["dataset"], r["model"])].append(r["aa_kea_similarity"])

    n_datasets = len(datasets)
    fig, axes = plt.subplots(2, n_datasets, figsize=(6 * n_datasets, 10))
    if n_datasets == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, dataset in enumerate(datasets):
        ax_bar = axes[0][col]
        ax_box = axes[1][col]

        means  = [np.mean(groups[(dataset, m)]) if groups[(dataset, m)] else 0 for m in models]
        stds   = [np.std(groups[(dataset, m)])  if groups[(dataset, m)] else 0 for m in models]
        colors = [MODEL_COLORS.get(m, "#888888") for m in models]

        x    = np.arange(len(models))
        bars = ax_bar.bar(x, means, yerr=stds, capsize=5,
                          color=colors, alpha=0.75, edgecolor="black", linewidth=0.8)
        for bar, mean in zip(bars, means):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{mean:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        ax_bar.set_title(f"{dataset}\nMean AA-KEA Similarity (± std)",
                         fontsize=11, fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models, fontsize=9)
        ax_bar.set_ylabel("AA-KEA Similarity", fontsize=9)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

        data_per_model = [groups[(dataset, m)] for m in models]
        bp = ax_box.boxplot(data_per_model, patch_artist=True,
                            tick_labels=models, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_box.set_title(f"{dataset}\nScore Distribution",
                         fontsize=11, fontweight="bold")
        ax_box.set_ylabel("AA-KEA Similarity", fontsize=9)
        ax_box.set_ylim(0, 1.05)
        ax_box.grid(axis="y", linestyle="--", alpha=0.4)

    patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#888888"), label=m) for m in models]
    fig.legend(handles=patches, loc="upper center", ncol=len(models),
               fontsize=10, title="Model", title_fontsize=10,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("LLM KG Quality — AA-KEA Similarity (gold_kg vs llm_kg)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(workers=DEFAULT_WORKERS, batch_size=DEFAULT_BATCH_SIZE):
    csv_files = sorted(glob.glob(INPUT_GLOB, recursive=True))
    csv_files = [f for f in csv_files if "results" not in f.replace("\\", "/")]

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} file(s) to process:")
    for f in csv_files:
        print(f"  {os.path.relpath(f, SCRIPT_DIR)}")

    all_records = []

    for filepath in csv_files:
        _, _, dataset_raw = infer_names(filepath)
        stem = os.path.splitext(os.path.basename(filepath))[0]

        records = score_file(filepath, workers=workers, batch_size=batch_size)
        all_records.extend(records)

        out_csv = os.path.join(RESULTS_DIR, dataset_raw, f"{stem}_scored.csv")
        save_scored_csv(records, out_csv)
        print(f"  Saved → {os.path.relpath(out_csv, SCRIPT_DIR)}")

    summary = build_summary(all_records)
    print_summary_table(summary)

    summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
    save_summary_csv(summary, summary_csv)
    print(f"Summary CSV saved → {os.path.relpath(summary_csv, SCRIPT_DIR)}")

    plot_path = os.path.join(RESULTS_DIR, "comparison.png")
    plot_comparison(all_records, plot_path)


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # required for CUDA + multiprocessing

    parser = argparse.ArgumentParser(description="LLM Benchmark AA-KEA Similarity")
    parser.add_argument("--workers",    type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Rows per batch (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Benchmark Similarity Evaluation (AA-KEA)")
    print("=" * 60)
    print(f"Workers: {args.workers}  |  Batch size: {args.batch_size}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: not available, using CPU")
    except ImportError:
        pass
    print()

    main(workers=args.workers, batch_size=args.batch_size)
