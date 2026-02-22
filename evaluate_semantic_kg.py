#!/usr/bin/env python3
"""
Evaluate graph similarity methods on semantic_kg_transformed dataset.
Computes similarity scores for each method and reports correlation
(Pearson + Spearman) against the ground truth similarity score.
"""

import csv
import ast
import os
import gc
import torch
import matplotlib.pyplot as plt
from scipy import stats

INPUT_FILE  = "data/semantic_kg_transformed.csv"
OUTPUT_FILE = "results/semantic_kg_eval/similarity_results.csv"

METHODS = [
    'kea_similarity',
    'kea_composite', 'kea_structural', 'kea_semantic',
    'transe_similarity', 'rotate_similarity',
    'wl_kernel_similarity',
    'aa_kea_similarity',
    'kea_bert_similarity',
]

FIELDNAMES = ['graph_1', 'graph_2', 'similarity_score_ground'] + METHODS


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_triple(triple_str):
    """Parse string representation of triples into a list of triples."""
    try:
        parsed = ast.literal_eval(triple_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                return parsed
            elif len(parsed) == 3:
                return [parsed]
        return []
    except Exception as e:
        print(f"ERROR parsing: {e}")
        return []


from Methods import (
    calculate_similarity,
    calculate_composite_similarity,
    calculate_aa_kea_similarity,
    calculate_pure_wl_kernel_similarity,
    GraphEmbeddingSimilarity,
    calculate_kea_bert_similarity,
)


def compute_row(triples1, triples2):
    """Run all methods on a pair of triple lists. Returns a dict of scores."""

    scores = {m: None for m in METHODS}
    embedding_calculator = GraphEmbeddingSimilarity(embedding_dim=50)

    try:
        kea_sim, _, _ = calculate_similarity(triples1, triples2)
        scores['kea_similarity'] = kea_sim
    except Exception:
        pass

    try:
        composite = calculate_composite_similarity(triples1, triples2, alpha=0.1, sigma=1.0)
        scores['kea_composite']  = composite['composite']
        scores['kea_structural'] = composite['structural']
        scores['kea_semantic']   = composite['semantic']
    except Exception:
        pass

    try:
        scores['transe_similarity'] = embedding_calculator.calculate_transe_similarity(triples1, triples2)
    except Exception:
        pass

    try:
        scores['rotate_similarity'] = embedding_calculator.calculate_rotate_similarity(triples1, triples2)
    except Exception:
        pass

    try:
        scores['wl_kernel_similarity'] = calculate_pure_wl_kernel_similarity(triples1, triples2)
    except Exception:
        pass

    try:
        scores['aa_kea_similarity'] = calculate_aa_kea_similarity(triples1, triples2)
    except Exception:
        pass

    try:
        scores['kea_bert_similarity'] = calculate_kea_bert_similarity(triples1, triples2)
    except Exception:
        pass

    del embedding_calculator
    return scores


# ── processing ─────────────────────────────────────────────────────────────────

def process(input_file, output_file, limit=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    results = []

    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{total}...", end=" ", flush=True)

            triples1 = parse_triple(row['graph_1'])
            triples2 = parse_triple(row['graph_2'])

            result = {
                'graph_1': row['graph_1'],
                'graph_2': row['graph_2'],
                'similarity_score_ground': row['similarity_score_ground'],
            }

            if not triples1 or not triples2:
                print("skipped (empty triples)")
                result.update({m: None for m in METHODS})
            else:
                result.update(compute_row(triples1, triples2))
                print("done")

            writer.writerow(result)
            out_f.flush()
            results.append(result)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ── correlation reporting ──────────────────────────────────────────────────────

def _collect_correlations(results):
    method_scores      = {m: [] for m in METHODS}
    ground_per_method  = {m: [] for m in METHODS}

    for r in results:
        try:
            g = float(r['similarity_score_ground'])
        except (TypeError, ValueError):
            continue
        for m in METHODS:
            if r[m] is not None:
                try:
                    method_scores[m].append(float(r[m]))
                    ground_per_method[m].append(g)
                except (TypeError, ValueError):
                    pass

    rows = []
    for m in METHODS:
        g_vals = ground_per_method[m]
        m_vals = method_scores[m]
        n = len(g_vals)
        if n < 3:
            rows.append({'method': m, 'pearson_r': None, 'pearson_p': None,
                         'spearman_r': None, 'spearman_p': None, 'n': n})
        else:
            pr, pp = stats.pearsonr(g_vals, m_vals)
            sr, sp = stats.spearmanr(g_vals, m_vals)
            rows.append({'method': m, 'pearson_r': pr, 'pearson_p': pp,
                         'spearman_r': sr, 'spearman_p': sp, 'n': n})
    return rows


def report_correlations(results, output_file):
    """Print correlation table and save it as a PNG next to the output CSV."""
    rows = _collect_correlations(results)

    col_w = 38
    print(f"\n{'='*70}")
    print("CORRELATION WITH GROUND TRUTH")
    print(f"{'='*70}")
    print(f"{'Method':<{col_w}}  {'Pearson r':>10}  {'p-value':>10}  "
          f"{'Spearman r':>11}  {'p-value':>10}  {'N':>5}")
    print(f"{'-'*70}")
    for r in rows:
        if r['pearson_r'] is None:
            print(f"{r['method']:<{col_w}}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'N/A':>11}  {'N/A':>10}  {r['n']:>5}")
        else:
            print(f"{r['method']:<{col_w}}  {r['pearson_r']:>10.4f}  {r['pearson_p']:>10.4e}"
                  f"  {r['spearman_r']:>11.4f}  {r['spearman_p']:>10.4e}  {r['n']:>5}")
    print(f"{'='*70}\n")

    headers   = ['Method', 'Pearson r', 'Pearson p', 'Spearman r', 'Spearman p', 'N']
    cell_data = []
    for r in rows:
        if r['pearson_r'] is None:
            cell_data.append([r['method'], 'N/A', 'N/A', 'N/A', 'N/A', str(r['n'])])
        else:
            cell_data.append([
                r['method'],
                f"{r['pearson_r']:.4f}",
                f"{r['pearson_p']:.2e}",
                f"{r['spearman_r']:.4f}",
                f"{r['spearman_p']:.2e}",
                str(r['n']),
            ])

    n_rows = len(cell_data)
    fig, ax = plt.subplots(figsize=(13, 0.5 * n_rows + 1.2))
    ax.axis('off')
    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(headers))))
    for col in range(len(headers)):
        tbl[0, col].set_facecolor('#2E86AB')
        tbl[0, col].set_text_props(color='white', fontweight='bold')
    for row_idx in range(1, n_rows + 1):
        bg = '#f0f4f8' if row_idx % 2 == 0 else 'white'
        for col in range(len(headers)):
            tbl[row_idx, col].set_facecolor(bg)

    plt.title('Correlation with Ground Truth Similarity Score',
              fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()

    img_path = os.path.join(os.path.dirname(output_file), 'correlation_table.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation table saved to: {img_path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate similarity methods on semantic KG dataset')
    parser.add_argument('--input',      type=str, default=INPUT_FILE)
    parser.add_argument('--output',     type=str, default=OUTPUT_FILE)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit rows (for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("Semantic KG Similarity Evaluation")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}\n")

    results = process(args.input, args.output, limit=args.limit)

    print(f"\nResults written to: {args.output}")
    print(f"Total rows processed: {len(results)}")

    report_correlations(results, args.output)
