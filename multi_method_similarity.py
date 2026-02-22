#!/usr/bin/env python3
"""
Multi-method graph similarity comparison
Compares gemini answer pairs using:
1. KEA (Graph Kernel - Weisfeiler-Lehman + Clustering)
2. KEA Composite (Gaussian + WL Kernel)
3. KEA Semantic (Gaussian kernel only)
4. TransE graph embedding
5. RotatE graph embedding
6. Pure WL kernel (structural only)
7. AA-KEA (Attention-Augmented KEA)
8. KEA-BERT (BERTScore-inspired semantic similarity)
"""

import csv
import ast
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc

# Import from Methods package
from Methods import (
    calculate_similarity,
    calculate_composite_similarity,
    calculate_aa_kea_similarity,
    calculate_pure_wl_kernel_similarity,
    GraphEmbeddingSimilarity,
    calculate_kea_bert_similarity
)


def parse_triple(triple_str):
    """Parse string representation of triple(s)"""
    try:
        parsed = ast.literal_eval(triple_str)

        # Handle both formats:
        # Format 1: [['A', 'B', 'C'], ['D', 'E', 'F']] -> return as is
        # Format 2: ['A', 'B', 'C'] -> wrap to [['A', 'B', 'C']]

        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                return parsed  # Already nested
            elif len(parsed) == 3:
                return [parsed]  # Wrap single triple
        return []
    except Exception as e:
        print(f"ERROR parsing: {e}")
        return []


def process_dataset(input_file, output_file):
    """Process dataset and calculate similarities using all methods.

    Writes results incrementally and cleans up after each row.
    """
    results = []
    fieldnames = ['question', 'gold_kg', 'llm_kg',
                  'kea_similarity',
                  'kea_composite', 'kea_structural', 'kea_semantic',
                  'transe_similarity', 'rotate_similarity', 'wl_kernel_similarity',
                  'aa_kea_similarity',
                  'kea_bert_similarity']

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)

    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{total_rows}...", end=" ", flush=True)

            question = row['question']
            answer1_str = row['gold_kg']
            answer2_str = row['llm_kg']

            triples1 = parse_triple(answer1_str)
            triples2 = parse_triple(answer2_str)

            result = {
                'question': question,
                'gold_kg': answer1_str,
                'llm_kg': answer2_str
            }

            if not triples1 or not triples2:
                print("skipped (empty triples)")
                result.update({
                    'kea_similarity': None,
                    'kea_composite': None,
                    'kea_structural': None,
                    'kea_semantic': None,
                    'transe_similarity': None,
                    'rotate_similarity': None,
                    'wl_kernel_similarity': None,
                    'aa_kea_similarity': None,
                    'kea_bert_similarity': None,
                })
            else:
                embedding_calculator = GraphEmbeddingSimilarity(embedding_dim=50)

                # 1. KEA method
                try:
                    kea_sim, _, _ = calculate_similarity(triples1, triples2)
                    result['kea_similarity'] = kea_sim
                except Exception:
                    result['kea_similarity'] = None

                # 2. KEA Composite
                try:
                    composite_result = calculate_composite_similarity(triples1, triples2, alpha=0.1, sigma=1.0)
                    result['kea_composite'] = composite_result['composite']
                    result['kea_structural'] = composite_result['structural']
                    result['kea_semantic'] = composite_result['semantic']
                except Exception:
                    result['kea_composite'] = None
                    result['kea_structural'] = None
                    result['kea_semantic'] = None

                # 3. TransE
                try:
                    result['transe_similarity'] = embedding_calculator.calculate_transe_similarity(triples1, triples2)
                except Exception:
                    result['transe_similarity'] = None

                # 4. RotatE
                try:
                    result['rotate_similarity'] = embedding_calculator.calculate_rotate_similarity(triples1, triples2)
                except Exception:
                    result['rotate_similarity'] = None

                # 5. Pure WL Kernel
                try:
                    result['wl_kernel_similarity'] = calculate_pure_wl_kernel_similarity(triples1, triples2)
                except Exception:
                    result['wl_kernel_similarity'] = None

                # 6. AA-KEA (Attention-Augmented KEA)
                try:
                    result['aa_kea_similarity'] = calculate_aa_kea_similarity(triples1, triples2)
                except Exception:
                    result['aa_kea_similarity'] = None

                # 7. KEA-BERT (BERTScore-inspired semantic similarity)
                try:
                    result['kea_bert_similarity'] = calculate_kea_bert_similarity(triples1, triples2)
                except Exception:
                    result['kea_bert_similarity'] = None

                del embedding_calculator

            writer.writerow(result)
            out_f.flush()
            results.append(result)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("done")

    print(f"\n{'='*60}")
    print(f"Results written to: {output_file}")
    print(f"Total rows processed: {len(results)}")
    print(f"{'='*60}")

    return results


def plot_individual_method_results(results, output_file_base):
    """Create individual plots for each similarity method."""
    plots_dir = os.path.join(os.path.dirname(output_file_base), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df['Question_ID'] = range(1, len(df) + 1)
    plot_df = df.dropna(subset=['aa_kea_similarity'])

    methods = {
        'KEA (Semantic Clustering + WL Kernel)': ('kea_similarity', '#2E86AB'),
        'KEA Composite (Gaussian + WL)': ('kea_composite', '#1E5F8C'),
        'KEA Semantic (Gaussian Only)': ('kea_semantic', '#5BA3D0'),
        'TransE (Translation Embedding)': ('transe_similarity', '#A23B72'),
        'RotatE (Rotation Embedding)': ('rotate_similarity', '#F18F01'),
        'Pure WL Kernel (Structural Only)': ('wl_kernel_similarity', '#6A994E'),
        'AA-KEA (Attention + WL)': ('aa_kea_similarity', '#9B59B6'),
        'KEA-BERT (BERTScore Semantic)': ('kea_bert_similarity', '#E67E22'),
    }

    print("\n" + "="*60)
    print("Generating individual plots for each method...")
    print(f"Saving plots to: {plots_dir}")
    print("="*60)

    for method_name, (col_name, color) in methods.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        scores = plot_df[col_name] * 100
        ax1.plot(plot_df['Question_ID'], scores,
                 marker='o', linestyle='-', linewidth=2,
                 markersize=6, color=color, alpha=0.8)
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect (100%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='No Match (0%)')
        ax1.set_title(f'{method_name}\nSimilarity Scores', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Similarity (%)', fontsize=10)
        ax1.set_xlabel('Question ID', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        ax1.set_ylim(max(-10, scores.min() - 10), min(110, scores.max() + 10))

        ax2.hist(scores, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=scores.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {scores.mean():.1f}%')
        ax2.axvline(x=scores.median(), color='blue', linestyle='--',
                    linewidth=2, label=f'Median: {scores.median():.1f}%')
        ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Similarity (%)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        base_name = os.path.basename(output_file_base)
        filename = os.path.join(plots_dir, f"{base_name}_{col_name}_plot.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(filename)}")
        plt.close()

    # Comprehensive comparison plot
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    for method_name, (col_name, color) in methods.items():
        scores = plot_df[col_name] * 100
        ax1.plot(plot_df['Question_ID'], scores,
                 marker='o', linestyle='-', linewidth=1.5,
                 markersize=4, color=color, label=method_name.split('(')[0].strip(),
                 alpha=0.7)
    ax1.set_title('All Methods Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Similarity (%)', fontsize=10)
    ax1.set_xlabel('Question ID', fontsize=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    method_labels = [name.split('(')[0].strip() for name in methods.keys()]
    averages = [plot_df[col] * 100 for _, (col, _) in methods.items()]
    colors_list = [color for _, (_, color) in methods.items()]
    bp = ax2.boxplot(averages, tick_labels=method_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_title('Score Distributions', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Similarity (%)', fontsize=10)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    avg_values = [plot_df[col].mean() * 100 for _, (col, _) in methods.items()]
    bars = ax3.bar(method_labels, avg_values, color=colors_list, alpha=0.7, edgecolor='black')
    for bar, avg in zip(bars, avg_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{avg:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title('Average Similarity by Method', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Similarity (%)', fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=15, ha='right')

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(plot_df['Question_ID'], plot_df['kea_similarity'] * 100,
             marker='o', linestyle='-', linewidth=2, markersize=5,
             color='#2E86AB', label='KEA (with semantics)', alpha=0.8)
    ax4.plot(plot_df['Question_ID'], plot_df['wl_kernel_similarity'] * 100,
             marker='d', linestyle='--', linewidth=2, markersize=5,
             color='#6A994E', label='Pure WL (structural)', alpha=0.8)
    ax4.set_title('KEA vs Pure WL: Semantic Impact', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Similarity (%)', fontsize=10)
    ax4.set_xlabel('Question ID', fontsize=10)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    base_name = os.path.basename(output_file_base)
    comparison_file = os.path.join(plots_dir, f"{base_name}_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {os.path.basename(comparison_file)}")
    plt.close()

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for method_name, (col_name, _) in methods.items():
        scores = plot_df[col_name] * 100
        perfect = (plot_df[col_name] >= 0.99).sum()
        high = (plot_df[col_name] >= 0.8).sum()
        print(f"\n{method_name}:")
        print(f"  Average:  {scores.mean():.2f}%")
        print(f"  Median:   {scores.median():.2f}%")
        print(f"  Std Dev:  {scores.std():.2f}%")
        print(f"  Max:      {scores.max():.2f}%")
        print(f"  Min:      {scores.min():.2f}%")
        print(f"  Perfect (>=1.0): {perfect}/{len(plot_df)} ({perfect/len(plot_df)*100:.1f}%)")
        print(f"  High (>=0.8):    {high}/{len(plot_df)} ({high/len(plot_df)*100:.1f}%)")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description='Multi-Method Graph Similarity Comparison')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing)')
    parser.add_argument('--input', type=str,
                        default="/Users/subu/Desktop/KGX-Graph-Similarity/data/validation_set.csv",
                        help='Input CSV file path')
    parser.add_argument('--output', type=str,
                        default="/Users/subu/Desktop/KGX-Graph-Similarity/results/validation_for_all_methods/multi_method_similarity_results.csv",
                        help='Output CSV file path')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("="*60)
    print("Multi-Method Graph Similarity Comparison")
    print("="*60)

    if args.limit:
        print(f"\n*** LIMITING TO {args.limit} ROWS (for testing) ***\n")

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if args.limit:
        all_rows = all_rows[:args.limit]
        print(f"Processing {len(all_rows)} rows (limited from original dataset)\n")

    input_file = args.input
    if args.limit:
        temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        temp_writer = csv.DictWriter(temp_input, fieldnames=['question', 'gold_kg', 'llm_kg'])
        temp_writer.writeheader()
        temp_writer.writerows(all_rows)
        temp_input.close()
        input_file = temp_input.name

    results = process_dataset(input_file, args.output)

    if args.limit:
        os.remove(input_file)

    if not args.skip_plots:
        plot_individual_method_results(results, args.output.replace('.csv', ''))
    else:
        print("\n[Skipping plot generation as requested]")
