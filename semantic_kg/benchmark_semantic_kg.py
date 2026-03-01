#!/usr/bin/env python3
"""
Benchmark similarity methods against ground truth scores
Evaluates which method best predicts the ground truth similarity
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score, roc_curve
import os


def load_results(results_file):
    """Load the results CSV file"""
    data = {
        'ground_truth': [],
        'kea_similarity': [],
        'kea_composite': [],
        'kea_structural': [],
        'kea_semantic': [],
        'transe_similarity': [],
        'rotate_similarity': [],
        'wl_kernel_similarity': [],
        'aa_kea_similarity': [],
        'sbert_direct_similarity': []
    }

    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ground truth
            try:
                data['ground_truth'].append(float(row['similarity_score_ground']))
            except (ValueError, KeyError):
                continue

            # All methods
            for method in ['kea_similarity', 'kea_composite', 'kea_structural',
                          'kea_semantic', 'transe_similarity', 'rotate_similarity',
                          'wl_kernel_similarity', 'aa_kea_similarity', 'sbert_direct_similarity']:
                try:
                    value = float(row[method]) if row[method] and row[method].lower() != 'none' else np.nan
                    data[method].append(value)
                except (ValueError, KeyError):
                    data[method].append(np.nan)

    return data


def calculate_metrics(predicted, ground_truth, threshold=0.7):
    """Calculate evaluation metrics including AUC with binary classification"""
    # Remove NaN values
    mask = ~np.isnan(predicted)
    pred_clean = np.array(predicted)[mask]
    gt_clean = np.array(ground_truth)[mask]

    if len(pred_clean) < 2:
        return None

    # Binary classification: ground truth > threshold = 1 (similar), else 0 (dissimilar)
    gt_binary = (gt_clean > threshold).astype(int)

    # Calculate AUC if we have both classes
    auc_score = None
    ap_score = None
    if len(np.unique(gt_binary)) > 1:  # Need both positive and negative samples
        try:
            auc_score = roc_auc_score(gt_binary, pred_clean)
            ap_score = average_precision_score(gt_binary, pred_clean)
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")

    metrics = {
        'mae': mean_absolute_error(gt_clean, pred_clean),
        'rmse': np.sqrt(mean_squared_error(gt_clean, pred_clean)),
        'r2': r2_score(gt_clean, pred_clean),
        'pearson': pearsonr(pred_clean, gt_clean)[0],
        'spearman': spearmanr(pred_clean, gt_clean)[0],
        'auc': auc_score,
        'average_precision': ap_score,
        'n_samples': len(pred_clean),
        'n_positive': np.sum(gt_binary),
        'n_negative': len(gt_binary) - np.sum(gt_binary)
    }

    return metrics


def benchmark_methods(results_file, output_dir):
    """Benchmark all methods and create visualizations"""

    print(f"\nLoading results from: {results_file}")

    # Load data
    data = load_results(results_file)
    ground_truth = data['ground_truth']

    if len(ground_truth) == 0:
        print("ERROR: No data loaded!")
        return

    print(f"Loaded {len(ground_truth)} samples\n")

    # Define methods with display names and colors
    methods = {
        'KEA (Clustering + WL)': ('kea_similarity', '#2E86AB'),
        'KEA Composite': ('kea_composite', '#1E5F8C'),
        'KEA Structural': ('kea_structural', '#A8DADC'),
        'KEA Semantic': ('kea_semantic', '#5BA3D0'),
        'TransE': ('transe_similarity', '#A23B72'),
        'RotatE': ('rotate_similarity', '#F18F01'),
        'WL Kernel': ('wl_kernel_similarity', '#6A994E'),
        'AA-KEA': ('aa_kea_similarity', '#9B59B6'),
        'SBERT Direct': ('sbert_direct_similarity', '#E63946')
    }

    # Calculate metrics for all methods
    results = {}
    print("\nCALCULATING METRICS (Threshold = 0.7 for AUC)")

    for display_name, (method_key, color) in methods.items():
        metrics = calculate_metrics(data[method_key], ground_truth, threshold=0.7)
        if metrics:
            results[display_name] = {
                'metrics': metrics,
                'predictions': data[method_key],
                'color': color
            }
            print(f"\n{display_name}:")
            print(f"  MAE:       {metrics['mae']:.4f}")
            print(f"  RMSE:      {metrics['rmse']:.4f}")
            print(f"  R²:        {metrics['r2']:.4f}")
            print(f"  Pearson:   {metrics['pearson']:.4f}")
            print(f"  Spearman:  {metrics['spearman']:.4f}")
            if metrics['auc'] is not None:
                print(f"  AUC:       {metrics['auc']:.4f}")
                print(f"  AP:        {metrics['average_precision']:.4f}")
            print(f"  Samples:   {metrics['n_samples']} (pos: {metrics['n_positive']}, neg: {metrics['n_negative']})")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating visualizations...")

    n_methods = len(results)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, (method_name, result) in enumerate(results.items()):
        ax = axes[idx]
        predictions = np.array(result['predictions'])
        gt = np.array(ground_truth)

        # Remove NaN
        mask = ~np.isnan(predictions)
        pred_clean = predictions[mask]
        gt_clean = gt[mask]

        # Scatter plot
        ax.scatter(gt_clean, pred_clean, alpha=0.5, color=result['color'], s=30)

        # Perfect prediction line
        min_val = min(gt_clean.min(), pred_clean.min())
        max_val = max(gt_clean.max(), pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Metrics in title
        metrics = result['metrics']
        ax.set_title(f"{method_name}\nMAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}, ρ={metrics['pearson']:.3f}",
                    fontweight='bold', fontsize=10)
        ax.set_xlabel('Ground Truth Similarity', fontsize=9)
        ax.set_ylabel('Predicted Similarity', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    scatter_file = os.path.join(output_dir, 'scatter_plots_comparison.png')
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {scatter_file}")
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    method_names = list(results.keys())
    colors_list = [results[m]['color'] for m in method_names]

    # MAE (lower is better)
    mae_values = [results[m]['metrics']['mae'] for m in method_names]
    axes[0, 0].bar(range(len(method_names)), mae_values, color=colors_list, alpha=0.7)
    axes[0, 0].set_title('Mean Absolute Error (MAE)\nLower is Better', fontweight='bold')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_xticks(range(len(method_names)))
    axes[0, 0].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # RMSE (lower is better)
    rmse_values = [results[m]['metrics']['rmse'] for m in method_names]
    axes[0, 1].bar(range(len(method_names)), rmse_values, color=colors_list, alpha=0.7)
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)\nLower is Better', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_xticks(range(len(method_names)))
    axes[0, 1].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # R² (higher is better)
    r2_values = [results[m]['metrics']['r2'] for m in method_names]
    axes[0, 2].bar(range(len(method_names)), r2_values, color=colors_list, alpha=0.7)
    axes[0, 2].set_title('R² Score\nHigher is Better', fontweight='bold')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].set_xticks(range(len(method_names)))
    axes[0, 2].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    for i, v in enumerate(r2_values):
        axes[0, 2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Pearson correlation (higher is better)
    pearson_values = [results[m]['metrics']['pearson'] for m in method_names]
    axes[1, 0].bar(range(len(method_names)), pearson_values, color=colors_list, alpha=0.7)
    axes[1, 0].set_title('Pearson Correlation\nHigher is Better', fontweight='bold')
    axes[1, 0].set_ylabel('Pearson ρ')
    axes[1, 0].set_xticks(range(len(method_names)))
    axes[1, 0].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(pearson_values):
        axes[1, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Spearman correlation (higher is better)
    spearman_values = [results[m]['metrics']['spearman'] for m in method_names]
    axes[1, 1].bar(range(len(method_names)), spearman_values, color=colors_list, alpha=0.7)
    axes[1, 1].set_title('Spearman Correlation\nHigher is Better', fontweight='bold')
    axes[1, 1].set_ylabel('Spearman ρ')
    axes[1, 1].set_xticks(range(len(method_names)))
    axes[1, 1].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim(0, 1)
    for i, v in enumerate(spearman_values):
        axes[1, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # AUC (higher is better)
    auc_values = [results[m]['metrics']['auc'] if results[m]['metrics']['auc'] is not None else 0
                  for m in method_names]
    axes[1, 2].bar(range(len(method_names)), auc_values, color=colors_list, alpha=0.7)
    axes[1, 2].set_title('AUC (threshold=0.7)\nHigher is Better', fontweight='bold')
    axes[1, 2].set_ylabel('AUC Score')
    axes[1, 2].set_xticks(range(len(method_names)))
    axes[1, 2].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim(0, 1)
    for i, v in enumerate(auc_values):
        axes[1, 2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Overall ranking (composite score)
    # Normalize metrics to 0-1 and calculate composite score (including AUC)
    mae_norm = 1 - (np.array(mae_values) - min(mae_values)) / (max(mae_values) - min(mae_values) + 1e-10)
    rmse_norm = 1 - (np.array(rmse_values) - min(rmse_values)) / (max(rmse_values) - min(rmse_values) + 1e-10)
    r2_norm = (np.array(r2_values) - min(r2_values)) / (max(r2_values) - min(r2_values) + 1e-10)
    pearson_norm = np.array(pearson_values)
    spearman_norm = np.array(spearman_values)
    auc_norm = np.array(auc_values)  # AUC is already 0-1

    # Composite score: average of all normalized metrics
    composite_scores = (mae_norm + rmse_norm + r2_norm + pearson_norm + spearman_norm + auc_norm) / 6

    axes[1, 3].bar(range(len(method_names)), composite_scores, color=colors_list, alpha=0.7)
    axes[1, 3].set_title('Composite Score\nHigher is Better', fontweight='bold')
    axes[1, 3].set_ylabel('Composite Score')
    axes[1, 3].set_xticks(range(len(method_names)))
    axes[1, 3].set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    axes[1, 3].grid(True, alpha=0.3, axis='y')
    axes[1, 3].set_ylim(0, 1)
    for i, v in enumerate(composite_scores):
        axes[1, 3].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    metrics_file = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {metrics_file}")
    plt.close()

    n_methods = len(results)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, (method_name, result) in enumerate(results.items()):
        predictions = np.array(result['predictions'])
        gt = np.array(ground_truth)

        mask = ~np.isnan(predictions)
        errors = predictions[mask] - gt[mask]

        axes[idx].hist(errors, bins=30, color=result['color'], alpha=0.7, edgecolor='black')
        axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[idx].axvline(x=errors.mean(), color='blue', linestyle='--', linewidth=2,
                         label=f'Mean: {errors.mean():.3f}')
        axes[idx].set_title(f'{method_name}\nError Distribution', fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Prediction Error', fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    error_file = os.path.join(output_dir, 'error_distributions.png')
    plt.savefig(error_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {error_file}")
    plt.close()

    print("\nGenerating ROC curves...")

    # Individual ROC curves
    n_methods = len(results)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    threshold = 0.7

    for idx, (method_name, result) in enumerate(results.items()):
        ax = axes[idx]
        predictions = np.array(result['predictions'])
        gt = np.array(ground_truth)

        # Remove NaN
        mask = ~np.isnan(predictions)
        pred_clean = predictions[mask]
        gt_clean = gt[mask]

        # Binary labels
        gt_binary = (gt_clean > threshold).astype(int)

        if len(np.unique(gt_binary)) > 1:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(gt_binary, pred_clean)
            auc_score = result['metrics']['auc']

            # Plot ROC curve
            ax.plot(fpr, tpr, color=result['color'], linewidth=2.5,
                   label=f'AUC = {auc_score:.3f}')

            # Plot diagonal (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5,
                   label='Random (AUC = 0.5)')

            # Mark optimal point (closest to top-left)
            optimal_idx = np.argmax(tpr - fpr)
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                   label=f'Optimal (thresh={thresholds[optimal_idx]:.3f})')

            ax.set_title(f'{method_name}\nROC Curve', fontweight='bold', fontsize=11)
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor ROC curve',
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{method_name}\nROC Curve', fontweight='bold', fontsize=11)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    roc_individual_file = os.path.join(output_dir, 'roc_curves_individual.png')
    plt.savefig(roc_individual_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {roc_individual_file}")
    plt.close()

    # Combined ROC curve comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for method_name, result in results.items():
        predictions = np.array(result['predictions'])
        gt = np.array(ground_truth)

        mask = ~np.isnan(predictions)
        pred_clean = predictions[mask]
        gt_clean = gt[mask]

        gt_binary = (gt_clean > threshold).astype(int)

        if len(np.unique(gt_binary)) > 1:
            fpr, tpr, _ = roc_curve(gt_binary, pred_clean)
            auc_score = result['metrics']['auc']

            ax.plot(fpr, tpr, color=result['color'], linewidth=2.5,
                   label=f'{method_name} (AUC = {auc_score:.3f})', alpha=0.8)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC = 0.5)')

    ax.set_title('ROC Curve Comparison - All Methods\n(Threshold = 0.7 for binary classification)',
                fontweight='bold', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    roc_comparison_file = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(roc_comparison_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {roc_comparison_file}")
    plt.close()


    summary_file = os.path.join(output_dir, 'benchmark_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK SUMMARY - SEMANTIC KG SIMILARITY METHODS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total samples: {len(ground_truth)}\n\n")

        # Rank by composite score
        ranked = sorted(zip(method_names, composite_scores), key=lambda x: x[1], reverse=True)

        f.write("OVERALL RANKING (by Composite Score):\n")
        f.write("-" * 80 + "\n")
        for rank, (method, score) in enumerate(ranked, 1):
            f.write(f"{rank}. {method}: {score:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("="*80 + "\n\n")

        for method_name, result in results.items():
            m = result['metrics']
            f.write(f"{method_name}:\n")
            f.write(f"  MAE:          {m['mae']:.4f} (lower is better)\n")
            f.write(f"  RMSE:         {m['rmse']:.4f} (lower is better)\n")
            f.write(f"  R²:           {m['r2']:.4f} (higher is better)\n")
            f.write(f"  Pearson:      {m['pearson']:.4f} (higher is better)\n")
            f.write(f"  Spearman:     {m['spearman']:.4f} (higher is better)\n")
            if m['auc'] is not None:
                f.write(f"  AUC:          {m['auc']:.4f} (higher is better)\n")
                f.write(f"  Avg Precision: {m['average_precision']:.4f} (higher is better)\n")
            f.write(f"  Valid samples: {m['n_samples']} (positive: {m['n_positive']}, negative: {m['n_negative']})\n\n")

        f.write("="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*80 + "\n")
        f.write("- MAE/RMSE: Average prediction error (0 = perfect)\n")
        f.write("- R²: Proportion of variance explained (1 = perfect, 0 = baseline)\n")
        f.write("- Pearson: Linear correlation (-1 to 1, 1 = perfect positive)\n")
        f.write("- Spearman: Rank correlation (-1 to 1, 1 = perfect monotonic)\n")
        f.write("- AUC: Area Under ROC Curve for binary classification (threshold=0.7)\n")
        f.write("       (1 = perfect, 0.5 = random, <0.5 = worse than random)\n")
        f.write("- Avg Precision: Precision-Recall curve summary (1 = perfect)\n")
        f.write("- Composite Score: Normalized average of all metrics (1 = best)\n")

    print(f"Saved: {summary_file}")

    print(f"\nResults saved to: {output_dir}")
    print("\nBest method (by composite score):")
    best_method, best_score = ranked[0]
    print(f"  → {best_method}: {best_score:.4f}")
    print("\nCheck benchmark_summary.txt for full details.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark similarity methods against ground truth')
    parser.add_argument('--results', type=str,
                       default='/Users/subu/Desktop/FYP/KGX-Graph-Similarity/results/semantic_kg/semantic_kg_similarity_results.csv',
                       help='Results CSV file from semantic_kg_similarity.py')
    parser.add_argument('--output', type=str,
                       default='/Users/subu/Desktop/FYP/KGX-Graph-Similarity/results/semantic_kg/benchmark',
                       help='Output directory for benchmark results')
    args = parser.parse_args()

    benchmark_methods(args.results, args.output)
