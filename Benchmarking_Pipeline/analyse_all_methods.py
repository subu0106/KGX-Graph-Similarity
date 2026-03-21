#!/usr/bin/env python3
"""
Step 2 — Evaluate all methods across all labeled datasets.

Follows the exact evaluation approach of:
  src/evaluation/semantic_kg_evaluation.py  → find_optimal_threshold, evaluate_method,
                                              evaluate_by_perturbation_type, ROC curves
  src/evaluation/update_cross_dataset.py    → _draw_dual_heatmap (RdBu, vmin=0.5/vmax=1.0,
                                              dual F1+AUC panels, navy border), ranking bar

Input : Data_labeled/*_labeled.csv
Output: Results_Analysis/
  per_dataset/<dataset>/
    roc_curves.png
    perturbation_analysis.png
    overall_results.csv
  cross_dataset/
    all_methods_heatmap.png          (F1 + AUC dual panel, RdBu, like update_cross_dataset.py)
    variants_comparison_heatmap.png  (our variants only)
    all_methods_ranks.png            (per-dataset rank heatmap, like plot_ranks)
    ranking_table.png
    ranking_bar.png                  (mean-rank bar, like all_methods_ranks.png)
    performance_summary_table.png    (our best vs best other, like plot_performance_table)
    performance_summary.csv
    cross_dataset_summary.csv

Usage:
    python analyse_all_methods.py
    python analyse_all_methods.py --dataset mrpc_400_KGs_labeled
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, accuracy_score,
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).parent
DATA_DIR = HERE / 'Data_labeled'
OUT_DIR  = HERE / 'Results_Analysis'
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
METHODS = [
    'kea_similarity',
    'transe_similarity',
    'rotate_similarity',
    'wl_kernel_similarity',
    'aa_kea_similarity',
    'snea_bert_alpha_0.0',
    'snea_bert_alpha_0.1',
    'snea_bert_alpha_0.2',
    'snea_bert_alpha_0.3',
    'snea_bert_alpha_0.4',
    'snea_bert_alpha_0.5',
    'snea_bert_alpha_0.6',
    'snea_bert_alpha_0.7',
    'snea_bert_alpha_0.8',
    'snea_bert_alpha_0.9',
    'snea_bert_alpha_1.0_SNEA_alone',
    'kea_bert_similarity',
    'semantic_wl_similarity',
]

# Full display names (for tables and CSV)
METHOD_LABELS = {
    'kea_similarity':                 'KEA',
    'transe_similarity':              'TransE',
    'rotate_similarity':              'RotatE',
    'wl_kernel_similarity':           'WL Kernel',
    'aa_kea_similarity':              'AA-KEA',
    'snea_bert_alpha_0.0':            'SNEA-SBERT α=0.0',
    'snea_bert_alpha_0.1':            'SNEA-SBERT α=0.1',
    'snea_bert_alpha_0.2':            'SNEA-SBERT α=0.2',
    'snea_bert_alpha_0.3':            'SNEA-SBERT α=0.3',
    'snea_bert_alpha_0.4':            'SNEA-SBERT α=0.4',
    'snea_bert_alpha_0.5':            'SNEA-SBERT α=0.5',
    'snea_bert_alpha_0.6':            'SNEA-SBERT α=0.6',
    'snea_bert_alpha_0.7':            'SNEA-SBERT α=0.7',
    'snea_bert_alpha_0.8':            'SNEA-SBERT α=0.8',
    'snea_bert_alpha_0.9':            'SNEA-SBERT α=0.9',
    'snea_bert_alpha_1.0_SNEA_alone': 'SNEA (WL only)',
    'kea_bert_similarity':            'KEA-BERT',
    'semantic_wl_similarity':         'Semantic WL',
}

# Short names for heatmap x-axis (matches METHOD_SHORT style in update_cross_dataset.py)
METHOD_SHORT = {
    'kea_similarity':                 'KEA',
    'transe_similarity':              'TransE',
    'rotate_similarity':              'RotatE',
    'wl_kernel_similarity':           'WL',
    'aa_kea_similarity':              'AA-KEA\n(Ours)',
    'snea_bert_alpha_0.0':            'α=0.0',
    'snea_bert_alpha_0.1':            'α=0.1',
    'snea_bert_alpha_0.2':            'α=0.2',
    'snea_bert_alpha_0.3':            'α=0.3',
    'snea_bert_alpha_0.4':            'α=0.4',
    'snea_bert_alpha_0.5':            'SNEA-SBERT\nα=0.5\n(Ours)',
    'snea_bert_alpha_0.6':            'α=0.6',
    'snea_bert_alpha_0.7':            'α=0.7',
    'snea_bert_alpha_0.8':            'α=0.8',
    'snea_bert_alpha_0.9':            'α=0.9',
    'snea_bert_alpha_1.0_SNEA_alone': 'SNEA\n(Ours)',
    'kea_bert_similarity':            'KEA-BERT',
    'semantic_wl_similarity':         'Sem-WL',
}

# Our KG-based variants (highlighted with navy border in heatmap)
OUR_VARIANTS = [
    'aa_kea_similarity',
    'snea_bert_alpha_0.0', 'snea_bert_alpha_0.1', 'snea_bert_alpha_0.2',
    'snea_bert_alpha_0.3', 'snea_bert_alpha_0.4', 'snea_bert_alpha_0.5',
    'snea_bert_alpha_0.6', 'snea_bert_alpha_0.7', 'snea_bert_alpha_0.8',
    'snea_bert_alpha_0.9', 'snea_bert_alpha_1.0_SNEA_alone',
]

# Baselines (non-KG methods included in results)
BASELINE_METHODS = [
    'kea_similarity',
    'transe_similarity',
    'rotate_similarity',
    'wl_kernel_similarity',
    'kea_bert_similarity',
    'semantic_wl_similarity',
]

DATASET_SHORT = {
    'mrpc_400_KGs_labeled':                         'MRPC',
    'paws_wiki_400_KGs_labeled':                    'PAWS-Wiki',
    'semantic_kg_codex_400_KGs_labeled':            'SKG-Codex',
    'semantic_kg_combined_400_KGs_labeled':         'SKG-Combined',
    'semantic_kg_findkg_400_KGs_labeled':           'SKG-FindKG',
    'semantic_kg_globi_400_KGs_labeled':            'SKG-GloBI',
    'semantic_kg_oregano_400_KGs_labeled':          'SKG-Oregano',
    'sts12_400_KGs_labeled':                        'STS12',
    'wikipedia_entity_swap_400_KGs_labeled':        'Wiki-Swap',
    'pubmedqa_ranked_faithfulness_400_KGs_labeled': 'PubMedQA',
}


# ===========================================================================
# EVALUATION — identical to semantic_kg_evaluation.py
# ===========================================================================

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold that maximizes F1 score.
    Identical to semantic_kg_evaluation.py::find_optimal_threshold."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_method(df, score_column, method_name):
    """Evaluate a single method.
    Identical to semantic_kg_evaluation.py::evaluate_method,
    with NaN-safe handling for missing scores."""
    valid    = df[score_column].notna()
    y_true   = df.loc[valid, 'label'].values.astype(int)
    y_scores = df.loc[valid, score_column].values.astype(float)

    if len(y_true) < 10:
        return None, None

    threshold, _ = find_optimal_threshold(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = float('nan')

    metrics = {
        'method':     method_name,
        'col':        score_column,
        'threshold':  threshold,
        'accuracy':   accuracy_score(y_true, y_pred),
        'precision':  precision_score(y_true, y_pred, zero_division=0),
        'recall':     recall_score(y_true, y_pred, zero_division=0),
        'f1':         f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':    roc_auc,
        'n':          int(len(y_true)),
    }
    return metrics, y_pred


def evaluate_by_perturbation_type(df, active_methods):
    """Evaluate methods stratified by perturbation type.
    Follows semantic_kg_evaluation.py::evaluate_by_perturbation_type."""
    if 'perturbation_type' not in df.columns:
        return pd.DataFrame()

    perturbation_types = df['perturbation_type'].dropna().unique()
    perturbation_results = []

    for pert_type in perturbation_types:
        subset = df[df['perturbation_type'] == pert_type].copy()
        if len(subset) < 5:
            continue

        for m in active_methods:
            if m not in df.columns:
                continue
            valid    = subset[m].notna()
            y_true   = subset.loc[valid, 'label'].values.astype(int)
            y_scores = subset.loc[valid, m].values.astype(float)

            if len(y_true) < 5:
                continue

            threshold, _ = find_optimal_threshold(y_true, y_scores)
            y_pred = (y_scores >= threshold).astype(int)
            f1     = f1_score(y_true, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = float('nan')

            perturbation_results.append({
                'perturbation_type': pert_type,
                'method':            METHOD_LABELS.get(m, m),
                'f1':                f1,
                'roc_auc':           auc,
                'n_samples':         int(valid.sum()),
            })

    return pd.DataFrame(perturbation_results)


# ===========================================================================
# PLOTS — heatmap follows update_cross_dataset.py::_draw_dual_heatmap exactly
# ===========================================================================

def _draw_dual_heatmap(pivot_f1, pivot_auc, col_labels, suptitle, out_path, n_ours=0):
    """Dual F1/AUC heatmap.
    Identical style to update_cross_dataset.py::_draw_dual_heatmap:
      - RdBu colormap, vmin=0.5, vmax=1.0
      - annot_kws size=11 bold
      - navy border box around our-variant columns
      - dpi=350, bbox_inches=tight
    """
    ncols = len(pivot_f1.columns)
    nrows = len(pivot_f1.index)
    fig, axes = plt.subplots(1, 2, figsize=(max(ncols * 2.0 + 4, 22),
                                             max(nrows * 1.3 + 3, 7)))

    for ax, pivot, metric in zip(axes, [pivot_f1, pivot_auc], ['F1 Score', 'ROC-AUC']):
        mask = pivot.isna()
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt='.3f',
            annot_kws={'size': 11, 'weight': 'bold'},
            cmap='RdBu',
            vmin=0.5, vmax=1.0,
            mask=mask,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': metric, 'shrink': 0.75},
            xticklabels=col_labels,
        )
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10, rotation=0)

        if n_ours > 0:
            ax.add_patch(plt.Rectangle((0, 0), n_ours, nrows,
                                        fill=False, edgecolor='navy',
                                        linewidth=2.5, clip_on=False))

    ours_p = mpatches.Patch(color='#4878d0', label='Our KG Variants')
    base_p = mpatches.Patch(color='#aec7e8', label='Baselines')
    fig.legend(handles=[ours_p, base_p], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')


# ===========================================================================
# PER-DATASET ANALYSIS
# ===========================================================================

def analyse_dataset(df, dataset_name):
    out = OUT_DIR / 'per_dataset' / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    active = [m for m in METHODS if m in df.columns]

    # ── Evaluate all methods ─────────────────────────────────────────────────
    records = []
    for m in active:
        metrics, _ = evaluate_method(df, m, METHOD_LABELS.get(m, m))
        if metrics:
            records.append(metrics)

    if not records:
        print(f'  No valid methods for {dataset_name}')
        return {}

    results_df = pd.DataFrame(records).sort_values('f1', ascending=False)
    results_df.to_csv(out / 'overall_results.csv', index=False)

    # Terminal print
    ds_label = DATASET_SHORT.get(dataset_name, dataset_name)
    print(f'\n  {ds_label}')
    print(f'  {"Method":<28}  {"F1":>6}  {"AUC":>6}  {"Prec":>6}  {"Rec":>6}  {"Thr":>5}  N')
    print('  ' + '-' * 68)
    for _, row in results_df.iterrows():
        print(f'  {row["method"]:<28}  {row["f1"]:>6.4f}  {row["roc_auc"]:>6.4f}  '
              f'{row["precision"]:>6.4f}  {row["recall"]:>6.4f}  '
              f'{row["threshold"]:>5.2f}  {row["n"]}')

    # ── ROC curves (same style as semantic_kg_evaluation.py) ─────────────────
    _fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap('tab20', len(active))
    for i, m in enumerate(active):
        valid    = df[m].notna()
        y_true   = df.loc[valid, 'label'].values.astype(int)
        y_scores = df.loc[valid, m].values.astype(float)
        if len(y_true) < 10:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            ax.plot(fpr, tpr, color=cmap(i), lw=1.5,
                    label=f'{METHOD_SHORT.get(m, m)} (AUC={auc:.3f})')
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {ds_label}', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(out / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Perturbation-type analysis (same as semantic_kg_evaluation.py) ───────
    pert_df = evaluate_by_perturbation_type(df, active)
    if not pert_df.empty:
        pert_df.to_csv(out / 'perturbation_results.csv', index=False)
        top5 = results_df.head(5)['method'].tolist()
        plot_df = pert_df[pert_df['method'].isin(top5)]
        if not plot_df.empty:
            pivot = plot_df.pivot_table(index='perturbation_type', columns='method',
                                        values='f1', aggfunc='mean')
            _, ax = plt.subplots(figsize=(max(8, len(top5) * 1.5),
                                          max(4, len(pivot) * 0.6 + 1)))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                        vmin=0, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title(f'F1 by Perturbation Type — {ds_label}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(out / 'perturbation_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f'  → {out}')
    return {row['col']: {'f1': row['f1'], 'roc_auc': row['roc_auc'],
                          'precision': row['precision'], 'recall': row['recall']}
            for _, row in results_df.iterrows()}


# ===========================================================================
# CROSS-DATASET — follows update_cross_dataset.py
# ===========================================================================

def cross_dataset_analysis(all_results):
    cross_dir = OUT_DIR / 'cross_dataset'
    cross_dir.mkdir(exist_ok=True)

    datasets      = list(all_results.keys())
    active_methods = [m for m in METHODS if any(m in all_results[d] for d in datasets)]

    ds_labels = [DATASET_SHORT.get(d, d) for d in datasets]
    m_labels  = [METHOD_SHORT.get(m, m)  for m in active_methods]

    # Build F1 and AUC matrices
    f1_mat  = np.full((len(datasets), len(active_methods)), np.nan)
    auc_mat = np.full((len(datasets), len(active_methods)), np.nan)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(active_methods):
            if m in all_results[ds]:
                f1_mat[i, j]  = all_results[ds][m]['f1']
                auc_mat[i, j] = all_results[ds][m]['roc_auc']

    pivot_f1  = pd.DataFrame(f1_mat,  index=ds_labels, columns=m_labels)
    pivot_auc = pd.DataFrame(auc_mat, index=ds_labels, columns=m_labels)

    # Count how many columns are "our variants" (put them first)
    n_ours = sum(1 for m in active_methods if m in OUR_VARIANTS)

    # ── Dual heatmap — identical to update_cross_dataset.py::_draw_dual_heatmap
    _draw_dual_heatmap(
        pivot_f1, pivot_auc, m_labels,
        suptitle='All Methods × All Datasets  (Our KG Variants | Baselines)',
        out_path=cross_dir / 'all_methods_heatmap.png',
        n_ours=n_ours,
    )

    # ── Ranking table + bar — follows update_cross_dataset.py::plot_ranks ────
    mean_f1  = np.nanmean(f1_mat,  axis=0)
    mean_auc = np.nanmean(auc_mat, axis=0)
    avg_score = (mean_f1 + mean_auc) / 2
    rank_df = pd.DataFrame({
        'method_col': active_methods,
        'Method':     [METHOD_LABELS.get(m, m) for m in active_methods],
        'Mean F1':    mean_f1,
        'Mean AUC':   mean_auc,
        'Avg Score':  avg_score,
        'Rank':       pd.Series(avg_score).rank(ascending=False, method='min').values,
    })
    rank_df = rank_df.sort_values('Rank').reset_index(drop=True)
    rank_df.to_csv(cross_dir / 'cross_dataset_summary.csv', index=False)

    # Ranking bar chart (like all_methods_ranks.png in update_cross_dataset.py)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(active_methods) * 0.4 + 2)))
    colors = ['#4878d0' if m in OUR_VARIANTS else '#aec7e8'
              for m in rank_df['method_col']]
    for ax, col in zip(axes, ['Mean F1', 'Mean AUC']):
        vals = rank_df[col].values
        bars = ax.barh(range(len(rank_df)), vals, color=colors, edgecolor='black', lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=8)
        ax.set_yticks(range(len(rank_df)))
        ax.set_yticklabels(rank_df['Method'], fontsize=8)
        ax.set_xlabel(col, fontsize=10)
        ax.set_title(f'{col} — Cross-Dataset Mean', fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    ours_p = mpatches.Patch(color='#4878d0', label='Our KG Variants')
    base_p = mpatches.Patch(color='#aec7e8', label='Baselines')
    fig.legend(handles=[ours_p, base_p], loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Method Ranking by Mean F1 and AUC (sorted by Avg Score = (F1+AUC)/2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(cross_dir / 'ranking_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: ranking_bar.png')

    # Ranking table image
    rows = [[f'{int(row["Rank"])}', row['Method'],
             f'{row["Mean F1"]:.4f}', f'{row["Mean AUC"]:.4f}',
             f'{row["Avg Score"]:.4f}']
            for _, row in rank_df.iterrows()]
    cols = ['Rank', 'Method', 'Mean F1', 'Mean AUC', 'Avg Score']
    h = 0.35 * (len(rows) + 1) + 0.6
    fig, ax = plt.subplots(figsize=(10, h))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for c in range(len(cols)):
        tbl[0, c].set_facecolor('#2E5594')
        tbl[0, c].set_text_props(color='white', fontweight='bold')
    for r in range(1, len(rows) + 1):
        bg = '#EAF2FB' if r % 2 == 0 else 'white'
        for c in range(len(cols)):
            tbl[r, c].set_facecolor(bg)
            tbl[r, c].set_text_props(color='black')
    ax.set_title('Method Ranking — Cross-Dataset  (rank by Avg Score = (Mean F1 + Mean AUC) / 2)',
                 fontsize=11, fontweight='bold')
    plt.savefig(cross_dir / 'ranking_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: ranking_table.png')

    # Terminal print
    print(f'\n{"="*65}')
    print('CROSS-DATASET RANKING  (sorted by Avg Score = (Mean F1 + Mean AUC) / 2)')
    print(f'{"="*65}')
    print(f'{"Rank":>4}  {"Method":<28}  {"Mean F1":>7}  {"Mean AUC":>8}  {"Avg Score":>9}')
    print('-' * 62)
    for _, row in rank_df.iterrows():
        print(f'{int(row["Rank"]):>4}  {row["Method"]:<28}  {row["Mean F1"]:>7.4f}  '
              f'{row["Mean AUC"]:>8.4f}  {row["Avg Score"]:>9.4f}')

    # ── Variants-only heatmap (update_cross_dataset.py::plot_variants_heatmap) ─
    our_idx   = [j for j, m in enumerate(active_methods) if m in OUR_VARIANTS]
    if our_idx:
        our_cols     = [active_methods[j] for j in our_idx]
        our_labels   = [METHOD_SHORT.get(m, m) for m in our_cols]
        pf1_ours  = pivot_f1.iloc[:, our_idx]
        pf1_ours.columns  = our_labels
        pauc_ours = pivot_auc.iloc[:, our_idx]
        pauc_ours.columns = our_labels
        _draw_dual_heatmap(
            pf1_ours, pauc_ours, our_labels,
            suptitle='Our KG Method Variants — Internal Comparison',
            out_path=cross_dir / 'variants_comparison_heatmap.png',
            n_ours=len(our_cols),
        )

    # ── Per-dataset rank heatmap (update_cross_dataset.py::plot_ranks) ─────────
    f1_rank_mat  = np.full_like(f1_mat, np.nan)
    auc_rank_mat = np.full_like(auc_mat, np.nan)
    for i in range(len(datasets)):
        row_f1  = f1_mat[i]
        row_auc = auc_mat[i]
        valid_f1  = ~np.isnan(row_f1)
        valid_auc = ~np.isnan(row_auc)
        if valid_f1.any():
            ranks = pd.Series(row_f1[valid_f1]).rank(ascending=False, method='min').values
            f1_rank_mat[i, valid_f1] = ranks
        if valid_auc.any():
            ranks = pd.Series(row_auc[valid_auc]).rank(ascending=False, method='min').values
            auc_rank_mat[i, valid_auc] = ranks

    pivot_f1_rank  = pd.DataFrame(f1_rank_mat,  index=ds_labels, columns=m_labels)
    pivot_auc_rank = pd.DataFrame(auc_rank_mat, index=ds_labels, columns=m_labels)

    max_rank = int(np.nanmax([f1_rank_mat, auc_rank_mat]))
    ncols    = len(active_methods)
    nrows    = len(datasets)

    fig_r, axes_r = plt.subplots(1, 2, figsize=(max(ncols * 2.0 + 4, 22),
                                                  max(nrows * 1.3 + 3, 7)))
    for ax, pivot, metric in zip(axes_r, [pivot_f1_rank, pivot_auc_rank], ['F1', 'AUC']):
        mask = pivot.isna()
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt='.0f',
            annot_kws={'size': 11, 'weight': 'bold'},
            cmap='RdBu',
            vmin=1, vmax=max_rank,
            mask=mask,
            linewidths=0.4, linecolor='#cccccc',
            cbar_kws={'label': f'Rank (1 = best)  [{metric}]', 'shrink': 0.75},
            xticklabels=m_labels,
        )
        ax.set_title(f'{metric} Rank per Dataset  (1 = best)',
                     fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=10, rotation=0)
        # Grey fill for NaN cells
        for i in range(nrows):
            for j in range(ncols):
                if mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                               fill=True, color='#d9d9d9',
                                               lw=0, zorder=0))
        # Navy border around our-variant columns
        if n_ours > 0:
            ax.add_patch(plt.Rectangle((0, 0), n_ours, nrows,
                                        fill=False, edgecolor='navy',
                                        linewidth=2.5, clip_on=False))

    ours_p  = mpatches.Patch(color='#4878d0', label='Our KG Variants')
    base_p  = mpatches.Patch(color='#aec7e8', label='Baselines')
    grey_p  = mpatches.Patch(color='#d9d9d9', label='Not evaluated')
    fig_r.legend(handles=[ours_p, base_p, grey_p], loc='lower center',
                 ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    plt.suptitle('Per-Dataset Method Rankings  (F1 | AUC)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(cross_dir / 'all_methods_ranks.png', dpi=350, bbox_inches='tight')
    plt.close()
    print('  Saved: all_methods_ranks.png')

    # ── Performance summary table (update_cross_dataset.py::plot_performance_table) ─
    # Build flat dataframe: dataset × method → f1, roc_auc, precision, recall, is_ours
    flat_rows = []
    for i, ds in enumerate(datasets):
        for j, m in enumerate(active_methods):
            if not np.isnan(f1_mat[i, j]):
                flat_rows.append({
                    'Dataset':    ds_labels[i],
                    'method_col': m,
                    'Method':     METHOD_LABELS.get(m, m),
                    'is_ours':    m in OUR_VARIANTS,
                    'F1':         f1_mat[i, j],
                    'AUC':        auc_mat[i, j],
                    'Precision':  all_results[ds].get(m, {}).get('precision', np.nan),
                    'Recall':     all_results[ds].get(m, {}).get('recall', np.nan),
                })
    flat_df = pd.DataFrame(flat_rows)

    VERDICT_COLORS = {
        '✔ Strong':       '#d4edda',
        '~ Comparable':   '#fff3cd',
        '▼ Slight gap':   '#fde8d8',
        '✘ Underperforms':'#f8d7da',
    }
    HIGHLIGHT_VERDICTS = {'✔ Strong', '~ Comparable'}

    perf_rows = []
    for ds_lbl in ds_labels:
        sub       = flat_df[flat_df['Dataset'] == ds_lbl]
        our_sub   = sub[sub['is_ours']]
        other_sub = sub[~sub['is_ours']]
        if our_sub.empty or other_sub.empty:
            continue
        best_our   = our_sub.loc[our_sub['F1'].idxmax()]
        best_other = other_sub.loc[other_sub['F1'].idxmax()]
        delta_f1   = best_our['F1']  - best_other['F1']
        delta_auc  = best_our['AUC'] - best_other['AUC']
        if delta_f1 >= 0.03:
            verdict = '✔ Strong'
        elif delta_f1 >= 0:
            verdict = '~ Comparable'
        elif delta_f1 >= -0.05:
            verdict = '▼ Slight gap'
        else:
            verdict = '✘ Underperforms'
        perf_rows.append({
            'Dataset':           ds_lbl,
            'Our Best Variant':  best_our['Method'],
            'Our F1':            round(best_our['F1'],  3),
            'Our AUC':           round(best_our['AUC'], 3),
            'Best Other Method': best_other['Method'],
            'Other F1':          round(best_other['F1'],  3),
            'Other AUC':         round(best_other['AUC'], 3),
            'ΔF1':               round(delta_f1,  3),
            'ΔAUC':              round(delta_auc, 3),
            'Verdict':           verdict,
        })

    perf_df = pd.DataFrame(perf_rows)
    if not perf_df.empty:
        perf_df.to_csv(cross_dir / 'performance_summary.csv', index=False)

        n     = len(perf_df)
        col_keys   = list(perf_df.columns)
        col_widths = [0.10, 0.14, 0.07, 0.07, 0.14, 0.07, 0.07, 0.06, 0.06, 0.12]
        fig_t, ax_t = plt.subplots(figsize=(22, n * 0.65 + 2.5))
        ax_t.axis('off')
        table = ax_t.table(
            cellText=perf_df.values.tolist(),
            colLabels=col_keys,
            cellLoc='center', loc='center',
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 2.0)
        for j in range(len(col_keys)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i, row in enumerate(perf_df.itertuples(index=False)):
            bg        = VERDICT_COLORS.get(row.Verdict, '#f9f9f9')
            highlight = row.Verdict in HIGHLIGHT_VERDICTS
            for j in range(len(col_keys)):
                cell = table[i + 1, j]
                cell.set_facecolor(bg)
                cell.set_edgecolor('#1a6e2e' if highlight else '#cccccc')
                cell.set_linewidth(2.0 if highlight else 0.5)
            verdict_cell = table[i + 1, col_keys.index('Verdict')]
            if highlight:
                verdict_cell.set_text_props(fontweight='bold', color='#1a6e2e',
                                             fontstyle='italic')
            delta_f1_val  = perf_df.iloc[i]['ΔF1']
            delta_auc_val = perf_df.iloc[i]['ΔAUC']
            table[i + 1, col_keys.index('ΔF1')].set_text_props(
                color='#1a6e2e' if delta_f1_val >= 0 else '#a94442', fontweight='bold')
            table[i + 1, col_keys.index('ΔAUC')].set_text_props(
                color='#1a6e2e' if delta_auc_val >= 0 else '#a94442', fontweight='bold')

        patches = [mpatches.Patch(color=c, label=v) for v, c in VERDICT_COLORS.items()]
        fig_t.legend(handles=patches, loc='lower center', ncol=4,
                     fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=True)
        plt.suptitle('Per-Dataset Performance Summary — Our Best Variant vs Best Other Method',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(cross_dir / 'performance_summary_table.png', dpi=350,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print('  Saved: performance_summary_table.png')

        # Terminal print
        print(f'\n{"="*80}')
        print('PERFORMANCE SUMMARY — Our Best Variant vs Best Other Method')
        print(f'{"="*80}')
        print(f'{"Dataset":<15}  {"Our Variant":<28}  {"F1":>6}  {"AUC":>6}  '
              f'{"Best Other":<18}  {"F1":>6}  {"ΔF1":>7}  Verdict')
        print('-' * 100)
        for _, row in perf_df.iterrows():
            print(f'{row["Dataset"]:<15}  {row["Our Best Variant"]:<28}  '
                  f'{row["Our F1"]:>6.3f}  {row["Our AUC"]:>6.3f}  '
                  f'{row["Best Other Method"]:<18}  {row["Other F1"]:>6.3f}  '
                  f'{row["ΔF1"]:>+7.3f}  {row["Verdict"]}')


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None,
                        help='Single dataset stem to analyse (default: all)')
    args = parser.parse_args()

    labeled_files = sorted(DATA_DIR.glob('*_labeled.csv'))
    if not labeled_files:
        print(f'No labeled CSVs found in {DATA_DIR}')
        print('Run prepare_data_labeled.py first.')
        return

    if args.dataset:
        labeled_files = [f for f in labeled_files if args.dataset in f.stem]

    print(f'Found {len(labeled_files)} labeled dataset(s)\n')

    all_results = {}
    for f in labeled_files:
        print(f'Loading: {f.name}')
        df = pd.read_csv(f)
        if 'label' not in df.columns:
            print(f'  SKIP — no label column')
            continue
        ds_results = analyse_dataset(df, f.stem)
        if ds_results:
            all_results[f.stem] = ds_results

    if len(all_results) > 1:
        print(f'\n{"="*65}')
        print('Cross-dataset analysis...')
        cross_dataset_analysis(all_results)

    print(f'\nAll outputs saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
