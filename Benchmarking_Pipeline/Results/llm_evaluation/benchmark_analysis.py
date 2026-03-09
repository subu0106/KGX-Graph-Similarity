"""
LLM Benchmarking Analysis — SNEA-SBERT
Models  : Llama-2-7b-chat-hf, Gemma-7b-it, Mistral-7B-Instruct-v0.2
Datasets: PubMedQA (biomedical), MesaQA (general health)

Analyses:
  0.  Summary stats table      — Average/Median/Std/Max/Min/Perfect/High per model per dataset
  1.  Primary benchmark table  — Mean GoldSim + ContextSim per model per dataset
  2.  CUS table                — Contextual Understanding Score (F1 of gold + context)
  3.  RMSE against gold        — error-based benchmark, lower = better
  4.  Grouped bar chart        — GoldSim / ContextSim / CUS side by side
  5.  Pairwise win-rate        — relative dominance between models (gold_similarity)
  6.  Distribution analysis    — histogram + boxplot + violin + summary stats
  7.  Gold vs Context scatter  — reveals memorisation vs context-grounding trade-off
  8.  Statistical significance — Wilcoxon signed-rank (paired) + Mann-Whitney U
  9.  Radar chart              — multi-dimensional comparison
  10. Heatmap                  — model × dataset × metric grid
"""

import os
import csv
import glob
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(RESULTS_DIR, 'benchmark_plots')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    'Llama-2-7b-chat-hf':       'Llama-2-7B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B',
    'gemma-7b-it':              'Gemma-7B',
}

MODEL_FULL_NAMES = {
    'Llama-2-7B': 'meta-llama/Llama-2-7b-chat-hf',
    'Gemma-7B':   'google/gemma-7b-it',
    'Mistral-7B': 'mistralai/Mistral-7B-Instruct-v0.2',
}

DATASET_LABELS = {
    'mesaqa':   'MesaQA',
    'pubmedqa': 'PubMedQA',
    'pubpubmedqa': 'PubMedQA',
}

MODEL_COLORS = {
    'Llama-2-7B': '#2E86AB',
    'Mistral-7B': '#A23B72',
    'Gemma-7B':   '#6A994E',
}

MODELS   = ['Llama-2-7B', 'Gemma-7B', 'Mistral-7B']
DATASETS = ['PubMedQA', 'MesaQA']


def infer_names(filepath):
    """Return (dataset_label, model_label) inferred from filename."""
    stem = os.path.splitext(os.path.basename(filepath))[0]

    dataset = next(
        (label for key, label in DATASET_LABELS.items() if key in stem.lower()),
        stem.split('_')[0]
    )
    model = next(
        (label for key, label in MODEL_LABELS.items() if key in stem),
        stem
    )
    return dataset, model


def load_all():
    """Load all *_scored.csv files and return a flat list of record dicts."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_scored.csv')))
    if not files:
        print("No *_scored.csv files found in", RESULTS_DIR)
        return []

    records = []
    for filepath in files:
        dataset, model = infer_names(filepath)
        with open(filepath, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    gold = float(row.get('gold_similarity',    0) or 0)
                    ctx  = float(row.get('context_similarity', 0) or 0)
                except ValueError:
                    gold, ctx = 0.0, 0.0
                cus = (2 * gold * ctx / (gold + ctx)) if (gold + ctx) > 0 else 0.0
                records.append({
                    'id':                 row.get('id', ''),
                    'dataset':            dataset,
                    'model':              model,
                    'gold_similarity':    gold,
                    'context_similarity': ctx,
                    'cus':                cus,
                })
    print(f"Loaded {len(records)} rows from {len(files)} files\n")
    return records


def build_summary(records, metric='gold_similarity'):
    """Aggregate mean/median/std/min/max per dataset × model for a given metric."""
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r[metric])

    summary = []
    for (dataset, model), scores in sorted(groups.items()):
        arr = np.array(scores)
        summary.append({
            'dataset': dataset, 'model': model, 'n': len(arr),
            'mean':    arr.mean(),   'median': np.median(arr),
            'std':     arr.std(),    'min':    arr.min(), 'max': arr.max(),
        })
    return summary


def print_summary_table(summary, metric='gold_similarity'):
    print(f"\nSummary — {metric}")
    print(f"{'Dataset':<12}  {'Model':<12}  {'N':>5}  {'Mean':>8}  "
          f"{'Median':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print('-' * 72)
    for r in summary:
        print(f"{r['dataset']:<12}  {r['model']:<12}  {r['n']:>5}  "
              f"{r['mean']:>8.4f}  {r['median']:>8.4f}  {r['std']:>7.4f}  "
              f"{r['min']:>7.4f}  {r['max']:>7.4f}")


def _table_figure(rows, cols, title, filename, figsize=None):
    h = 0.15 * (len(rows) + 1) + 0.6
    fig, ax = plt.subplots(figsize=figsize or (14, h))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for c in range(len(cols)):
        tbl[0, c].set_facecolor('white')
        tbl[0, c].set_text_props(color='black', fontweight='bold')
    for r in range(1, len(rows) + 1):
        for c in range(len(cols)):
            tbl[r, c].set_facecolor('white')
            tbl[r, c].set_text_props(color='black')
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def _summary_rows(records, metric):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            vals = np.array([r[metric] for r in records
                             if r['dataset'] == dataset and r['model'] == model])
            if len(vals) == 0:
                continue
            n       = len(vals)
            perfect = int((vals >= 0.99).sum())
            high    = int((vals >= 0.80).sum())
            rows.append([
                dataset,
                MODEL_FULL_NAMES.get(model, model),
                f'{vals.mean()*100:.2f}',
                f'{np.median(vals)*100:.2f}',
                f'{vals.std()*100:.2f}',
                f'{vals.max()*100:.2f}',
                f'{vals.min()*100:.2f}',
                f'{perfect}/{n} ({100*perfect/n:.1f}%)',
                f'{high}/{n} ({100*high/n:.1f}%)',
            ])
    return rows


def plot_summary_stats(records):
    """
    Analysis 0 — Summary stats tables.
    One table for gold_similarity, one for context_similarity.
    Average/Median/Std/Max/Min/Perfect (≥0.99)/High (≥0.80), scores as percentages.
    """
    cols = ['Dataset', 'Model', 'Average (%)', 'Median (%)', 'Std Dev (%)',
            'Max (%)', 'Min (%)', 'Perfect (≈1.0)', 'High (≥0.8)']

    gold_rows = _summary_rows(records, 'gold_similarity')
    _table_figure(gold_rows, cols,
                  'Statistics of Gold Similarity',
                  '0a_summary_gold.png', figsize=(20, 0.45 * (len(gold_rows) + 1) + 0.6))

    ctx_rows = _summary_rows(records, 'context_similarity')
    _table_figure(ctx_rows, cols,
                  'Statistics of Context Similarity',
                  '0b_summary_context.png', figsize=(20, 0.45 * (len(ctx_rows) + 1) + 0.6))


def plot_comparison(records):
    """
    Analysis 1 — Bar chart (mean ± std) + boxplot, one column per dataset.
    Mirrors the plot_comparison style from llm_benchmark_similarity.py.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r['gold_similarity'])

    n_datasets = len(DATASETS)
    fig, axes  = plt.subplots(2, n_datasets, figsize=(6 * n_datasets, 10))

    for col, dataset in enumerate(DATASETS):
        ax_bar = axes[0][col]
        ax_box = axes[1][col]
        colors = [MODEL_COLORS.get(m, '#888888') for m in MODELS]
        means  = [np.mean(groups[(dataset, m)]) if groups[(dataset, m)] else 0 for m in MODELS]
        stds   = [np.std(groups[(dataset, m)])  if groups[(dataset, m)] else 0 for m in MODELS]
        x      = np.arange(len(MODELS))

        bars = ax_bar.bar(x, means, yerr=stds, capsize=5,
                          color=colors, alpha=0.75, edgecolor='black', linewidth=0.8)
        for bar, mean in zip(bars, means):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{mean:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        ax_bar.set_title(f'{dataset}\nMean Gold Similarity (± std)',
                         fontsize=11, fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(MODELS, fontsize=9)
        ax_bar.set_ylabel('Gold Similarity')
        ax_bar.set_ylim(0, 1.05)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

        data_per_model = [groups[(dataset, m)] for m in MODELS]
        bp = ax_box.boxplot(data_per_model, patch_artist=True,
                            tick_labels=MODELS, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_title(f'{dataset}\nScore Distribution', fontsize=11, fontweight='bold')
        ax_box.set_ylabel('Gold Similarity')
        ax_box.set_ylim(0, 1.05)
        ax_box.grid(axis='y', linestyle='--', alpha=0.4)

    patches = [mpatches.Patch(color=MODEL_COLORS.get(m, '#888888'), label=m) for m in MODELS]
    fig.legend(handles=patches, loc='upper center', ncol=len(MODELS),
               fontsize=10, title='Model', title_fontsize=10,
               bbox_to_anchor=(0.5, 0.98))
    plt.suptitle('LLM KG Quality — SNEA-SBERT Gold Similarity',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, '1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 1_comparison.png')


def plot_primary_table(records):
    """
    Analysis 2 — Primary benchmark table.
    Score_model = (1/N) * sum(Similarity_i), reported for GoldSim and ContextSim.
    """
    rows = []
    for model in MODELS:
        row = [model]
        for dataset in DATASETS:
            vals_g = [r['gold_similarity']    for r in records if r['dataset'] == dataset and r['model'] == model]
            vals_c = [r['context_similarity'] for r in records if r['dataset'] == dataset and r['model'] == model]
            row.append(f'{np.mean(vals_g):.4f}' if vals_g else 'N/A')
            row.append(f'{np.mean(vals_c):.4f}' if vals_c else 'N/A')
        rows.append(row)

    cols = ['Model'] + [f'{d} {m}' for d in DATASETS for m in ['GoldSim', 'ContextSim']]
    _table_figure(rows, cols,
                  'Primary Benchmark Table — Mean Similarity Score per Model per Dataset',
                  '2_primary_table.png')


def plot_cus_table(records):
    """
    Analysis 3 — CUS leaderboard.
    CUS = 2 * (gold * context) / (gold + context)  — harmonic mean (F1-style).
    """
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            sub_gold = [r['gold_similarity']    for r in records if r['dataset'] == dataset and r['model'] == model]
            sub_ctx  = [r['context_similarity'] for r in records if r['dataset'] == dataset and r['model'] == model]
            sub_cus  = [r['cus']                for r in records if r['dataset'] == dataset and r['model'] == model]
            if not sub_cus:
                continue
            rows.append([dataset, model,
                         f'{np.mean(sub_gold):.4f}',
                         f'{np.mean(sub_ctx):.4f}',
                         f'{np.mean(sub_cus):.4f}'])

    rows.sort(key=lambda r: (r[0], -float(r[4])))
    
    cols = [ 'Dataset', 'Model', 'Gold Sim', 'Context Sim', 'CUS (F1)']
    _table_figure(rows, cols,
                  'Contextual Understanding Score',
                  '3_cus_leaderboard.png')


def plot_rmse(records):
    """
    Analysis 4 — RMSE against gold.
    RMSE = sqrt( (1/N) * sum( (1 - similarity)^2 ) )
    Lower RMSE = model answers are closer to perfect gold match.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r['gold_similarity'])

    x     = np.arange(len(DATASETS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(MODELS):
        vals = []
        for dataset in DATASETS:
            arr = np.array(groups[(dataset, model)])
            vals.append(np.sqrt(np.mean((1 - arr) ** 2)) if len(arr) else 0)
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=MODEL_COLORS.get(model, '#888'), alpha=0.85, edgecolor='black', linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel('RMSE (lower = better)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.title('RMSE Against Gold Answer\nRMSE = √( mean( (1 − similarity)² ) )',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '4_rmse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 4_rmse.png')


def plot_grouped_bar(records):
    """
    Analysis 5 — Grouped bar chart: GoldSim, ContextSim, CUS side by side.
    """
    metrics = [
        ('gold_similarity',    'Gold Similarity'),
        ('context_similarity', 'Context Similarity'),
        ('cus',                'CUS (F1 of Gold & Context)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        groups = defaultdict(list)
        for r in records:
            groups[(r['dataset'], r['model'])].append(r[metric])
        x     = np.arange(len(DATASETS))
        width = 0.25
        for i, model in enumerate(MODELS):
            means = [np.mean(groups[(d, model)]) if groups[(d, model)] else 0 for d in DATASETS]
            ax.bar(x + i * width, means, width, label=model,
                   color=MODEL_COLORS.get(model, '#888'), alpha=0.88)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(DATASETS)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('LLM Performance by Dataset and Metric', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, '5_grouped_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 5_grouped_bar.png')


def plot_winrate(records):
    """
    Analysis 6 — Pairwise win-rate on gold_similarity.
    WinRate(A, B) = #(score_A > score_B) / N
    """
    rows = []
    for dataset in DATASETS:
        by_id = defaultdict(dict)
        for r in records:
            if r['dataset'] == dataset:
                by_id[r['id']][r['model']] = r['gold_similarity']

        for m1, m2 in combinations(MODELS, 2):
            pairs = [(v[m1], v[m2]) for v in by_id.values() if m1 in v and m2 in v]
            if not pairs:
                continue
            n    = len(pairs)
            win1 = sum(1 for a, b in pairs if a > b)
            win2 = sum(1 for a, b in pairs if b > a)
            tie  = sum(1 for a, b in pairs if a == b)
            rows.append([dataset, m1, m2,
                         f'{win1/n:.3f}', f'{win2/n:.3f}', f'{tie/n:.3f}', str(n)])

    cols = ['Dataset', 'Model A', 'Model B', 'Win Rate A', 'Win Rate B', 'Tie Rate', 'N']
    _table_figure(rows, cols, 'Pairwise Win-Rate — Gold Similarity', '6_winrate.png')


def plot_distributions(records):
    """
    Analysis 7 — Histogram + boxplot + violin per dataset.
    Summary stats table: mean, median, std, min, max.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r['gold_similarity'])

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for model in MODELS:
            vals = groups[(dataset, model)]
            axes[0].hist(vals, bins=20, alpha=0.55, label=model,
                         color=MODEL_COLORS.get(model, '#888'))
        axes[0].set_xlabel('Gold Similarity')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Histogram')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        data = [groups[(dataset, m)] for m in MODELS]
        bp   = axes[1].boxplot(data, patch_artist=True, tick_labels=MODELS, widths=0.5)
        for patch, model in zip(bp['boxes'], MODELS):
            patch.set_facecolor(MODEL_COLORS.get(model, '#888'))
            patch.set_alpha(0.75)
        axes[1].set_ylabel('Gold Similarity')
        axes[1].set_title('Boxplot')
        axes[1].grid(axis='y', alpha=0.3)

        parts = axes[2].violinplot(data, positions=range(len(MODELS)), showmedians=True)
        for pc, model in zip(parts['bodies'], MODELS):
            pc.set_facecolor(MODEL_COLORS.get(model, '#888'))
            pc.set_alpha(0.75)
        axes[2].set_xticks(range(len(MODELS)))
        axes[2].set_xticklabels(MODELS)
        axes[2].set_ylabel('Gold Similarity')
        axes[2].set_title('Violin Plot')
        axes[2].grid(axis='y', alpha=0.3)

        fig.suptitle(f'Distribution Analysis — {dataset} (Gold Similarity)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUT_DIR, f'7_distribution_{dataset.lower()}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    stat_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            arr = np.array(groups[(dataset, model)])
            if len(arr) == 0:
                continue
            stat_rows.append([dataset, model,
                               f'{arr.mean():.4f}', f'{np.median(arr):.4f}',
                               f'{arr.std():.4f}',  f'{arr.min():.4f}',
                               f'{arr.max():.4f}',  str(len(arr))])
    cols = ['Dataset', 'Model', 'Mean', 'Median', 'Std', 'Min', 'Max', 'N']
    _table_figure(stat_rows, cols, 'Distribution Summary Statistics — Gold Similarity',
                  '7_distribution_stats.png')
    print('Saved: 7_distribution_*.png')


def plot_scatter(records):
    """
    Analysis 8 — Gold vs Context scatter.
    Above diagonal: better context use than factual accuracy (context-grounded).
    Below diagonal: factually accurate but ignores context (memorisation risk).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            gold = [r['gold_similarity']    for r in records if r['dataset'] == dataset and r['model'] == model]
            ctx  = [r['context_similarity'] for r in records if r['dataset'] == dataset and r['model'] == model]
            ax.scatter(gold, ctx, label=model, color=MODEL_COLORS.get(model, '#888'),
                       alpha=0.45, s=18, edgecolors='none')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='gold = context')
        ax.set_xlabel('Gold Similarity (Factual Accuracy)')
        ax.set_ylabel('Context Similarity (Context Utilisation)')
        ax.set_title(dataset, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Gold vs Context Similarity per Model\n'
                 'Above diagonal = better context use than factual accuracy',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, '8_scatter_gold_vs_context.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 8_scatter_gold_vs_context.png')


def plot_significance(records):
    """
    Analysis 9 — Statistical significance.
    Wilcoxon signed-rank (paired) + Mann-Whitney U (unpaired).
    H0: Model A = Model B  |  H1: Model A ≠ Model B
    """
    rows = []
    for dataset in DATASETS:
        by_id = defaultdict(dict)
        for r in records:
            if r['dataset'] == dataset:
                by_id[r['id']][r['model']] = r['gold_similarity']

        for m1, m2 in combinations(MODELS, 2):
            pairs = [(v[m1], v[m2]) for v in by_id.values() if m1 in v and m2 in v]
            if len(pairs) < 5:
                continue
            g1 = np.array([p[0] for p in pairs])
            g2 = np.array([p[1] for p in pairs])

            try:
                _, w_p = stats.wilcoxon(g1, g2)
                sig_w  = '***' if w_p < 0.001 else ('**' if w_p < 0.01 else ('*' if w_p < 0.05 else 'ns'))
                w_str  = f'{w_p:.4e} {sig_w}'
            except Exception:
                w_str  = 'N/A'

            _, mw_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            sig_mw  = '***' if mw_p < 0.001 else ('**' if mw_p < 0.01 else ('*' if mw_p < 0.05 else 'ns'))

            rows.append([dataset, m1, m2,
                         f'{g1.mean():.4f}', f'{g2.mean():.4f}',
                         w_str, f'{mw_p:.4e} {sig_mw}'])

    cols = ['Dataset', 'Model A', 'Model B', 'Mean A', 'Mean B',
            'Wilcoxon p', 'Mann-Whitney p']
    _table_figure(rows, cols,
                  'Statistical Significance Tests\n'
                  '*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
                  '9_significance.png', figsize=(17, 0.62 * len(rows) + 1.8))


def plot_radar(records):
    """
    Analysis 10 — Radar chart.
    4 axes: GoldSim × PubMedQA, ContextSim × PubMedQA, GoldSim × MesaQA, ContextSim × MesaQA.
    """
    categories = ['Gold\nPubMedQA', 'Context\nPubMedQA', 'Gold\nMesaQA', 'Context\nMesaQA']
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.grid(alpha=0.3)

    for model in MODELS:
        vals = []
        for dataset in DATASETS:
            g = [r['gold_similarity']    for r in records if r['dataset'] == dataset and r['model'] == model]
            c = [r['context_similarity'] for r in records if r['dataset'] == dataset and r['model'] == model]
            vals.append(np.mean(g) if g else 0)
            vals.append(np.mean(c) if c else 0)
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=model, color=MODEL_COLORS.get(model, '#888'))
        ax.fill(angles, vals, alpha=0.12, color=MODEL_COLORS.get(model, '#888'))

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Model Capability Radar\n(Gold & Context Similarity per Dataset)',
              fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '10_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 10_radar.png')


def plot_heatmap(records):
    """
    Analysis 11 — Heatmap: model × dataset × metric grid.
    """
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['gold_similarity', 'context_similarity', 'cus']):
        data = defaultdict(dict)
        for r in records:
            data[r['model']][r['dataset']] = r[metric]
        df_h = pd.DataFrame(data).T.reindex(index=MODELS, columns=DATASETS)
        df_h = df_h.apply(pd.to_numeric, errors='coerce')
        df_h = df_h.groupby(level=0).mean()
        pivot = pd.DataFrame(
            {d: {m: np.mean([r[metric] for r in records if r['dataset'] == d and r['model'] == m])
                 for m in MODELS}
             for d in DATASETS}
        ).reindex(MODELS)
        sns.heatmap(pivot, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                    vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.suptitle('Performance Heatmap — Model × Dataset', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, '11_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 11_heatmap.png')


if __name__ == '__main__':
    print('Loading scored CSVs...')
    records = load_all()
    if not records:
        exit(1)

    summary = build_summary(records, 'gold_similarity')
    print_summary_table(summary, 'gold_similarity')

    plot_summary_stats(records)
    plot_comparison(records)
    plot_primary_table(records)
    plot_cus_table(records)
    plot_rmse(records)
    plot_grouped_bar(records)
    plot_winrate(records)
    plot_distributions(records)
    plot_scatter(records)
    plot_significance(records)
    plot_radar(records)
    plot_heatmap(records)

    print(f'\nAll outputs saved to: {OUT_DIR}/')
