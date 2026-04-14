"""
LLM Benchmarking Analysis — SNEA-SBERT
Models  : Llama-2-7b-chat-hf, Gemma-7b-it, Mistral-7B-Instruct-v0.2, Falcon-7b-instruct
Datasets: PubMedQA (biomedical), MesaQA (general health)
Source  : Results_KGs_with_temps/  (temperature subfolders: 0, 0.3, 0.7, 1.0)

Score columns in source CSVs:
  snea_sbert_gold_llm  → GoldSim    (how much LLM output matches gold)
  snea_sbert_ctx_llm   → ContextSim (how much LLM output is grounded in context)

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
  11. Temperature trend        — GoldSim & ContextSim vs temperature per model per dataset
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', 'Results_KGs_with_temps_final')
)
OUT_DIR = os.path.normpath(
    os.path.join(RESULTS_DIR, 'Analysis_plot_temps')
)
os.makedirs(OUT_DIR, exist_ok=True)

# Score column names in the source CSVs
COL_GOLD = 'snea_sbert_gold_llm'
COL_CTX  = 'snea_sbert_ctx_llm'

# ---------------------------------------------------------------------------
# Model / dataset metadata
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    'Llama-2-7b-chat-hf':       'Llama-2-7B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B',
    'gemma-7b-it':              'Gemma-7B',
    'falcon-7b-instruct':       'Falcon-7B',
}

MODEL_FULL_NAMES = {
    'Llama-2-7B': 'meta-llama/Llama-2-7b-chat-hf',
    'Gemma-7B':   'google/gemma-7b-it',
    'Mistral-7B': 'mistralai/Mistral-7B-Instruct-v0.2',
    'Falcon-7B':  'tiiuae/falcon-7b-instruct',
}

DATASET_LABELS = {
    'mesaqa':   'MesaQA',
    'pubmedqa': 'PubMedQA',
}

MODEL_COLORS = {
    'Llama-2-7B': '#2E86AB',
    'Mistral-7B': '#A23B72',
    'Gemma-7B':   '#6A994E',
    'Falcon-7B':  '#E07B39',
}

MODELS       = ['Llama-2-7B', 'Gemma-7B', 'Mistral-7B', 'Falcon-7B']
DATASETS     = ['PubMedQA', 'MesaQA']
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_names(filepath):
    """Return (dataset_label, model_label, temperature_float) from filepath."""
    stem   = os.path.splitext(os.path.basename(filepath))[0]
    parent = os.path.basename(os.path.dirname(filepath))   # "0", "0.3", "0.7", "1.0"

    dataset = next(
        (label for key, label in DATASET_LABELS.items() if key in stem.lower()),
        stem.split('_')[0]
    )
    model = next(
        (label for key, label in MODEL_LABELS.items() if key in stem),
        stem
    )
    try:
        temperature = float(parent)
    except ValueError:
        temperature = 0.0

    return dataset, model, temperature


def load_all():
    """Load all CSV files from Results_KGs_with_temps/ subfolders."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '**', '*.csv'), recursive=True))
    if not files:
        print("No CSV files found in", RESULTS_DIR)
        return []

    records = []
    for filepath in files:
        dataset, model, temperature = infer_names(filepath)
        with open(filepath, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if COL_GOLD not in row or COL_CTX not in row:
                    continue
                try:
                    gold = float(row.get(COL_GOLD, 0) or 0)
                    ctx  = float(row.get(COL_CTX,  0) or 0)
                except ValueError:
                    gold, ctx = 0.0, 0.0
                cus = (2 * gold * ctx / (gold + ctx)) if (gold + ctx) > 0 else 0.0
                records.append({
                    'id':                 row.get('id', ''),
                    'dataset':            dataset,
                    'model':              model,
                    'temperature':        temperature,
                    'gold_similarity':    gold,
                    'context_similarity': ctx,
                    'cus':                cus,
                })
    print(f"Loaded {len(records)} rows from {len(files)} files\n")
    return records


def build_summary(records, metric='gold_similarity'):
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


def _print_table(rows, cols, title):
    print(f'\n{title}')
    widths = [max(len(str(cols[i])), max((len(str(r[i])) for r in rows), default=0))
              for i in range(len(cols))]
    sep    = '  '.join('-' * w for w in widths)
    header = '  '.join(str(cols[i]).ljust(widths[i]) for i in range(len(cols)))
    print(header)
    print(sep)
    for row in rows:
        print('  '.join(str(row[i]).ljust(widths[i]) for i in range(len(cols))))


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


# ---------------------------------------------------------------------------
# Analysis 0 — Summary stats
# ---------------------------------------------------------------------------

def plot_summary_stats(records):
    cols = ['Dataset', 'Model', 'Average (%)', 'Median (%)', 'Std Dev (%)',
            'Max (%)', 'Min (%)', 'Perfect (≈1.0)', 'High (≥0.8)']

    gold_rows = _summary_rows(records, 'gold_similarity')
    _print_table(gold_rows, cols, 'Statistics of Gold Similarity')
    _table_figure(gold_rows, cols,
                  'Statistics of Gold Similarity',
                  '0a_summary_gold.png', figsize=(20, 0.45 * (len(gold_rows) + 1) + 0.6))

    ctx_rows = _summary_rows(records, 'context_similarity')
    _print_table(ctx_rows, cols, 'Statistics of Context Similarity')
    _table_figure(ctx_rows, cols,
                  'Statistics of Context Similarity',
                  '0b_summary_context.png', figsize=(20, 0.45 * (len(ctx_rows) + 1) + 0.6))


# ---------------------------------------------------------------------------
# Analysis 1 — Bar chart + boxplot
# ---------------------------------------------------------------------------

def plot_comparison(records):
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


# ---------------------------------------------------------------------------
# Analysis 2 — Primary benchmark table
# ---------------------------------------------------------------------------

def plot_primary_table(records):
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
    _print_table(rows, cols, 'Primary Benchmark Table — Mean Similarity Score per Model per Dataset')
    _table_figure(rows, cols,
                  'Primary Benchmark Table — Mean Similarity Score per Model per Dataset',
                  '2_primary_table.png')


# ---------------------------------------------------------------------------
# Analysis 3 — CUS leaderboard
# ---------------------------------------------------------------------------

def plot_cus_table(records):
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
    cols = ['Dataset', 'Model', 'Gold Sim', 'Context Sim', 'CUS (F1)']
    _print_table(rows, cols, 'Contextual Understanding Score (CUS) Leaderboard')
    _table_figure(rows, cols,
                  'Contextual Understanding Score',
                  '3_cus_leaderboard.png')


# ---------------------------------------------------------------------------
# Analysis 4 — RMSE
# ---------------------------------------------------------------------------

def plot_rmse(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r['gold_similarity'])

    x     = np.arange(len(DATASETS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(MODELS):
        vals = []
        for dataset in DATASETS:
            arr = np.array(groups[(dataset, model)])
            vals.append(np.sqrt(np.mean((1 - arr) ** 2)) if len(arr) else 0)
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=MODEL_COLORS.get(model, '#888'), alpha=0.85,
                      edgecolor='black', linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
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


# ---------------------------------------------------------------------------
# Analysis 5 — Grouped bar chart
# ---------------------------------------------------------------------------

def plot_grouped_bar(records):
    metrics = [
        ('gold_similarity',    'Gold Similarity'),
        ('context_similarity', 'Context Similarity'),
        ('cus',                'CUS (F1 of Gold & Context)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    width = 0.2
    for ax, (metric, title) in zip(axes, metrics):
        groups = defaultdict(list)
        for r in records:
            groups[(r['dataset'], r['model'])].append(r[metric])
        x = np.arange(len(DATASETS))
        for i, model in enumerate(MODELS):
            means = [np.mean(groups[(d, model)]) if groups[(d, model)] else 0 for d in DATASETS]
            ax.bar(x + i * width, means, width, label=model,
                   color=MODEL_COLORS.get(model, '#888'), alpha=0.88)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(DATASETS)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('LLM Performance by Dataset and Metric', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, '5_grouped_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: 5_grouped_bar.png')


# ---------------------------------------------------------------------------
# Analysis 6 — Pairwise win-rate
# ---------------------------------------------------------------------------

def plot_winrate(records):
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
    _print_table(rows, cols, 'Pairwise Win-Rate — Gold Similarity')
    _table_figure(rows, cols, 'Pairwise Win-Rate — Gold Similarity', '6_winrate.png')


# ---------------------------------------------------------------------------
# Analysis 7 — Distribution analysis
# ---------------------------------------------------------------------------

def plot_distributions(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['dataset'], r['model'])].append(r['gold_similarity'])

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for model in MODELS:
            vals = groups[(dataset, model)]
            if vals:
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

        parts = axes[2].violinplot([d if d else [0] for d in data],
                                   positions=range(len(MODELS)), showmedians=True)
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
    _print_table(stat_rows, cols, 'Distribution Summary Statistics — Gold Similarity')
    _table_figure(stat_rows, cols, 'Distribution Summary Statistics — Gold Similarity',
                  '7_distribution_stats.png')
    print('Saved: 7_distribution_*.png')


# ---------------------------------------------------------------------------
# Analysis 8 — Gold vs Context scatter
# ---------------------------------------------------------------------------

def plot_scatter(records):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            gold = [r['gold_similarity']    for r in records if r['dataset'] == dataset and r['model'] == model]
            ctx  = [r['context_similarity'] for r in records if r['dataset'] == dataset and r['model'] == model]
            if gold:
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


# ---------------------------------------------------------------------------
# Analysis 9 — Statistical significance
# ---------------------------------------------------------------------------

def plot_significance(records):
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
    _print_table(rows, cols,
                 'Statistical Significance Tests  (*** p<0.001  ** p<0.01  * p<0.05  ns)')
    _table_figure(rows, cols,
                  'Statistical Significance Tests\n'
                  '*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
                  '9_significance.png', figsize=(17, 0.62 * len(rows) + 1.8))


# ---------------------------------------------------------------------------
# Analysis 10 — Radar chart
# ---------------------------------------------------------------------------

def plot_radar(records):
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


# ---------------------------------------------------------------------------
# Analysis 11 — Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(records):
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, metric in zip(axes, ['gold_similarity', 'context_similarity', 'cus']):
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


# ---------------------------------------------------------------------------
# Analysis 12 — Temperature trend
# ---------------------------------------------------------------------------

def plot_temperature_trend(records):
    """
    Line chart: mean GoldSim, ContextSim, and CUS vs temperature (0, 0.3, 0.7, 1.0).
    One subplot per dataset, one line per model.
    """
    for metric, metric_label, filename_tag in [
        ('gold_similarity',    'Gold Similarity',    'gold'),
        ('context_similarity', 'Context Similarity', 'context'),
        ('cus',                'CUS (F1)',            'cus'),
    ]:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5), sharey=True)

        for ax, dataset in zip(axes, DATASETS):
            for model in MODELS:
                means = []
                for temp in TEMPERATURES:
                    vals = [r[metric] for r in records
                            if r['dataset'] == dataset
                            and r['model'] == model
                            and r['temperature'] == temp]
                    means.append(np.mean(vals) if vals else np.nan)

                if any(not np.isnan(v) for v in means):
                    ax.plot(TEMPERATURES, means,
                            marker='o', linewidth=2, markersize=6,
                            label=model, color=MODEL_COLORS.get(model, '#888'))
                    for x, y in zip(TEMPERATURES, means):
                        if not np.isnan(y):
                            ax.annotate(f'{y:.3f}', (x, y),
                                        textcoords='offset points', xytext=(0, 8),
                                        ha='center', fontsize=7)

            ax.set_title(dataset, fontweight='bold')
            ax.set_xlabel('Temperature')
            ax.set_ylabel(metric_label)
            ax.set_xticks(TEMPERATURES)
            ax.set_xticklabels([str(t) for t in TEMPERATURES])
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle(f'Effect of Temperature on {metric_label}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f'12_temperature_trend_{filename_tag}.png'
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {fname}')

    # Summary table
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for temp in TEMPERATURES:
                gold_vals = [r['gold_similarity'] for r in records
                             if r['dataset'] == dataset and r['model'] == model and r['temperature'] == temp]
                ctx_vals  = [r['context_similarity'] for r in records
                             if r['dataset'] == dataset and r['model'] == model and r['temperature'] == temp]
                if not gold_vals:
                    continue
                rows.append([dataset, model, str(temp),
                             f'{np.mean(gold_vals):.4f}',
                             f'{np.mean(ctx_vals):.4f}' if ctx_vals else 'N/A'])

    cols = ['Dataset', 'Model', 'Temperature', 'Mean GoldSim', 'Mean ContextSim']
    _print_table(rows, cols, 'Temperature Effect — Mean Scores per Model per Dataset')
    _table_figure(rows, cols,
                  'Effect of Temperature on SNEA-SBERT Scores',
                  '12_temperature_table.png',
                  figsize=(16, 0.4 * (len(rows) + 1) + 0.8))


# ---------------------------------------------------------------------------
# Analysis 13 — Evaluation results table at T=0
# ---------------------------------------------------------------------------

def plot_eval_table_t0(records, temperature=0.0):
    """
    Table of mean GoldSim / CtxSim / CUS per model per dataset at a single
    temperature (default T=0). Bold marks the best value per metric per dataset.
    """
    t0 = [r for r in records if abs(r['temperature'] - temperature) < 1e-6]
    if not t0:
        print(f'No records found for temperature={temperature}')
        return

    rows = []
    for dataset in ['MesaQA', 'PubMedQA']:
        # Compute means per model
        means = {}
        for model in MODELS:
            sub = [r for r in t0 if r['dataset'] == dataset and r['model'] == model]
            if sub:
                means[model] = (
                    np.mean([r['gold_similarity']    for r in sub]),
                    np.mean([r['context_similarity'] for r in sub]),
                    np.mean([r['cus']                for r in sub]),
                )

        if not means:
            continue

        best_gold = max(v[0] for v in means.values())
        best_ctx  = max(v[1] for v in means.values())
        best_cus  = max(v[2] for v in means.values())

        for model in MODELS:
            if model not in means:
                continue
            g, c, u = means[model]
            rows.append([
                dataset,
                model,
                f'{"*" if abs(g - best_gold) < 1e-6 else ""}{g:.4f}',
                f'{"*" if abs(c - best_ctx)  < 1e-6 else ""}{c:.4f}',
                f'{"*" if abs(u - best_cus)  < 1e-6 else ""}{u:.4f}',
            ])

    cols = ['Dataset', 'Model', 'GoldSim', 'CtxSim', 'CUS']
    title = f'LLM Evaluation Results — Mean per Model per Dataset (T={temperature})\n* = best per metric per dataset'
    _print_table(rows, cols, title)
    _table_figure(rows, cols, title, f'13_eval_table_t{temperature}.png', figsize=(10, 0.4 * len(rows) + 1.2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading scored CSVs from Results_KGs_with_temps/ ...')
    print(f'Data  : {RESULTS_DIR}')
    print(f'Output: {OUT_DIR}\n')
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
    plot_temperature_trend(records)
    plot_eval_table_t0(records, temperature=0.0)

    print(f'\nAll outputs saved to: {OUT_DIR}/')