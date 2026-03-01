#!/usr/bin/env python3
"""
Full evaluation of similarity_results.csv vs ground truth.

Sections:
  1. Core          — Pearson, Spearman, MAE, RMSE
  2. Residuals     — bias, std, histogram, residuals vs ground truth
  3. Calibration   — scatter + ideal diagonal + regression (a·x + b)
  4. Significance  — Fisher r-to-z, paired t-test, Wilcoxon vs best method
  5. Bootstrap CI  — 1000-resample 95% CI for Pearson r and RMSE
  6. Variance      — discriminative power (std of method scores)
  7. Ranking table — avg rank across all metrics (significant methods only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_CSV = "results/semantic_kg_eval/similarity_results.csv"
OUT_DIR     = "results/semantic_kg_eval/final_analysis"

METHODS = [
    'kea_similarity',
    # 'kea_composite',
    # 'kea_structural',
    # 'kea_semantic',
    'transe_similarity',
    # 'rotate_similarity',
    'wl_kernel_similarity',
    'aa_kea_similarity',
    'kea_bert_similarity',
]

COLORS = {
    'kea_similarity':       '#2E86AB',
    'kea_composite':        '#1E5F8C',
    'kea_structural':       '#5BA3D0',
    'kea_semantic':         '#88C0D0',
    'transe_similarity':    '#A23B72',
    'rotate_similarity':    '#F18F01',
    'wl_kernel_similarity': '#6A994E',
    'aa_kea_similarity':    '#9B59B6',
    'kea_bert_similarity':  '#E67E22',
}

N_BOOTSTRAP = 1000
CI_LEVEL    = 95
ALPHA       = 0.05



def load(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows  |  columns: {list(df.columns)}\n")
    return df


def paired(df, m):
    """Return (ground_values, method_values) dropping NaN rows."""
    p = df[['similarity_score_ground', m]].dropna()
    return p['similarity_score_ground'].values, p[m].values



def compute_core_metrics(df):
    records = []
    for m in METHODS:
        gv, mv = paired(df, m)
        n = len(gv)
        if n < 3:
            records.append(dict(method=m, pearson_r=None, pearson_p=None,
                                spearman_r=None, spearman_p=None,
                                mae=None, rmse=None, n=n, significant=False))
            continue
        pr, pp = stats.pearsonr(gv, mv)
        sr, sp = stats.spearmanr(gv, mv)
        mae    = np.mean(np.abs(gv - mv))
        rmse   = np.sqrt(np.mean((gv - mv) ** 2))
        records.append(dict(method=m, pearson_r=pr, pearson_p=pp,
                            spearman_r=sr, spearman_p=sp,
                            mae=mae, rmse=rmse, n=n,
                            significant=(pp < ALPHA)))
    return pd.DataFrame(records)


def print_core_metrics(metrics):
    hdr = (f"{'Method':<28}  {'Pearson r':>9}  {'p':>9}  "
           f"{'Spearman r':>10}  {'p':>9}  {'MAE':>7}  {'RMSE':>7}  {'Sig':>4}  N")
    print("=" * len(hdr))
    print("CORE METRICS vs GROUND TRUTH")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for _, r in metrics.iterrows():
        if r['pearson_r'] is None:
            print(f"{r['method']:<28}  {'N/A':>9}  {'N/A':>9}  {'N/A':>10}  "
                  f"{'N/A':>9}  {'N/A':>7}  {'N/A':>7}  {'N/A':>4}  {int(r['n'])}")
        else:
            sig = "YES" if r['significant'] else "no"
            print(f"{r['method']:<28}  {r['pearson_r']:>9.4f}  {r['pearson_p']:>9.3e}  "
                  f"{r['spearman_r']:>10.4f}  {r['spearman_p']:>9.3e}  "
                  f"{r['mae']:>7.4f}  {r['rmse']:>7.4f}  {sig:>4}  {int(r['n'])}")
    print("=" * len(hdr) + "\n")


def plot_scatter_grid(df, out_dir):
    valid = [m for m in METHODS if df[m].notna().sum() >= 3]
    ncols, nrows = 3, -(-len(valid) // 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, m in enumerate(valid):
        ax   = axes[idx]
        gv, mv = paired(df, m)
        pr, _  = stats.pearsonr(gv, mv)
        rmse   = np.sqrt(np.mean((gv - mv) ** 2))
        color  = COLORS.get(m, '#333333')

        ax.scatter(gv, mv, alpha=0.4, s=20, color=color, edgecolors='none')

        # regression line
        a, b = np.polyfit(gv, mv, 1)
        xl   = np.linspace(gv.min(), gv.max(), 200)
        ax.plot(xl, a * xl + b, 'k-', linewidth=1.5, label=f'fit  a={a:.2f}')

        # ideal diagonal
        lo, hi = min(gv.min(), mv.min()), max(gv.max(), mv.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, alpha=0.6, label='y = x')

        ax.set_title(m, fontsize=9, fontweight='bold')
        ax.set_xlabel('Ground Truth', fontsize=8)
        ax.set_ylabel('Method Score', fontsize=8)
        ax.text(0.05, 0.90, f'r={pr:.3f}  RMSE={rmse:.3f}',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.3)

    for idx in range(len(valid), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Scatter: Method Score vs Ground Truth', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'scatter_plots.png')



def compute_residuals(df):
    """residual = method_score - ground_truth"""
    res = {}
    for m in METHODS:
        gv, mv = paired(df, m)
        if len(gv) < 3:
            continue
        r = mv - gv
        res[m] = dict(gv=gv, mv=mv, residuals=r,
                      mean=r.mean(), std=r.std(), n=len(r))
    return res


def print_residuals(res):
    print("=" * 55)
    print("RESIDUAL SUMMARY  (method_score − ground_truth)")
    print("=" * 55)
    print(f"{'Method':<28}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 55)
    for m, d in res.items():
        print(f"{m:<28}  {d['mean']:>8.4f}  {d['std']:>8.4f}  "
              f"{d['residuals'].min():>8.4f}  {d['residuals'].max():>8.4f}")
    print("=" * 55 + "\n")


def plot_residual_analysis(res, out_dir):
    methods = list(res.keys())
    ncols, nrows = 3, -(-len(methods) // 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    for idx, m in enumerate(methods):
        ax    = axes[idx]
        r     = res[m]['residuals']
        color = COLORS.get(m, '#333333')
        ax.hist(r, bins=20, color=color, alpha=0.75, edgecolor='black')
        ax.axvline(0,            color='red',  linestyle='--', linewidth=1.5, label='zero')
        ax.axvline(r.mean(),     color='navy', linestyle='-',  linewidth=1.5,
                   label=f'mean={r.mean():.3f}')
        ax.set_title(m, fontsize=9, fontweight='bold')
        ax.set_xlabel('Residual', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle('Residual Histograms  (want: centred at 0, bell-shaped)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'residual_histograms.png')

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    for idx, m in enumerate(methods):
        ax    = axes[idx]
        gv    = res[m]['gv']
        r     = res[m]['residuals']
        color = COLORS.get(m, '#333333')
        ax.scatter(gv, r, alpha=0.4, s=20, color=color, edgecolors='none')
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        # trend line on residuals
        a, b = np.polyfit(gv, r, 1)
        xl   = np.linspace(gv.min(), gv.max(), 200)
        ax.plot(xl, a * xl + b, 'k-', linewidth=1.2,
                label=f'trend  slope={a:.3f}')
        ax.set_title(m, fontsize=9, fontweight='bold')
        ax.set_xlabel('Ground Truth', fontsize=8)
        ax.set_ylabel('Residual', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        # annotate bias direction
        # bias_txt = 'overestimates' if res[m]['mean'] > 0.01 else \
        #            'underestimates' if res[m]['mean'] < -0.01 else 'unbiased'
        ax.text(0.05, 0.92, f"bias: {res[m]['mean']:.3f}", transform=ax.transAxes, fontsize=8,
                color='darkred', fontweight='bold')
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle('Residuals vs Ground Truth',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'residuals_vs_ground.png')



def compute_calibration(df):
    cal = {}
    for m in METHODS:
        gv, mv = paired(df, m)
        if len(gv) < 3:
            continue
        a, b   = np.polyfit(gv, mv, 1)
        cal[m] = dict(gv=gv, mv=mv, a=a, b=b)
    return cal


def print_calibration(cal):
    print("=" * 60)
    print("CALIBRATION  predicted = a · ground_truth + b")
    print("  Ideal: a ≈ 1, b ≈ 0")
    print("  a < 1 → compresses differences")
    print("  a > 1 → exaggerates differences")
    print("  b ≠ 0 → systematic bias")
    print("=" * 60)
    print(f"{'Method':<28}  {'a (slope)':>10}  {'b (intercept)':>14}")
    print("-" * 60)
    for m, d in cal.items():
        flag = ""
        if abs(d['a'] - 1) > 0.2:
            flag = "  ← slope off"
        if abs(d['b']) > 0.05:
            flag += "  ← bias"
        print(f"{m:<28}  {d['a']:>10.4f}  {d['b']:>14.4f}{flag}")
    print("=" * 60 + "\n")


def plot_calibration(cal, out_dir):
    methods = list(cal.keys())
    ncols, nrows = 3, -(-len(methods) // 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, m in enumerate(methods):
        ax    = axes[idx]
        gv    = cal[m]['gv']
        mv    = cal[m]['mv']
        a, b  = cal[m]['a'], cal[m]['b']
        color = COLORS.get(m, '#333333')

        ax.scatter(gv, mv, alpha=0.4, s=20, color=color, edgecolors='none')

        # ideal diagonal
        lo = min(gv.min(), mv.min())
        hi = max(gv.max(), mv.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, alpha=0.7, label='ideal (a=1,b=0)')

        # fitted line
        xl = np.linspace(gv.min(), gv.max(), 200)
        ax.plot(xl, a * xl + b, 'k-', linewidth=1.5, label=f'fit: a={a:.3f}, b={b:.3f}')

        ax.set_title(m, fontsize=9, fontweight='bold')
        ax.set_xlabel('Ground Truth', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)

    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Calibration: Predicted vs Ground Truth',
                 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'calibration.png')



def fisher_rz_test(r1, r2, n):
    """
    Compare two dependent correlations from the same sample using
    Fisher r-to-z transform (approximate for dependent samples).
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(2.0 / (n - 3))
    z_stat = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p


def significance_tests(df, metrics, out_dir):
    """
    Compare every method against the best method (highest Pearson r).
    Tests: Fisher r-to-z, paired t-test, Wilcoxon signed-rank.
    """
    sig_methods = metrics.dropna(subset=['pearson_r'])
    best_method = sig_methods.loc[sig_methods['pearson_r'].idxmax(), 'method']
    best_r      = sig_methods.loc[sig_methods['pearson_r'].idxmax(), 'pearson_r']
    gv_best, mv_best = paired(df, best_method)
    ae_best = np.abs(gv_best - mv_best)
    n       = len(gv_best)

    print("=" * 80)
    print(f"SIGNIFICANCE TESTS  (vs best method: {best_method}, r={best_r:.4f})")
    print("  Fisher r-to-z : tests if correlation difference is significant")
    print("  Paired t-test : tests if mean absolute error differs")
    print("  Wilcoxon      : non-parametric version of paired t-test")
    print("=" * 80)
    hdr = (f"{'Method':<28}  {'Fisher z':>8}  {'p':>9}  "
           f"{'t-stat':>7}  {'p':>9}  {'W-stat':>7}  {'p':>9}")
    print(hdr)
    print("-" * 80)

    rows = []
    for m in METHODS:
        if m == best_method:
            continue
        gv, mv = paired(df, m)
        if len(gv) < 3:
            continue
        mr, _ = stats.pearsonr(gv, mv)
        ae_m  = np.abs(gv - mv)

        fz, fp       = fisher_rz_test(best_r, mr, n)
        t_stat, t_p  = stats.ttest_rel(ae_best, ae_m)
        try:
            w_stat, w_p = stats.wilcoxon(ae_best, ae_m)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')

        sig_f = "*" if fp   < ALPHA else ""
        sig_t = "*" if t_p  < ALPHA else ""
        sig_w = "*" if w_p  < ALPHA else ""

        print(f"{m:<28}  {fz:>8.3f}{sig_f:1}  {fp:>9.3e}  "
              f"{t_stat:>7.3f}{sig_t:1}  {t_p:>9.3e}  "
              f"{w_stat:>7.1f}{sig_w:1}  {w_p:>9.3e}")
        rows.append(dict(method=m, fisher_z=fz, fisher_p=fp,
                         t_stat=t_stat, t_p=t_p,
                         w_stat=w_stat, w_p=w_p))

    print("* = significant at p < 0.05\n")

    # save as table image
    if not rows:
        return
    headers = ['Method', 'Fisher z', 'Fisher p', 't-stat', 't p-val', 'Wilcoxon W', 'W p-val']
    cell_data = []
    for r in rows:
        cell_data.append([
            r['method'],
            f"{r['fisher_z']:.3f}",
            f"{r['fisher_p']:.3e}",
            f"{r['t_stat']:.3f}",
            f"{r['t_p']:.3e}",
            f"{r['w_stat']:.1f}" if not np.isnan(r['w_stat']) else 'N/A',
            f"{r['w_p']:.3e}"    if not np.isnan(r['w_p'])   else 'N/A',
        ])

    nr = len(cell_data)
    fig, ax = plt.subplots(figsize=(14, 0.55 * nr + 1.8))
    ax.axis('off')
    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(headers))))
    for col in range(len(headers)):
        tbl[0, col].set_facecolor('#2E86AB')
        tbl[0, col].set_text_props(color='white', fontweight='bold')
    for ri in range(1, nr + 1):
        bg = '#f0f4f8' if ri % 2 == 0 else 'white'
        for col in range(len(headers)):
            tbl[ri, col].set_facecolor(bg)
    plt.title(f'Significance Tests vs Best Method: {best_method}\n* p < 0.05',
              fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    _save(fig, out_dir, 'significance_tests.png')



def bootstrap_ci(df, out_dir, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    alpha = (100 - ci) / 2
    results = {}

    print("=" * 60)
    print(f"BOOTSTRAP CI  ({n_boot} resamples, {ci}% CI)")
    print("=" * 60)
    print(f"{'Method':<28}  {'Pearson r':>9}  {'CI':>20}  {'RMSE':>7}  {'CI':>18}")
    print("-" * 60)

    for m in METHODS:
        gv, mv = paired(df, m)
        n = len(gv)
        if n < 3:
            continue
        rng = np.random.default_rng(42)
        boot_r, boot_rmse = [], []
        for _ in range(n_boot):
            idx  = rng.integers(0, n, size=n)
            gb, mb = gv[idx], mv[idx]
            if mb.std() == 0 or gb.std() == 0:
                continue
            r, _   = stats.pearsonr(gb, mb)
            rmse_b = np.sqrt(np.mean((gb - mb) ** 2))
            boot_r.append(r)
            boot_rmse.append(rmse_b)

        r_lo,    r_hi    = np.percentile(boot_r,    [alpha, 100 - alpha])
        rmse_lo, rmse_hi = np.percentile(boot_rmse, [alpha, 100 - alpha])
        pr, _            = stats.pearsonr(gv, mv)
        rmse_obs         = np.sqrt(np.mean((gv - mv) ** 2))

        results[m] = dict(r=pr, r_lo=r_lo, r_hi=r_hi,
                          rmse=rmse_obs, rmse_lo=rmse_lo, rmse_hi=rmse_hi)
        print(f"{m:<28}  {pr:>9.4f}  [{r_lo:>6.4f}, {r_hi:>6.4f}]  "
              f"{rmse_obs:>7.4f}  [{rmse_lo:>5.4f}, {rmse_hi:>5.4f}]")

    print("=" * 60 + "\n")

    # plot
    methods = list(results.keys())
    x = np.arange(len(methods))
    labels = [m.replace('_similarity', '').replace('_', '\n') for m in methods]
    colors = [COLORS.get(m, '#333') for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for ax, key, lo_key, hi_key, title, ylabel in [
        (ax1, 'r',    'r_lo',    'r_hi',    f'Pearson r  ({ci}% CI)',  'r'),
        (ax2, 'rmse', 'rmse_lo', 'rmse_hi', f'RMSE  ({ci}% CI)',       'RMSE'),
    ]:
        vals  = [results[m][key]    for m in methods]
        lo    = [results[m][lo_key] for m in methods]
        hi    = [results[m][hi_key] for m in methods]
        yerr  = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]

        ax.bar(x, vals, color=colors, alpha=0.7, edgecolor='black', width=0.6)
        ax.errorbar(x, vals, yerr=yerr, fmt='none', color='black',
                    capsize=5, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.suptitle(f'Bootstrap Confidence Intervals  ({n_boot} resamples)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'bootstrap_ci.png')



def variance_check(df, out_dir):
    print("=" * 55)
    print("VARIANCE CHECK  (discriminative power)")
    print("  Low std → method cannot distinguish KG pairs")
    print("=" * 55)
    print(f"{'Method':<28}  {'Std':>8}  {'Min':>7}  {'Max':>7}  {'Range':>7}")
    print("-" * 55)

    methods, stds = [], []
    for m in METHODS:
        col = df[m].dropna()
        if len(col) < 3:
            continue
        s = col.std()
        methods.append(m)
        stds.append(s)
        print(f"{m:<28}  {s:>8.4f}  {col.min():>7.4f}  {col.max():>7.4f}  "
              f"{col.max()-col.min():>7.4f}")

    # ground truth std for reference
    g = df['similarity_score_ground'].dropna()
    print(f"\n{'ground_truth (ref)':<28}  {g.std():>8.4f}  "
          f"{g.min():>7.4f}  {g.max():>7.4f}  {g.max()-g.min():>7.4f}")
    print("=" * 55 + "\n")

    colors = [COLORS.get(m, '#333') for m in methods]
    x      = np.arange(len(methods))
    labels = [m.replace('_similarity', '').replace('_', '\n') for m in methods]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, stds, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    ax.axhline(g.std(), color='red', linestyle='--', linewidth=1.5,
               label=f'ground truth std = {g.std():.4f}')
    for bar_, val in zip(bars, stds):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Standard Deviation', fontsize=10)
    ax.set_title('Discriminative Power: Std of Method Scores\n'
                 'Red line = ground truth std (target to match)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, 'variance_check.png')



def plot_ranking_table(metrics, out_dir):
    valid = metrics.dropna(subset=['pearson_r']).copy()
    valid = valid.sort_values('pearson_r', ascending=False)

    headers   = ['Method', 'Pearson r', 'Spearman r', 'MAE', 'RMSE']
    cell_data = []
    for _, row in valid.iterrows():
        cell_data.append([
            row['method'],
            f"{row['pearson_r']:.4f}",
            f"{row['spearman_r']:.4f}",
            f"{row['mae']:.4f}",
            f"{row['rmse']:.4f}",
        ])

    # Find row index of max per numeric column (best = max pearson/spearman, min mae/rmse)
    best_col = {
        1: valid['pearson_r'].idxmax(),
        2: valid['spearman_r'].idxmax(),
        3: valid['mae'].idxmin(),
        4: valid['rmse'].idxmin(),
    }
    # Map pandas index → table row index (1-based, row 0 is header)
    idx_to_row = {idx: ri + 1 for ri, idx in enumerate(valid.index)}

    nr = len(cell_data)
    fig, ax = plt.subplots(figsize=(12, 0.55 * nr + 1.8))
    ax.axis('off')
    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    # Header style
    for col in range(len(headers)):
        tbl[0, col].set_facecolor('#2E86AB')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    # Alternating row background
    for ri in range(1, nr + 1):
        bg = '#f0f4f8' if ri % 2 == 0 else 'white'
        for col in range(len(headers)):
            tbl[ri, col].set_facecolor(bg)

    # Bold the best value per column
    for col, best_idx in best_col.items():
        ri = idx_to_row[best_idx]
        tbl[ri, col].set_text_props(fontweight='bold')

    plt.title('Method Comparison  (Bold = Best value per metric)',
              fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    _save(fig, out_dir, 'ranking_table.png')

    best = valid.iloc[0]
    print(f"\nBEST METHOD (by Pearson r): {best['method']}")
    print(f"  Pearson r  = {best['pearson_r']:.4f}")
    print(f"  Spearman r = {best['spearman_r']:.4f}")
    print(f"  MAE        = {best['mae']:.4f}")
    print(f"  RMSE       = {best['rmse']:.4f}\n")



def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Full evaluation of similarity results')
    parser.add_argument('--input',  type=str, default=RESULTS_CSV)
    parser.add_argument('--outdir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.input)

    # 1. Core
    print("1. Core metrics")
    metrics = compute_core_metrics(df)
    print_core_metrics(metrics)
    plot_scatter_grid(df, args.outdir)

    # 2. Residuals
    print("2. Residual analysis")
    res = compute_residuals(df)
    print_residuals(res)
    plot_residual_analysis(res, args.outdir)

    # 3. Calibration
    print("3. Calibration")
    cal = compute_calibration(df)
    print_calibration(cal)
    plot_calibration(cal, args.outdir)

    # 4. Significance tests
    print("4. Significance tests")
    significance_tests(df, metrics, args.outdir)

    # 5. Bootstrap CI
    print("5. Bootstrap CI")
    bootstrap_ci(df, args.outdir)

    # 6. Variance
    print("6. Variance check")
    variance_check(df, args.outdir)

    # 7. Ranking
    print("7. Final ranking")
    plot_ranking_table(metrics, args.outdir)

    print(f"\nAll outputs saved to: {args.outdir}")
