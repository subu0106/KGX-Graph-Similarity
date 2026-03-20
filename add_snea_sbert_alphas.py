#!/usr/bin/env python3
"""
Append SNEA-SBERT alpha blend columns to an existing all_methods_results.csv.

For each row the WL score and SBERT score are computed ONCE via
calculate_snea_sbert_similarity, then the 11 alpha blends are derived
from those two values — no redundant model calls.

  alpha=0.0  →  pure SBERT
  alpha=0.5  →  equal blend  (default SNEA-SBERT)
  alpha=1.0  →  pure WL (same as snea_similarity)

Usage:
  python add_snea_sbert_alphas.py
  python add_snea_sbert_alphas.py --limit 10        # quick test
  python add_snea_sbert_alphas.py --resume          # skip rows already done
"""

import ast
import csv
import gc
import sys
import importlib.util
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_CSV = Path('results/final_analysis_semantic_kg/all_methods_results.csv')

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Column names exactly as they appear in semantic_kg_codex_400_KGs_results.csv
NEW_COLS = [
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
]

# alpha value for each column (same order as NEW_COLS)
ALPHA_COLS = dict(zip(ALPHAS, NEW_COLS))

# Base columns already present in all_methods_results.csv
BASE_COLS = [
    'graph_1', 'graph_2', 'similarity_score_ground',
    'kea_similarity', 'kea_composite', 'kea_structural', 'kea_semantic',
    'transe_similarity', 'rotate_similarity',
    'wl_kernel_similarity', 'snea_similarity', 'aa_kea_similarity',
    'snea_bert_similarity', 'kea_bert_similarity', 'semantic_wl_similarity',
]

OUTPUT_FIELDNAMES = BASE_COLS + NEW_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_triples(kg_str):
    try:
        parsed = ast.literal_eval(kg_str) if kg_str and kg_str.strip() else []
    except Exception:
        return []
    if not parsed:
        return []
    if isinstance(parsed[0], list):
        return [t for t in parsed if isinstance(t, list) and len(t) == 3]
    if len(parsed) == 3 and not isinstance(parsed[0], list):
        return [parsed]
    return []


def blend_scores(wl, sbert, fallback):
    """Return {col: score} for all alphas from one wl+sbert pair."""
    scores = {}
    for a in ALPHAS:
        col = ALPHA_COLS[a]
        if wl is not None and sbert is not None:
            scores[col] = float(a * wl + (1.0 - a) * sbert)
        else:
            scores[col] = fallback
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(results_csv, limit, resume):
    # ── Load existing results ─────────────────────────────────────────────────
    with open(results_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)

    if resume:
        sentinel = NEW_COLS[0]
        already_done = sum(
            1 for r in rows
            if r.get(sentinel) not in (None, '', 'None')
        )
        print(f'Resume mode: {already_done}/{total} rows already have alpha scores')

    # ── Load SNEA-SBERT model once, directly from the submodule file ──────────
    # Importing from Methods.snea_sbert directly (not via Methods/__init__.py)
    # prevents all other models (KEA-BERT, Semantic-WL, etc.) from loading.
    print('Loading SNEA-SBERT model (once)...')
    _methods_dir = Path(__file__).parent / 'Methods'

    def _direct_import(name, filename):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, _methods_dir / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    snea_sbert_mod = _direct_import('_snea_sbert_direct', 'snea_sbert.py')
    calculate_snea_sbert = snea_sbert_mod.calculate_snea_sbert_similarity
    print('Model loaded.\n')

    # ── Process rows, write to temp file ─────────────────────────────────────
    tmp_path = results_csv.with_suffix('.tmp')

    with open(tmp_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDNAMES,
                                extrasaction='ignore')
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            # Resume: pass through rows that already have scores
            if resume and row.get(NEW_COLS[0]) not in (None, '', 'None'):
                writer.writerow(row)
                continue

            if idx % 50 == 0 or idx == total:
                print(f'  [{idx:>5}/{total}]', flush=True)

            kg1 = parse_triples(row.get('graph_1', ''))
            kg2 = parse_triples(row.get('graph_2', ''))

            if not kg1 or not kg2:
                row.update({col: None for col in NEW_COLS})
                writer.writerow(row)
                continue

            # ── One call → wl_score + sbert_score_clipped → all alphas ───────
            try:
                _, debug = calculate_snea_sbert(kg1, kg2)
                wl_score    = debug.get('wl_score')
                sbert_score = debug.get('sbert_score_clipped')
                fallback    = debug.get('sbert_score', debug.get('blended_score'))
                row.update(blend_scores(wl_score, sbert_score, fallback))
            except Exception as e:
                print(f'  ERROR row {idx}: {e}')
                row.update({col: None for col in NEW_COLS})

            writer.writerow(row)
            gc.collect()

    # Replace original with updated file
    tmp_path.replace(results_csv)
    print(f'\nDone. Updated: {results_csv}')
    print(f'New columns: {NEW_COLS}')

    # ── Correlation summary ───────────────────────────────────────────────────
    from scipy import stats as scipy_stats

    with open(results_csv, 'r', encoding='utf-8') as f:
        final_rows = list(csv.DictReader(f))

    print(f'\nCorrelation with ground truth (SNEA-SBERT alphas):')
    print(f'{"Column":<38}  {"Pearson r":>10}  {"Spearman r":>11}  {"N":>5}')
    print('-' * 68)
    for col in NEW_COLS:
        pairs = []
        for r in final_rows:
            try:
                pairs.append((float(r['similarity_score_ground']), float(r[col])))
            except (TypeError, ValueError):
                pass
        if len(pairs) < 3:
            print(f'{col:<38}  {"N/A":>10}  {"N/A":>11}  {len(pairs):>5}')
            continue
        g_vals, s_vals = zip(*pairs)
        pr, _ = scipy_stats.pearsonr(g_vals, s_vals)
        sr, _ = scipy_stats.spearmanr(g_vals, s_vals)
        print(f'{col:<38}  {pr:>10.4f}  {sr:>11.4f}  {len(pairs):>5}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add SNEA-SBERT alpha columns to all_methods_results.csv'
    )
    parser.add_argument('--results', default=str(RESULTS_CSV),
                        help='Path to all_methods_results.csv (default: %(default)s)')
    parser.add_argument('--limit',  type=int, default=None,
                        help='Process only first N rows (for testing)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip rows that already have alpha scores filled in')
    args = parser.parse_args()

    run(
        results_csv=Path(args.results),
        limit=args.limit,
        resume=args.resume,
    )
