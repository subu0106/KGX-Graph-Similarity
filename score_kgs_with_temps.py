#!/usr/bin/env python3
"""
Compute SNEA-SBERT (alpha=0.5) similarity scores for all KG files in
KGs_with_temps/ and write results to Results_KGs_with_temps/ preserving
the same subfolder / filename structure.

For each row, three pair-wise scores are computed. In each call,
kg1 is the anchor that drives triple filtering and kg2 is filtered
against it (match_and_filter_triples(kg1, kg2)).

  Score                │ kg1 (anchor)  │ kg2 (filtered) │ What it measures
  ─────────────────────┼───────────────┼────────────────┼──────────────────────────────────────────────
  snea_sbert_gold_llm  │ kg_gold       │ kg_llm         │ How much of LLM output matches gold
  snea_sbert_ctx_llm   │ kg_context    │ kg_llm         │ How much of LLM output is grounded in context
  snea_sbert_ctx_gold  │ kg_context    │ kg_gold        │ How much of gold answer is grounded in context

Output columns = all original columns + the three score columns above.

Usage:
  python score_kgs_with_temps.py
  python score_kgs_with_temps.py --limit 5          # quick test
  python score_kgs_with_temps.py --resume           # skip already-done files
  python score_kgs_with_temps.py --temp 0           # only one temperature folder
"""

import ast
import csv
import gc
import argparse
from pathlib import Path
from Methods.snea_sbert import calculate_snea_sbert_similarity

_HERE      = Path(__file__).parent
INPUT_DIR  = _HERE / 'KGs_with_temps'
OUTPUT_DIR = _HERE / 'Results_KGs_with_temps'
ALPHA      = 0.5

SCORE_COLS = [
    'snea_sbert_gold_llm',   # anchor=kg_gold,    filtered=kg_llm  → how much of LLM matches gold
    'snea_sbert_ctx_llm',    # anchor=kg_context, filtered=kg_llm  → how much of LLM is grounded in context
    'snea_sbert_ctx_gold',   # anchor=kg_context, filtered=kg_gold → how much of gold is grounded in context
]


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


def snea_score(fn, kg_a, kg_b):
    """Call calculate_snea_sbert_similarity and return its blended score directly."""
    if not kg_a or not kg_b:
        return None
    try:
        similarity, _ = fn(kg_a, kg_b)
        return float(similarity)
    except Exception as e:
        print(f'    ERROR: {e}')
        return None


def process_file(input_path, output_path, fn, limit=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    base_cols = list(rows[0].keys()) if rows else []
    extra_cols = [c for c in SCORE_COLS if c not in base_cols]
    fieldnames = base_cols + extra_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix('.tmp')

    with open(tmp_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            if idx % 100 == 0 or idx == total:
                print(f'    [{idx:>4}/{total}]', flush=True)

            kg_gold = parse_triples(row.get('kg_gold', ''))
            kg_llm  = parse_triples(row.get('kg_llm',  ''))
            kg_ctx  = parse_triples(row.get('kg_context', ''))

            # kg1=anchor (drives filtering), kg2=what gets filtered against kg1
            row['snea_sbert_gold_llm'] = snea_score(fn, kg_gold, kg_llm)  # gold anchors    → how much of LLM matches gold
            row['snea_sbert_ctx_llm']  = snea_score(fn, kg_ctx,  kg_llm)  # context anchors → how much of LLM is grounded in context
            row['snea_sbert_ctx_gold'] = snea_score(fn, kg_ctx,  kg_gold) # context anchors → how much of gold is grounded in context

            writer.writerow(row)
            gc.collect()

    tmp_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Score KGs_with_temps files with SNEA-SBERT alpha=0.5'
    )
    parser.add_argument('--limit',  type=int, default=None,
                        help='Process only first N rows per file (for testing)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip files whose output already exists')
    parser.add_argument('--temp',   type=str, default=None,
                        help='Process only one temperature subfolder (e.g. --temp 0)')
    args = parser.parse_args()

    if args.temp:
        temp_dirs = [INPUT_DIR / args.temp]
    else:
        temp_dirs = sorted(d for d in INPUT_DIR.iterdir() if d.is_dir())

    all_files = []
    for temp_dir in temp_dirs:
        for csv_file in sorted(temp_dir.glob('*.csv')):
            rel      = csv_file.relative_to(INPUT_DIR)
            out_path = OUTPUT_DIR / rel
            all_files.append((csv_file, out_path))

    if not all_files:
        print(f'No CSV files found under {INPUT_DIR}')
        return

    print(f'Found {len(all_files)} file(s) across {len(temp_dirs)} temperature folder(s)')
    print(f'Alpha = {ALPHA}  |  Output → {OUTPUT_DIR}\n')

    for i, (inp, out) in enumerate(all_files, 1):
        print(f'[{i}/{len(all_files)}] temp={inp.parent.name}  |  {inp.name}')

        if args.resume and out.exists():
            print('    Skipping (output exists)\n')
            continue

        process_file(inp, out, calculate_snea_sbert_similarity, limit=args.limit)
        print(f'    Saved → {out}\n')

    print(f'Done. All outputs in: {OUTPUT_DIR}')
    print('Score columns added to each file:')
    for c in SCORE_COLS:
        print(f'  {c}')


if __name__ == '__main__':
    main()
