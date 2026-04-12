"""
For all *_scored.csv files, produce three outputs per file:

  low_gold/     — rows in the bottom 5% of gold_similarity
  low_context/  — rows in the bottom 5% of context_similarity
  filtered/     — rows that pass BOTH thresholds (cleaned dataset)
"""

import os
import csv
import glob
import numpy as np

INPUT_DIR       = os.path.dirname(os.path.abspath(__file__))
DIR_LOW_GOLD    = os.path.join(INPUT_DIR, 'low_gold')
DIR_LOW_CONTEXT = os.path.join(INPUT_DIR, 'low_context')
DIR_FILTERED    = os.path.join(INPUT_DIR, 'filtered')
THRESHOLD       = 0.05   # bottom 5%

for d in [DIR_LOW_GOLD, DIR_LOW_CONTEXT, DIR_FILTERED]:
    os.makedirs(d, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, '*_scored.csv')))
if not files:
    print("No *_scored.csv files found.")
    exit(1)

for filepath in files:
    filename = os.path.basename(filepath)
    stem     = filename.replace('_scored.csv', '')

    with open(filepath, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    gold_scores, ctx_scores = [], []
    for row in rows:
        try:
            gold_scores.append(float(row.get('gold_similarity',   0) or 0))
            ctx_scores.append(float(row.get('context_similarity', 0) or 0))
        except ValueError:
            gold_scores.append(0.0)
            ctx_scores.append(0.0)

    gold_cutoff = np.percentile(gold_scores, THRESHOLD * 100)
    ctx_cutoff  = np.percentile(ctx_scores,  THRESHOLD * 100)

    low_gold    = [row for row, g in zip(rows, gold_scores) if g < gold_cutoff]
    low_context = [row for row, c in zip(rows, ctx_scores)  if c < ctx_cutoff]
    filtered    = [row for row, g, c in zip(rows, gold_scores, ctx_scores)
                   if g >= gold_cutoff and c >= ctx_cutoff]

    def save(out_dir, suffix, subset):
        out_path = os.path.join(out_dir, f'{stem}{suffix}.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    save(DIR_LOW_GOLD,    '_low_gold',    low_gold)
    save(DIR_LOW_CONTEXT, '_low_context', low_context)
    save(DIR_FILTERED,    '_filtered',    filtered)

    print(f'{filename}  ({len(rows)} rows)')
    print(f'  gold_similarity    cutoff: {gold_cutoff:.4f}  → {len(low_gold):>3} low rows  → low_gold/')
    print(f'  context_similarity cutoff: {ctx_cutoff:.4f}  → {len(low_context):>3} low rows  → low_context/')
    print(f'  Both pass                → {len(filtered):>3} rows      → filtered/')
    print()

print('Done.')
print('  low_gold/    → rows in bottom 5% of gold_similarity')
print('  low_context/ → rows in bottom 5% of context_similarity')
print('  filtered/    → cleaned dataset (both scores above 5th percentile)')
