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

Output columns = all original columns + the requested score columns.

Usage:
  python score_kgs_with_temps.py
  python score_kgs_with_temps.py --limit 5                      # quick test
  python score_kgs_with_temps.py --resume                       # skip already-done files
  python score_kgs_with_temps.py --temp 0                       # only one temperature folder
  python score_kgs_with_temps.py --scores snea_sbert_gold_llm   # compute only one score (3x faster)

  # Single file (avoids processing all 32 files — use this to avoid server getting stuck)
  python score_kgs_with_temps.py --input KGs_with_temps/0/mesaqa_google_gemma-7b-it_answers_KGs.csv
  python score_kgs_with_temps.py --input KGs_with_temps/0/mesaqa_google_gemma-7b-it_answers_KGs.csv --scores snea_sbert_gold_llm

  # Server / GPU recommended over local MacBook (10-20x faster SBERT inference)
  python score_kgs_with_temps.py --multiprocessing --workers 6  # tune to GPU RAM (~400 MiB/worker)
  python score_kgs_with_temps.py --multiprocessing --workers 6 --batch-size 30

  # Run in background on server so SSH disconnect does not kill it
  nohup python score_kgs_with_temps.py --scores snea_sbert_gold_llm snea_sbert_ctx_llm --multiprocessing --workers 6 > run.log 2>&1 &
  tail -f run.log   # watch progress

Notes:
  - Results are written to a .tmp file first and renamed to .csv only after all rows
    are done. A leftover .tmp file means the run was interrupted before completing.
  - On Apple Silicon (M2/M3) the model runs on MPS (GPU) automatically — no CUDA needed.
  - Embedding cache in snea_sbert.py deduplicates SBERT calls within each row.
"""

import ast
import csv
import gc
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE      = Path(__file__).parent
INPUT_DIR  = _HERE / 'KGs_with_temps'
OUTPUT_DIR = _HERE / 'Results_KGs_with_temps'
ALPHA      = 0.5

SCORE_COLS = [
    'snea_sbert_gold_llm',   # anchor=kg_gold,    filtered=kg_llm  → how much of LLM matches gold
    'snea_sbert_ctx_llm',    # anchor=kg_context, filtered=kg_llm  → how much of LLM is grounded in context
]

DEFAULT_WORKERS    = 4
DEFAULT_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _snea_score(fn, kg_a, kg_b):
    """Call fn(kg_a, kg_b) and return the blended float score, or None on failure."""
    if not kg_a or not kg_b:
        return None
    try:
        similarity, _ = fn(kg_a, kg_b)
        return float(similarity)
    except Exception as e:
        print(f'    ERROR: {e}', flush=True)
        return None


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ---------------------------------------------------------------------------
# Worker  (runs in a spawned subprocess — must re-import the model here)
# ---------------------------------------------------------------------------

def _process_batch(batch_data):
    """
    Worker entry point.

    Each spawned process loads its own copy of the SBERT model on GPU.
    paraphrase-MPNet-base-v2 uses ~400 MiB VRAM, so with N workers the
    total GPU cost is N × ~400 MiB  (e.g. 6 workers → ~2.4 GiB on a
    16 GiB card).

    Args:
        batch_data: (rows: list[dict], base_cols: list[str], active_scores: list[str])

    Returns:
        list[dict] with all original columns + the requested score columns.
    """
    # pylint: disable=import-outside-toplevel
    from Methods.snea_sbert import calculate_snea_sbert_similarity as fn

    rows, base_cols, active_scores = batch_data
    active_set = set(active_scores)
    results = []

    for row in rows:
        kg_gold = parse_triples(row.get('kg_gold',    ''))
        kg_llm  = parse_triples(row.get('kg_llm',     ''))
        kg_ctx  = parse_triples(row.get('kg_context', ''))

        result = {col: row.get(col, '') for col in base_cols}
        if 'snea_sbert_gold_llm' in active_set:
            result['snea_sbert_gold_llm'] = _snea_score(fn, kg_gold, kg_llm)
        if 'snea_sbert_ctx_llm' in active_set:
            result['snea_sbert_ctx_llm']  = _snea_score(fn, kg_ctx,  kg_llm)

        results.append(result)
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def process_file(input_path, output_path, limit=None,
                 use_multiprocessing=False,
                 workers=DEFAULT_WORKERS,
                 batch_size=DEFAULT_BATCH_SIZE,
                 active_scores=None):

    if active_scores is None:
        active_scores = SCORE_COLS
    active_set = set(active_scores)

    with open(input_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total     = len(rows)
    base_cols = list(rows[0].keys()) if rows else []
    extra_cols = [c for c in active_scores if c not in base_cols]
    fieldnames = base_cols + extra_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix('.tmp')

    mode = f'multiprocess (workers={workers}, batch={batch_size})' if use_multiprocessing else 'single-process'
    print(f'    {total} rows  |  scores: {", ".join(active_scores)}  |  {mode}', flush=True)

    with open(tmp_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        if use_multiprocessing:
            batches   = [(list(chunk), base_cols, active_scores) for chunk in _chunk(rows, batch_size)]
            n_batches = len(batches)
            ordered   = [None] * n_batches
            done      = 0

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_process_batch, b): idx
                    for idx, b in enumerate(batches)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        ordered[idx] = future.result()
                    except Exception as e:
                        print(f'    Batch {idx} failed: {e}', flush=True)
                        ordered[idx] = []
                    done += 1
                    done_rows = sum(len(b) for b in ordered if b is not None)
                    print(f'    Batches done: {done}/{n_batches}  ({done_rows}/{total} rows)',
                          flush=True)

            for batch_results in ordered:
                if batch_results:
                    for result in batch_results:
                        writer.writerow(result)
                    out_f.flush()

        else:
            # Single-process path: model is already loaded at import time
            from Methods.snea_sbert import calculate_snea_sbert_similarity as fn

            for idx, row in enumerate(rows, 1):
                print(f'    [{idx:>4}/{total}]', end=' ', flush=True)

                kg_gold = parse_triples(row.get('kg_gold',    ''))
                kg_llm  = parse_triples(row.get('kg_llm',     ''))
                kg_ctx  = parse_triples(row.get('kg_context', ''))

                if 'snea_sbert_gold_llm' in active_set:
                    row['snea_sbert_gold_llm'] = _snea_score(fn, kg_gold, kg_llm)
                if 'snea_sbert_ctx_llm' in active_set:
                    row['snea_sbert_ctx_llm']  = _snea_score(fn, kg_ctx,  kg_llm)

                writer.writerow(row)
                gc.collect()

    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Score KGs_with_temps files with SNEA-SBERT alpha=0.5'
    )
    parser.add_argument('--scores', nargs='+', default=SCORE_COLS, choices=SCORE_COLS,
                        metavar='SCORE',
                        help=f'Which score columns to compute (default: all). '
                             f'Choices: {", ".join(SCORE_COLS)}')
    parser.add_argument('--input',  type=str, default=None,
                        help='Process a single input CSV file (overrides --temp)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for single-file mode (default: Results_KGs_with_temps/<rel_path>)')
    parser.add_argument('--limit',  type=int, default=None,
                        help='Process only first N rows per file (for testing)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Skip files whose output already exists')
    parser.add_argument('--temp',   type=str, default=None,
                        help='Process only one temperature subfolder (e.g. --temp 0)')
    parser.add_argument('--multiprocessing', action='store_true', default=False,
                        help='Use parallel workers (recommended on GPU server)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_WORKERS}). '
                             f'Each worker loads ~400 MiB VRAM. '
                             f'For a 16 GiB card, --workers 6 is a safe upper bound.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Rows per worker batch (default: {DEFAULT_BATCH_SIZE})')
    args = parser.parse_args()

    if args.multiprocessing:
        multiprocessing.set_start_method('spawn', force=True)

    # Single-file mode
    if args.input:
        inp = Path(args.input)
        if args.output:
            out = Path(args.output)
        else:
            try:
                rel = inp.relative_to(INPUT_DIR)
            except ValueError:
                rel = inp.name
            out = OUTPUT_DIR / rel
        print(f'Single-file mode: {inp.name}')
        print(f'Alpha = {ALPHA}  |  Output → {out}')
        if args.resume and out.exists():
            print('Skipping (output exists)')
            return
        process_file(inp, out, limit=args.limit,
                     use_multiprocessing=args.multiprocessing,
                     workers=args.workers, batch_size=args.batch_size,
                     active_scores=args.scores)
        print(f'Saved → {out}')
        return

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
    print(f'Alpha = {ALPHA}  |  Output → {OUTPUT_DIR}')
    if args.multiprocessing:
        print(f'Workers = {args.workers}  |  Batch size = {args.batch_size}  '
              f'|  Est. VRAM = {args.workers} × ~400 MiB = ~{args.workers * 400} MiB')
    print()

    for i, (inp, out) in enumerate(all_files, 1):
        print(f'[{i}/{len(all_files)}] temp={inp.parent.name}  |  {inp.name}')

        if args.resume and out.exists():
            print('    Skipping (output exists)\n')
            continue

        process_file(
            inp, out,
            limit=args.limit,
            use_multiprocessing=args.multiprocessing,
            workers=args.workers,
            batch_size=args.batch_size,
            active_scores=args.scores,
        )
        print(f'    Saved → {out}\n')

    print(f'Done. All outputs in: {OUTPUT_DIR}')
    print('Score columns added to each file:')
    for c in SCORE_COLS:
        print(f'  {c}')


if __name__ == '__main__':
    main()
