#!/usr/bin/env python3
"""
Evaluate all KG similarity methods across datasets in Benchmarking_Pipeline/Data/.

Input CSV columns : id, paragraph_1, paragraph_2, kg_1, kg_2
Output CSV columns: above + all method score columns

Methods:
  kea_similarity, transe_similarity, rotate_similarity, wl_kernel_similarity,
  snea_similarity, aa_kea_similarity, kea_bert_similarity, semantic_wl_similarity,
  snea_bert_alpha_0.0 .. snea_bert_alpha_1.0  (one column per alpha value)

Usage:
  Process every dataset under Benchmarking_Pipeline/Data/ (default):
    python evaluate_all_datasets.py

  Single file:
    python evaluate_all_datasets.py --input path/to/data.csv --output path/to/results.csv

  Limit rows per dataset (for quick testing):
    python evaluate_all_datasets.py --limit 10

  Choose specific methods:
    python evaluate_all_datasets.py --methods kea_similarity wl_kernel_similarity

  Parallel workers (safe on CPU; use with care on macOS + CUDA):
    python evaluate_all_datasets.py --multiprocessing --workers 4 --batch-size 20
"""

import csv
import ast
import os
import gc
import glob
import traceback
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

SNEA_BERT_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

METHODS = [
    'kea_similarity',
    'transe_similarity',
    'rotate_similarity',
    'wl_kernel_similarity',
    'snea_similarity',
    'aa_kea_similarity',
    *[f'snea_bert_alpha_{a}' for a in SNEA_BERT_ALPHAS],
    'kea_bert_similarity',
    'semantic_wl_similarity',
]

INPUT_FIELDNAMES  = ['id', 'paragraph_1', 'paragraph_2', 'kg_1', 'kg_2']
OUTPUT_FIELDNAMES = INPUT_FIELDNAMES + METHODS

DEFAULT_WORKERS    = 4
DEFAULT_BATCH_SIZE = 20


def parse_triples(kg_str):
    """Parse a stringified list-of-lists into valid (len==3) triples."""
    try:
        parsed = ast.literal_eval(kg_str) if kg_str and kg_str.strip() else []
    except Exception:
        return []
    return [t for t in parsed if isinstance(t, list) and len(t) == 3]


def _compute_scores(kg1, kg2, active_methods):
    """
    Run each active method on one KG pair.
    Imports live here so spawned workers load SBERT once into their own context.
    Returns dict {method: float | None}.
    """
    from Methods import (
        calculate_kea_similarity,
        calculate_wl_kernel_similarity,
        calculate_snea_similarity_score,
        calculate_aa_kea_similarity,
        calculate_snea_sbert_similarity,
        calculate_kea_bert_similarity_score,
        calculate_semantic_wl_similarity_score,
        GraphEmbeddingSimilarity,
    )

    scores = {m: None for m in active_methods}

    if not kg1 or not kg2:
        return scores

    if 'kea_similarity' in active_methods:
        try:
            result = calculate_kea_similarity(kg1, kg2)
            scores['kea_similarity'] = result[0] if isinstance(result, tuple) else result
        except Exception:
            pass

    # TransE and RotatE share one embedding calculator
    if 'transe_similarity' in active_methods or 'rotate_similarity' in active_methods:
        try:
            emb = GraphEmbeddingSimilarity(embedding_dim=50)
            if 'transe_similarity' in active_methods:
                try:
                    scores['transe_similarity'] = emb.calculate_transe_similarity(kg1, kg2)
                except Exception:
                    pass
            if 'rotate_similarity' in active_methods:
                try:
                    scores['rotate_similarity'] = emb.calculate_rotate_similarity(kg1, kg2)
                except Exception:
                    pass
            del emb
        except Exception:
            pass

    if 'wl_kernel_similarity' in active_methods:
        try:
            scores['wl_kernel_similarity'] = calculate_wl_kernel_similarity(kg1, kg2)
        except Exception:
            pass

    if 'snea_similarity' in active_methods:
        try:
            scores['snea_similarity'] = calculate_snea_similarity_score(kg1, kg2)
        except Exception:
            pass

    if 'aa_kea_similarity' in active_methods:
        try:
            scores['aa_kea_similarity'] = calculate_aa_kea_similarity(kg1, kg2)
        except Exception:
            pass

    for a in SNEA_BERT_ALPHAS:
        col = f'snea_bert_alpha_{a}'
        if col in active_methods:
            try:
                sim, _ = calculate_snea_sbert_similarity(kg1, kg2, alpha=a)
                scores[col] = sim
            except Exception:
                pass

    if 'kea_bert_similarity' in active_methods:
        try:
            scores['kea_bert_similarity'] = calculate_kea_bert_similarity_score(kg1, kg2)
        except Exception:
            pass

    if 'semantic_wl_similarity' in active_methods:
        try:
            scores['semantic_wl_similarity'] = calculate_semantic_wl_similarity_score(kg1, kg2)
        except Exception:
            pass

    return scores


def _process_batch(batch_data):
    """Worker: receives (list[row_dict], active_methods). Returns list[result_dict]."""
    rows, active_methods = batch_data
    results = []
    for row in rows:
        kg1 = parse_triples(row.get('kg_1', ''))
        kg2 = parse_triples(row.get('kg_2', ''))
        scores = _compute_scores(kg1, kg2, active_methods)
        result = {
            'id':          row.get('id', ''),
            'paragraph_1': row.get('paragraph_1', ''),
            'paragraph_2': row.get('paragraph_2', ''),
            'kg_1':        row.get('kg_1', ''),
            'kg_2':        row.get('kg_2', ''),
        }
        result.update(scores)
        results.append(result)
        gc.collect()
    return results


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def process_dataset(input_file, output_file, limit=None, active_methods=None,
                    use_multiprocessing=False, workers=DEFAULT_WORKERS,
                    batch_size=DEFAULT_BATCH_SIZE):
    """
    Process one dataset CSV and write results.

    Returns:
        (valid_counts, error_counts) dicts keyed by method name.
    """
    if active_methods is None:
        active_methods = METHODS

    fieldnames = INPUT_FIELDNAMES + active_methods

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    mode = 'multiprocess' if use_multiprocessing else 'single-process'
    print(f"  {total} rows  |  mode: {mode}  ->  {output_file}")

    all_results = []

    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        if use_multiprocessing:
            batches   = [(list(chunk), active_methods) for chunk in _chunk(rows, batch_size)]
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
                        print(f"  Batch {idx} failed: {e}")
                        ordered[idx] = []
                    done += 1
                    done_rows = sum(len(b) for b in ordered if b is not None)
                    print(f"  Batches done: {done}/{n_batches}  ({done_rows}/{total} rows)",
                          flush=True)

            for batch_results in ordered:
                if batch_results:
                    for result in batch_results:
                        writer.writerow(result)
                        all_results.append(result)
                    out_f.flush()

        else:
            for idx, row in enumerate(rows, 1):
                kg1 = parse_triples(row.get('kg_1', ''))
                kg2 = parse_triples(row.get('kg_2', ''))

                print(f"  [{idx:>4}/{total}] id={row.get('id', idx-1)}"
                      f"  kg1={len(kg1)} triples  kg2={len(kg2)} triples", flush=True)

                scores = _compute_scores(kg1, kg2, active_methods)

                result = {
                    'id':          row.get('id', ''),
                    'paragraph_1': row.get('paragraph_1', ''),
                    'paragraph_2': row.get('paragraph_2', ''),
                    'kg_1':        row.get('kg_1', ''),
                    'kg_2':        row.get('kg_2', ''),
                }
                result.update(scores)

                writer.writerow(result)
                out_f.flush()
                all_results.append(result)
                gc.collect()

    # Coverage summary
    valid_counts = {m: 0 for m in active_methods}
    error_counts = {m: 0 for m in active_methods}
    for result in all_results:
        for m in active_methods:
            val = result.get(m)
            if val is not None and val != '':
                valid_counts[m] += 1
            else:
                error_counts[m] += 1

    print(f"\n  Coverage ({total} rows):")
    for m in active_methods:
        print(f"    {m:<32}  valid={valid_counts[m]}  errors={error_counts[m]}")

    return valid_counts, error_counts


def discover_datasets(data_dir):
    pattern = os.path.join(data_dir, '*.csv')
    return sorted(glob.glob(pattern))


def main():
    script_dir          = os.path.dirname(os.path.abspath(__file__))
    default_data_dir    = os.path.join(script_dir, 'Benchmarking_Pipeline', 'Data')
    default_results_dir = os.path.join(script_dir, 'Benchmarking_Pipeline', 'Results_All_Methods')

    parser = argparse.ArgumentParser(
        description='Evaluate KG similarity methods on Benchmarking_Pipeline/Data datasets.'
    )
    parser.add_argument('--data-dir',    default=default_data_dir,
                        help='Root directory of input CSVs (searched recursively)')
    parser.add_argument('--results-dir', default=default_results_dir,
                        help='Root directory for output CSVs')
    parser.add_argument('--input',       default=None,
                        help='Single input CSV (overrides --data-dir)')
    parser.add_argument('--output',      default=None,
                        help='Output path for single-file mode')
    parser.add_argument('--methods',     nargs='+', default=None, choices=METHODS,
                        metavar='METHOD',
                        help='Subset of methods to run (default: all)')
    parser.add_argument('--limit',       type=int, default=None,
                        help='Max rows per dataset (for quick testing)')
    parser.add_argument('--multiprocessing', action='store_true', default=False,
                        help='Use parallel workers (may be unsafe on macOS with CUDA)')
    parser.add_argument('--workers',     type=int, default=DEFAULT_WORKERS,
                        help=f'Worker count for parallel mode (default: {DEFAULT_WORKERS})')
    parser.add_argument('--batch-size',  type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Rows per batch for parallel mode (default: {DEFAULT_BATCH_SIZE})')
    args = parser.parse_args()

    if args.multiprocessing:
        multiprocessing.set_start_method('spawn', force=True)

    active_methods = args.methods or METHODS

    print("KG Similarity — Full Dataset Evaluation")
    print(f"Methods : {', '.join(active_methods)}")
    if args.limit:
        print(f"Row limit: {args.limit}")
    print()


    if args.input:
        output_path = args.output or os.path.join(
            args.results_dir,
            os.path.splitext(os.path.basename(args.input))[0] + '_results.csv',
        )
        print(f"Processing: {args.input}")
        process_dataset(
            args.input, output_path,
            limit=args.limit, active_methods=active_methods,
            use_multiprocessing=args.multiprocessing,
            workers=args.workers, batch_size=args.batch_size,
        )
        return


    datasets = discover_datasets(args.data_dir)
    if not datasets:
        print(f"No CSV files found under: {args.data_dir}")
        return

    print(f"Found {len(datasets)} dataset(s) in {args.data_dir}\n")

    global_errors = {}

    for dataset_path in datasets:
        base_name   = os.path.splitext(os.path.basename(dataset_path))[0] + '_results.csv'
        output_path = os.path.join(args.results_dir, base_name)

        print(f"\n{'='*65}")
        print(f"Dataset : {os.path.basename(dataset_path)}")
        print(f"{'='*65}")

        try:
            _, errs = process_dataset(
                dataset_path, output_path,
                limit=args.limit, active_methods=active_methods,
                use_multiprocessing=args.multiprocessing,
                workers=args.workers, batch_size=args.batch_size,
            )
            global_errors[os.path.basename(dataset_path)] = errs
        except Exception as e:
            print(f"  FATAL: {e}")
            traceback.print_exc()


    if global_errors:
        pad = max(len(k) for k in global_errors)
        print(f"\n{'='*65}")
        print("Error summary (errors per method per dataset)")
        print(f"{'='*65}")
        print(f"{'Dataset':<{pad}}", end='')
        for m in active_methods:
            print(f"  {m[:13]:<13}", end='')
        print()
        for ds, errs in global_errors.items():
            print(f"{ds:<{pad}}", end='')
            for m in active_methods:
                print(f"  {str(errs.get(m, '-')):<13}", end='')
            print()


if __name__ == '__main__':
    main()
