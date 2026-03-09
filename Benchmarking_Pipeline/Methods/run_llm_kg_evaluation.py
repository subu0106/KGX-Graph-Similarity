#!/usr/bin/env python3
"""
LLM KG Evaluation — SNEA-SBERT
Computes two similarity scores per row:
  - gold_similarity    : how well kg_llm matches kg_gold    (kg_llm drives filtering)
  - context_similarity : how well kg_llm matches kg_context (kg_llm drives filtering)

Parallel processing notes:
  - Uses ProcessPoolExecutor with spawn start method (required for CUDA on Linux)
  - Each worker loads SBERT once into its own GPU context, then handles its batch
  - Methods are imported lazily inside the worker to avoid fork+CUDA deadlocks
"""

import csv
import ast
import os
import gc
import math
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIELDNAMES       = ['id', 'question', 'gold_answer', 'llm_answer',
                    'kg_gold', 'kg_llm', 'kg_context',
                    'gold_similarity', 'context_similarity']
DEFAULT_WORKERS    = 6
DEFAULT_BATCH_SIZE = 50


def _parse(s):
    try:
        triples = ast.literal_eval(s) if s and s.strip() else []
    except Exception:
        triples = []
    return [t for t in triples if isinstance(t, list) and len(t) == 3]


def _compute_scores(kg_llm, kg_gold, kg_context):
    """
    Lazy import so each spawned worker loads SBERT once into its own
    CUDA context — avoids fork+CUDA deadlocks.
    """
    from Methods import calculate_snea_sbert_similarity_score

    gold_sim = 0.0
    if kg_llm and kg_gold:
        try:
            gold_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_gold)
        except Exception as e:
            print(f"  ERROR gold_similarity: {e}", flush=True)

    context_sim = 0.0
    if kg_llm and kg_context:
        try:
            context_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_context)
        except Exception as e:
            print(f"  ERROR context_similarity: {e}", flush=True)

    return gold_sim, context_sim


def _process_batch(batch_data):
    """
    Worker: receives list of row dicts.
    SBERT loads on first _compute_scores() call, then stays cached for the batch.
    """
    results = []
    for row in batch_data:
        kg_llm     = _parse(row['kg_llm'])
        kg_gold    = _parse(row['kg_gold'])
        kg_context = _parse(row['kg_context'])

        gold_sim, context_sim = _compute_scores(kg_llm, kg_gold, kg_context)

        results.append({
            'id':                 row['id'],
            'question':           row['question'],
            'gold_answer':        row['gold_answer'],
            'llm_answer':         row['llm_answer'],
            'kg_gold':            row['kg_gold'],
            'kg_llm':             row['kg_llm'],
            'kg_context':         row['kg_context'],
            'gold_similarity':    gold_sim,
            'context_similarity': context_sim,
        })
        gc.collect()

    return results


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def process(input_file, output_file, limit=None,
            workers=DEFAULT_WORKERS, batch_size=DEFAULT_BATCH_SIZE,
            use_multiprocessing=False):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)

    # normalise rows to expected keys
    norm = []
    for i, r in enumerate(rows):
        norm.append({
            'id':           r.get('id', i + 1),
            'question':     r.get('question', ''),
            'gold_answer':  r.get('gold_answer', ''),
            'llm_answer':   r.get('llm_answer', ''),
            'kg_gold':      r.get('kg_gold', ''),
            'kg_llm':       r.get('kg_llm', ''),
            'kg_context':   r.get('kg_context', ''),
        })
    rows = norm

    if use_multiprocessing:
        n_batches = math.ceil(total / batch_size)
        print(f"Rows: {total}  |  Workers: {workers}  |  "
              f"Batch size: {batch_size}  |  Batches: {n_batches}  |  Mode: multiprocess")
    else:
        print(f"Rows: {total}  |  Mode: single-process")

    all_results = []

    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()

        if use_multiprocessing:
            batches   = list(_chunk(rows, batch_size))
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(_process_batch, batch): idx
                    for idx, batch in enumerate(batches)
                }
                ordered = [None] * n_batches
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        ordered[idx] = future.result()
                    except Exception as e:
                        print(f"\nBatch {idx} failed: {e}")
                        ordered[idx] = []
                    completed += 1
                    done_rows = sum(len(b) for b in ordered if b is not None)
                    print(f"  Batches done: {completed}/{n_batches}  "
                          f"({done_rows}/{total} rows)", flush=True)

            for batch_results in ordered:
                if batch_results:
                    for result in batch_results:
                        writer.writerow(result)
                        all_results.append(result)
                    out_f.flush()

        else:
            for i, row in enumerate(rows):
                kg_llm     = _parse(row['kg_llm'])
                kg_gold    = _parse(row['kg_gold'])
                kg_context = _parse(row['kg_context'])

                gold_sim, context_sim = _compute_scores(kg_llm, kg_gold, kg_context)

                result = {**row, 'gold_similarity': gold_sim, 'context_similarity': context_sim}
                writer.writerow(result)
                out_f.flush()
                all_results.append(result)
                gc.collect()

                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(f"  Processed {i + 1}/{total} rows  "
                          f"gold={gold_sim:.4f}  context={context_sim:.4f}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_gold    = [r['gold_similarity']    for r in all_results if r['gold_similarity']    > 0]
    valid_context = [r['context_similarity'] for r in all_results if r['context_similarity'] > 0]

    if valid_gold:
        print(f"\ngold_similarity    — avg={sum(valid_gold)/len(valid_gold):.4f}  "
              f"max={max(valid_gold):.4f}  min={min(valid_gold):.4f}  valid={len(valid_gold)}/{total}")
    if valid_context:
        print(f"context_similarity — avg={sum(valid_context)/len(valid_context):.4f}  "
              f"max={max(valid_context):.4f}  min={min(valid_context):.4f}  valid={len(valid_context)}/{total}")

    return all_results


if __name__ == '__main__':
    import argparse
    import multiprocessing

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(script_dir, '..', 'LLM_benchmarking_data')
    results_dir = os.path.join(script_dir, '..', 'Results', 'llm_evaluation')

    parser = argparse.ArgumentParser(description='LLM KG Evaluation — SNEA-SBERT')
    parser.add_argument('--input',  type=str,
                        default=os.path.join(data_dir,
                            'pubmedqa_mistralai_Mistral-7B-Instruct-v0.2_answers_KGs.csv'),
                        help='Input CSV with kg_gold, kg_llm, kg_context columns')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV (default: Results/llm_evaluation/<stem>_scored.csv)')
    parser.add_argument('--limit',       type=int,  default=None,
                        help='Limit rows for testing')
    parser.add_argument('--workers',     type=int,  default=DEFAULT_WORKERS,
                        help=f'Worker processes (default: {DEFAULT_WORKERS})')
    parser.add_argument('--batch-size',  type=int,  default=DEFAULT_BATCH_SIZE,
                        help=f'Rows per batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--multiprocessing', action='store_true', default=False,
                        help='Use parallel worker processes (requires CUDA; may fail on macOS CPU)')
    args = parser.parse_args()

    if args.multiprocessing:
        multiprocessing.set_start_method('spawn', force=True)

    if args.output is None:
        os.makedirs(results_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(results_dir, f"{stem}_scored.csv")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("LLM KG Evaluation — SNEA-SBERT")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Mode:    {'multiprocess' if args.multiprocessing else 'single-process'}")
    if args.multiprocessing:
        print(f"Workers: {args.workers}  |  Batch size: {args.batch_size}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    else:
        print("GPU:     not available, using CPU")
    print()

    process(
        args.input, args.output,
        limit=args.limit,
        workers=args.workers,
        batch_size=args.batch_size,
        use_multiprocessing=args.multiprocessing,
    )
