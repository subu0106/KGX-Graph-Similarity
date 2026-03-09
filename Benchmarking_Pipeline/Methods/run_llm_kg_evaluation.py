#!/usr/bin/env python3
"""
LLM KG Evaluation — SNEA-SBERT  (multiprocessing, 6 workers)
Computes two similarity scores per row:
  - gold_similarity    : how well kg_llm matches kg_gold    (kg_llm drives filtering)
  - context_similarity : how well kg_llm matches kg_context (kg_llm drives filtering)
"""

import csv
import ast
import sys
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Methods import calculate_snea_sbert_similarity_score

NUM_WORKERS = 6


# worker (runs in a subprocess)

def _score_row(task):
    """
    task : dict with keys idx, total, id, question, gold_answer, llm_answer,
                           kg_gold_str, kg_llm_str, kg_context_str
    Returns the result dict ready to write to CSV.
    """
    idx         = task['idx']
    total       = task['total']
    row_id      = task['id']
    kg_gold_str = task['kg_gold_str']
    kg_llm_str  = task['kg_llm_str']
    kg_ctx_str  = task['kg_context_str']

    def parse(s):
        try:
            triples = ast.literal_eval(s) if s and s.strip() else []
        except Exception:
            triples = []
        return [t for t in triples if isinstance(t, list) and len(t) == 3]

    kg_llm     = parse(kg_llm_str)
    kg_gold    = parse(kg_gold_str)
    kg_context = parse(kg_ctx_str)

    gold_sim = 0.0
    if kg_llm and kg_gold:
        try:
            gold_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_gold)
        except Exception as e:
            print(f"  [{idx}/{total}] ERROR gold_similarity id={row_id}: {e}", flush=True)

    context_sim = 0.0
    if kg_llm and kg_context:
        try:
            context_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_context)
        except Exception as e:
            print(f"  [{idx}/{total}] ERROR context_similarity id={row_id}: {e}", flush=True)

    print(f"[{idx}/{total}] id={row_id}  gold={gold_sim:.4f}  context={context_sim:.4f}", flush=True)

    return {
        'idx':                idx,
        'id':                 row_id,
        'question':           task['question'],
        'gold_answer':        task['gold_answer'],
        'llm_answer':         task['llm_answer'],
        'kg_gold':            kg_gold_str,
        'kg_llm':             kg_llm_str,
        'kg_context':         kg_ctx_str,
        'gold_similarity':    gold_sim,
        'context_similarity': context_sim,
    }


# main processing

def process(input_file, output_file, limit=None):

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    print(f"Input  : {input_file}")
    print(f"Rows   : {total}")
    print(f"Workers: {NUM_WORKERS}\n")

    # build task list
    tasks = [
        {
            'idx':          i + 1,
            'total':        total,
            'id':           row.get('id', i + 1),
            'question':     row.get('question', ''),
            'gold_answer':  row.get('gold_answer', ''),
            'llm_answer':   row.get('llm_answer', ''),
            'kg_gold_str':  row.get('kg_gold', ''),
            'kg_llm_str':   row.get('kg_llm', ''),
            'kg_context_str': row.get('kg_context', ''),
        }
        for i, row in enumerate(rows)
    ]

    fieldnames = ['id', 'question', 'gold_answer', 'llm_answer',
                  'kg_gold', 'kg_llm', 'kg_context',
                  'gold_similarity', 'context_similarity']

    # run workers, collect results keyed by original idx to preserve order
    results_map = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_score_row, t): t['idx'] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results_map[result['idx']] = result

    # write in original order
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, total + 1):
            r = results_map[i]
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nOutput: {output_file}")

    results = list(results_map.values())
    valid_gold    = [r['gold_similarity']    for r in results if r['gold_similarity']    > 0]
    valid_context = [r['context_similarity'] for r in results if r['context_similarity'] > 0]

    if valid_gold:
        print(f"\ngold_similarity    — avg={np.mean(valid_gold):.4f}  "
              f"max={np.max(valid_gold):.4f}  min={np.min(valid_gold):.4f}  "
              f"valid={len(valid_gold)}/{total}")
    if valid_context:
        print(f"context_similarity — avg={np.mean(valid_context):.4f}  "
              f"max={np.max(valid_context):.4f}  min={np.min(valid_context):.4f}  "
              f"valid={len(valid_context)}/{total}")

    return results


# entry point

if __name__ == '__main__':
    import argparse

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
    parser.add_argument('--limit',  type=int, default=None,
                        help='Limit rows for testing')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of parallel workers (default: {NUM_WORKERS})')
    args = parser.parse_args()

    NUM_WORKERS = args.workers

    if args.output is None:
        os.makedirs(results_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(results_dir, f"{stem}_scored.csv")

    process(args.input, args.output, limit=args.limit)
