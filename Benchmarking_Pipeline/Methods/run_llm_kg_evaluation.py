#!/usr/bin/env python3
"""
LLM KG Evaluation — SNEA-SBERT
Computes two similarity scores per row:
  - gold_similarity    : how well kg_llm matches kg_gold    (kg_llm drives filtering)
  - context_similarity : how well kg_llm matches kg_context (kg_llm drives filtering)
"""

import csv
import ast
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Methods import calculate_snea_sbert_similarity_score


def parse_kg(kg_str):
    try:
        triples = ast.literal_eval(kg_str) if kg_str and kg_str.strip() else []
    except Exception:
        triples = []
    return [t for t in triples if isinstance(t, list) and len(t) == 3]


def process(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Input : {input_file}")
    print(f"Rows  : {total}\n")

    fieldnames = ['id', 'question', 'gold_answer', 'llm_answer',
                  'kg_gold', 'kg_llm', 'kg_context',
                  'gold_similarity', 'context_similarity']

    results = []

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            row_id = row.get('id', idx)
            print(f"[{idx}/{total}] id={row_id}")

            kg_llm     = parse_kg(row.get('kg_llm', ''))
            kg_gold    = parse_kg(row.get('kg_gold', ''))
            kg_context = parse_kg(row.get('kg_context', ''))

            # kg_llm is the anchor — filtering happens from kg_gold / kg_context
            if kg_llm and kg_gold:
                try:
                    gold_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_gold)
                except Exception as e:
                    print(f"  ERROR gold_similarity: {e}")
                    gold_sim = 0.0
            else:
                gold_sim = 0.0

            if kg_llm and kg_context:
                try:
                    context_sim = calculate_snea_sbert_similarity_score(kg_llm, kg_context)
                except Exception as e:
                    print(f"  ERROR context_similarity: {e}")
                    context_sim = 0.0
            else:
                context_sim = 0.0

            print(f"  gold_similarity={gold_sim:.4f}  context_similarity={context_sim:.4f}")

            out = {
                'id':                 row_id,
                'question':           row.get('question', ''),
                'gold_answer':        row.get('gold_answer', ''),
                'llm_answer':         row.get('llm_answer', ''),
                'kg_gold':            row.get('kg_gold', ''),
                'kg_llm':             row.get('kg_llm', ''),
                'kg_context':         row.get('kg_context', ''),
                'gold_similarity':    gold_sim,
                'context_similarity': context_sim,
            }
            writer.writerow(out)
            f.flush()
            results.append(out)

    print(f"\nOutput: {output_file}")

    valid_gold    = [r['gold_similarity']    for r in results if r['gold_similarity']    > 0]
    valid_context = [r['context_similarity'] for r in results if r['context_similarity'] > 0]

    if valid_gold:
        print(f"\ngold_similarity    — avg={sum(valid_gold)/len(valid_gold):.4f}  "
              f"max={max(valid_gold):.4f}  min={min(valid_gold):.4f}  valid={len(valid_gold)}/{total}")
    if valid_context:
        print(f"context_similarity — avg={sum(valid_context)/len(valid_context):.4f}  "
              f"max={max(valid_context):.4f}  min={min(valid_context):.4f}  valid={len(valid_context)}/{total}")

    return results


if __name__ == '__main__':
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, '..', 'LLM_benchmarking_data')
    results_dir = os.path.join(script_dir, '..', 'Results', 'llm_evaluation')

    parser = argparse.ArgumentParser(description='LLM KG Evaluation — SNEA-SBERT')
    parser.add_argument('--input',  type=str,
                        default=os.path.join(data_dir, 'pubmedqa_mistralai_Mistral-7B-Instruct-v0.2_answers_KGs.csv'),
                        help='Input CSV with kg_gold, kg_llm, kg_context columns')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='Output CSV path (default: Results/llm_evaluation/<input_stem>_scored.csv)')
    parser.add_argument('--limit',  type=int, default=None,
                        help='Limit rows for testing')
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(results_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(results_dir, f"{stem}_scored.csv")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.limit:
        print(f"Limit: {args.limit} rows\n")
        import csv as _csv
        with open(args.input, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            rows   = list(reader)[:args.limit]
        tmp = args.input.replace('.csv', f'_limit{args.limit}.csv')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=reader.fieldnames)
            w.writeheader(); w.writerows(rows)
        args.input = tmp

    process(args.input, args.output)
