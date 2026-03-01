#!/usr/bin/env python3
"""
Run AA-KEA similarity on MRPC dataset (paraphrase pairs with KGs).
Compares kg_1 vs kg_2 for each paragraph pair.
"""

import csv
import ast
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Methods import calculate_aa_kea_similarity


def process_mrpc_dataset(input_file, output_file):
    """Process MRPC dataset and calculate AA-KEA similarity between kg_1 and kg_2."""

    results = []
    fieldnames = ['id', 'paragraph_1', 'paragraph_2', 'kg_1', 'kg_2', 'aa_kea_similarity']

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Total rows: {total_rows}\n")

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            row_id     = row['id']
            paragraph1 = row['paragraph_1']
            paragraph2 = row['paragraph_2']
            kg1_str    = row['kg_1']
            kg2_str    = row['kg_2']

            print(f"[{idx}/{total_rows}] Processing id={row_id}")

            try:
                kg1 = ast.literal_eval(kg1_str) if kg1_str.strip() else []
                kg2 = ast.literal_eval(kg2_str) if kg2_str.strip() else []
            except Exception as e:
                print(f"  ERROR parsing KGs: {e}")
                kg1, kg2 = [], []

            kg1 = [triple for triple in kg1 if isinstance(triple, list) and len(triple) == 3]
            kg2 = [triple for triple in kg2 if isinstance(triple, list) and len(triple) == 3]

            if kg1 and kg2:
                try:
                    aa_kea_sim = calculate_aa_kea_similarity(kg1, kg2)
                except Exception as e:
                    print(f"  ERROR in AA-KEA: {e}")
                    aa_kea_sim = 0.0
            else:
                aa_kea_sim = 0.0

            result_row = {
                'id':              row_id,
                'paragraph_1':     paragraph1,
                'paragraph_2':     paragraph2,
                'kg_1':            kg1_str,
                'kg_2':            kg2_str,
                'aa_kea_similarity': aa_kea_sim,
            }

            writer.writerow(result_row)
            outfile.flush()
            results.append(result_row)

    print(f"\nOutput saved to: {output_file}")

    similarities = [r['aa_kea_similarity'] for r in results if r['aa_kea_similarity'] > 0]

    if similarities:
        avg     = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        min_sim = min(similarities)
        print(f"Average: {avg:.4f}  Max: {max_sim:.4f}  Min: {min_sim:.4f}  Valid: {len(similarities)}/{len(results)}")

    return results


def generate_summary(results, output_dir):
    """Generate summary statistics and save to summary.txt."""

    valid_sims = [r['aa_kea_similarity'] for r in results if r['aa_kea_similarity'] > 0]

    if not valid_sims:
        print("No valid similarities to summarize")
        return

    import numpy as np

    avg     = np.mean(valid_sims)
    median  = np.median(valid_sims)
    std     = np.std(valid_sims)
    max_sim = np.max(valid_sims)
    min_sim = np.min(valid_sims)
    perfect = sum(1 for s in valid_sims if s >= 0.95)
    high    = sum(1 for s in valid_sims if s >= 0.8)

    summary_file = os.path.join(output_dir, "mrpc_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("MRPC AA-KEA SUMMARY STATISTICS\n\n")
        f.write(f"Total valid pairs: {len(valid_sims)}/{len(results)}\n\n")
        f.write(f"Average:           {avg*100:.2f}%\n")
        f.write(f"Median:            {median*100:.2f}%\n")
        f.write(f"Std Dev:           {std*100:.2f}%\n")
        f.write(f"Max:               {max_sim*100:.2f}%\n")
        f.write(f"Min:               {min_sim*100:.2f}%\n")
        f.write(f"Perfect (>=0.95):  {perfect}/{len(valid_sims)} ({perfect/len(valid_sims)*100:.1f}%)\n")
        f.write(f"High (>=0.80):     {high}/{len(valid_sims)} ({high/len(valid_sims)*100:.1f}%)\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='AA-KEA Similarity for wikipedia_entity_swap_400 Dataset')
    parser.add_argument('--input',  type=str,
                        default=os.path.join(script_dir, 'wikipedia_entity_swap_400_KGs.csv'),
                        help='Input CSV file')
    parser.add_argument('--output', type=str,
                        default=os.path.join(script_dir, 'part2_wikipedia_entity_swap_400_aa_kea_results.csv'),
                        help='Output CSV file')
    parser.add_argument('--limit',  type=int, default=None,
                        help='Limit number of rows (for testing)')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit:  {args.limit} rows")
    print()

    results = process_mrpc_dataset(args.input, args.output)

    if args.limit:
        results = results[:args.limit]

    generate_summary(results, script_dir)
