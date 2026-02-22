#!/usr/bin/env python3
"""
Run AA-KEA similarity on mamta dataset (STS-B with KGs)
Compares kg1 vs kg2 for sentence pairs
"""

import csv
import ast
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Methods import calculate_aa_kea_similarity

def process_mamta_dataset(input_file, output_file):
    """Process mamta dataset and calculate AA-KEA similarity"""

    results = []
    fieldnames = ['row_id', 'sentence1', 'sentence2', 'kg1', 'kg2', 'aa_kea_similarity']

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Total rows: {total_rows}\n")

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            row_id = row['row_id']
            sentence1 = row['sentence1']
            sentence2 = row['sentence2']
            kg1_str = row['kg1']
            kg2_str = row['kg2']

            print(f"[{idx}/{total_rows}] Processing row_id={row_id}")

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
                'row_id': row_id,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'kg1': kg1_str,
                'kg2': kg2_str,
                'aa_kea_similarity': aa_kea_sim
            }

            writer.writerow(result_row)
            outfile.flush()
            results.append(result_row)

    print(f"Output saved to: {output_file}")

    similarities = [r['aa_kea_similarity'] for r in results if r['aa_kea_similarity'] > 0]

    if similarities:
        avg = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        min_sim = min(similarities)

        print(f"Average: {avg:.4f}  Max: {max_sim:.4f}  Min: {min_sim:.4f}  Valid: {len(similarities)}/{len(results)}")

    return results


def generate_summary(results, output_dir):
    """Generate summary statistics"""

    similarities = [r['aa_kea_similarity'] for r in results]
    valid_sims = [s for s in similarities if s > 0]

    if not valid_sims:
        print("No valid similarities to summarize")
        return

    import numpy as np

    avg = np.mean(valid_sims)
    median = np.median(valid_sims)
    std = np.std(valid_sims)
    max_sim = np.max(valid_sims)
    min_sim = np.min(valid_sims)

    # Count high scores
    perfect = sum(1 for s in valid_sims if s >= 0.95)
    high = sum(1 for s in valid_sims if s >= 0.8)

    summary_file = os.path.join(output_dir, "summary.txt")

    with open(summary_file, 'w') as f:
        f.write("SUMMARY STATISTICS\n\n")

        f.write(f"AA-KEA (Attention + WL):\n")
        f.write(f"  Average:  {avg*100:.2f}%\n")
        f.write(f"  Median:   {median*100:.2f}%\n")
        f.write(f"  Std Dev:  {std*100:.2f}%\n")
        f.write(f"  Max:      {max_sim*100:.2f}%\n")
        f.write(f"  Min:      {min_sim*100:.2f}%\n")
        f.write(f"  Perfect (≈1.0): {perfect}/{len(valid_sims)} ({perfect/len(valid_sims)*100:.1f}%)\n")
        f.write(f"  High (≥0.8):    {high}/{len(valid_sims)} ({high/len(valid_sims)*100:.1f}%)\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AA-KEA Similarity for Mamta Dataset')
    parser.add_argument('--input', type=str,
                       default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/data/edc_data/mistralai_Mistral-7B-Instruct-v0.2_stsb2_kgv.csv",
                       help='Input CSV file (default: merged file)')
    parser.add_argument('--output', type=str,
                       default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/results/edc_data/stsb2_aa_kea_results.csv",
                       help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of rows (for testing)')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, args.input)
    output_file = os.path.join(script_dir, args.output)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else script_dir, exist_ok=True)

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    if args.limit:
        print(f"Limit:  {args.limit} rows")

    results = process_mamta_dataset(input_file, output_file)

    if args.limit and len(results) > args.limit:
        results = results[:args.limit]

    generate_summary(results, script_dir)
