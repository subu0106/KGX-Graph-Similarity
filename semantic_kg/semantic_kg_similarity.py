#!/usr/bin/env python3
"""
Multi-method graph similarity comparison for semantic_kg dataset
Compares graph pairs using:
1. KEA (Graph Kernel - Weisfeiler-Lehman + Clustering)
2. KEA Composite (Gaussian + WL Kernel)
3. KEA Semantic (Gaussian kernel only)
4. TransE graph embedding
5. RotatE graph embedding
6. Pure WL kernel (structural only)
7. AA-KEA (Attention-Augmented KEA)
8. Direct SBERT (sentence transformer embeddings with mean pooling)
"""

import csv
import ast
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
import os
import sys
import gc
from sentence_transformers import SentenceTransformer

# Import from Methods package
from Methods import (
    calculate_similarity,
    calculate_composite_similarity,
    calculate_aa_kea_similarity,
    calculate_pure_wl_kernel_similarity,
    GraphEmbeddingSimilarity
)

# Load the same SBERT model used in KEA for fair comparison
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2')


def parse_triple(triple_str):
    """Parse string representation of triple(s)"""
    try:
        parsed = ast.literal_eval(triple_str)

        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                return parsed
            elif len(parsed) == 3:
                return [parsed]
        return []
    except Exception as e:
        print(f"ERROR parsing: {e}")
        return []


def process_semantic_kg_dataset(input_file, output_file):
    """
    Process semantic_kg dataset and calculate similarities using all methods

    Input CSV format:
        graph_1, graph_2, similarity_score_ground

    Output CSV format:
        graph_1, graph_2, kea_similarity, kea_composite, kea_structural, kea_semantic,
        transe_similarity, rotate_similarity, wl_kernel_similarity, aa_kea_similarity,
        sbert_direct_similarity, similarity_score_ground
    """
    fieldnames = [
        'graph_1', 'graph_2',
        'kea_similarity', 'kea_composite', 'kea_structural', 'kea_semantic',
        'transe_similarity', 'rotate_similarity', 'wl_kernel_similarity',
        'aa_kea_similarity', 'sbert_direct_similarity',
        'similarity_score_ground'
    ]

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"Processing {total_rows} graph pairs...")

    # Open output file for incremental writing
    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{total_rows}...", end=" ", flush=True)

            graph1_str = row['graph_1']
            graph2_str = row['graph_2']
            ground_truth = row['similarity_score_ground']

            # Parse triples
            triples1 = parse_triple(graph1_str)
            triples2 = parse_triple(graph2_str)

            result = {
                'graph_1': graph1_str,
                'graph_2': graph2_str,
                'similarity_score_ground': ground_truth
            }

            if not triples1 or not triples2:
                print("skipped (empty triples)")
                result.update({
                    'kea_similarity': None,
                    'kea_composite': None,
                    'kea_structural': None,
                    'kea_semantic': None,
                    'transe_similarity': None,
                    'rotate_similarity': None,
                    'wl_kernel_similarity': None,
                    'aa_kea_similarity': None,
                    'sbert_direct_similarity': None
                })
            else:
                # Create fresh calculators for each row
                embedding_calculator = GraphEmbeddingSimilarity(embedding_dim=50)

                # 1. KEA method
                try:
                    kea_sim, _, _ = calculate_similarity(triples1, triples2)
                    result['kea_similarity'] = kea_sim
                except Exception as e:
                    print(f"KEA error: {e}", end=" ")
                    result['kea_similarity'] = None

                # 2. KEA Composite
                try:
                    composite_result = calculate_composite_similarity(triples1, triples2, alpha=0.1, sigma=1.0)
                    result['kea_composite'] = composite_result['composite']
                    result['kea_structural'] = composite_result['structural']
                    result['kea_semantic'] = composite_result['semantic']
                except Exception as e:
                    print(f"KEA Composite error: {e}", end=" ")
                    result['kea_composite'] = None
                    result['kea_structural'] = None
                    result['kea_semantic'] = None

                # 3. TransE method
                try:
                    result['transe_similarity'] = embedding_calculator.calculate_transe_similarity(triples1, triples2)
                except Exception as e:
                    print(f"TransE error: {e}", end=" ")
                    result['transe_similarity'] = None

                # 4. RotatE method
                try:
                    result['rotate_similarity'] = embedding_calculator.calculate_rotate_similarity(triples1, triples2)
                except Exception as e:
                    print(f"RotatE error: {e}", end=" ")
                    result['rotate_similarity'] = None

                # 5. Pure WL Kernel method
                try:
                    result['wl_kernel_similarity'] = calculate_pure_wl_kernel_similarity(triples1, triples2)
                except Exception as e:
                    print(f"WL Kernel error: {e}", end=" ")
                    result['wl_kernel_similarity'] = None

                # 6. AA-KEA (Attention-Augmented KEA)
                try:
                    result['aa_kea_similarity'] = calculate_aa_kea_similarity(triples1, triples2)
                except Exception as e:
                    print(f"AA-KEA error: {e}", end=" ")
                    result['aa_kea_similarity'] = None

                # 7. Direct SBERT Similarity (same model as KEA)
                try:
                    result['sbert_direct_similarity'] = calculate_direct_sbert_similarity(triples1, triples2)
                except Exception as e:
                    print(f"SBERT Direct error: {e}", end=" ")
                    result['sbert_direct_similarity'] = None

                # Clean up calculators
                del embedding_calculator

            # Write result immediately
            writer.writerow(result)
            out_f.flush()

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Semantic KG Multi-Method Similarity Comparison')
    parser.add_argument('--input', type=str,
                        default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/data/semantic_kg_transformed.csv",
                        help='Input CSV file path (semantic_kg_transformed.csv)')
    parser.add_argument('--output', type=str,
                        default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/results/semantic_kg/semantic_kg_similarity_results.csv",
                        help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    input_file = args.input
    output_file = args.output

    print("Semantic KG Multi-Method Graph Similarity Comparison")
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    if args.limit:
        print(f"\n*** LIMITING TO {args.limit} ROWS (for testing) ***\n")
        # Read and limit rows
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)[:args.limit]

        # Write to temp file
        import tempfile
        temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8')
        temp_writer = csv.DictWriter(temp_input, fieldnames=['graph_1', 'graph_2', 'similarity_score_ground'])
        temp_writer.writeheader()
        temp_writer.writerows(all_rows)
        temp_input.close()
        input_file = temp_input.name


    # Process dataset
    process_semantic_kg_dataset(input_file, output_file)

    # Clean up temp file
    if args.limit:
        os.remove(input_file)

    print("Processing complete.")
    print(f"Results saved to: {output_file}")
