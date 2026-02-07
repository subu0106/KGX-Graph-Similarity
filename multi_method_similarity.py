#!/usr/bin/env python3
"""
Multi-method graph similarity comparison
Compares gemini answer pairs using:
1. KEA (Graph Kernel - Weisfeiler-Lehman + Clustering)
2. KEA Composite (Gaussian + WL Kernel)
3. KEA Semantic (Gaussian kernel only)
4. TransE graph embedding
5. RotatE graph embedding
6. Pure WL kernel (structural only)
7. GESML (GAP + top-k pooling + ReLU)
8. AA-KEA (Attention-Augmented KEA) - NEW!
9. AA-KEA-Neighborhood (with neighborhood context) - NEW!
"""

import csv
import ast
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from KEA import calculate_similarity, calculate_composite_similarity  # Import KEA methods
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import gc  # For garbage collection

# Add GAP directory to path for GESML import
sys.path.append(os.path.join(os.path.dirname(__file__), 'GAP'))
from gesml_node_similarity import GESMLNodeSimilarity  # Node-level comparison (no training)

# Import Attention-Augmented KEA
from Attention_Augmented_KEA.attention_augmented_kea import (
    calculate_aa_kea_similarity
    # calculate_aa_kea_neighborhood_similarity
)

# TransE Model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=1.0):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # TransE scoring: ||h + r - t||
        score = torch.norm(h + r - t, p=2, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()


# RotatE Model
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=6.0):
        super(RotatE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)  # real + imaginary
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # Split into real and imaginary parts
        h_re, h_im = torch.chunk(h, 2, dim=1)
        t_re, t_im = torch.chunk(t, 2, dim=1)

        # Relation as rotation in complex space
        r_re = torch.cos(r)
        r_im = torch.sin(r)

        # Hadamard product for rotation: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Distance in complex space
        score = torch.sqrt(torch.pow(hr_re - t_re, 2) + torch.pow(hr_im - t_im, 2)).sum(dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()


# Pure WL Kernel (without semantic clustering)
def calculate_pure_wl_kernel_similarity(triples1, triples2):
    """
    Calculate pure WL kernel similarity without semantic clustering
    Only considers graph structure
    """
    def create_networkx_graph(triple_list):
        G = nx.Graph()
        for triple in triple_list:
            if len(triple) == 3:
                subject, predicate, obj = triple
                G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
                G.nodes[subject.lower()]['label'] = subject.lower()
                G.nodes[obj.lower()]['label'] = obj.lower()
        return G

    def convert_to_grakel_graph(nx_graph):
        node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
        edge_labels = {(u, v): data.get('relation', 'default_relation') for u, v, data in nx_graph.edges(data=True)}
        edges = {(u, v): 1 for u, v in nx_graph.edges()}
        return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

    try:
        # Create graphs WITHOUT semantic clustering
        kg1_graph = create_networkx_graph(triples1)
        kg2_graph = create_networkx_graph(triples2)

        # Convert to GraKel format
        kg1_grakel = convert_to_grakel_graph(kg1_graph)
        kg2_grakel = convert_to_grakel_graph(kg2_graph)

        # Apply pure WL kernel
        wl_kernel = WeisfeilerLehman(n_jobs=2, normalize=True)
        kernel_matrix = wl_kernel.fit_transform([kg1_grakel, kg2_grakel])

        similarity = kernel_matrix[0, 1]
        return float(similarity)
    except Exception as e:
        print(f"Error in pure WL kernel: {e}")
        return None


# Graph Embedding Similarity Calculator
class GraphEmbeddingSimilarity:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

    def build_vocabulary(self, triples1, triples2):
        """Build entity and relation vocabularies from both graphs"""
        entities = set()
        relations = set()

        for triple in triples1 + triples2:
            if len(triple) == 3:
                entities.add(triple[0].lower())
                entities.add(triple[2].lower())
                relations.add(triple[1].lower())

        # Create mappings
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.id2relation = {i: r for r, i in self.relation2id.items()}

        return len(entities), len(relations)

    def triples_to_indices(self, triples):
        """Convert triples to index format"""
        indexed = []
        for triple in triples:
            if len(triple) == 3:
                h = self.entity2id.get(triple[0].lower(), 0)
                r = self.relation2id.get(triple[1].lower(), 0)
                t = self.entity2id.get(triple[2].lower(), 0)
                indexed.append((h, r, t))
        return indexed

    def train_model(self, model, triples, epochs=100, lr=0.01):
        """Train graph embedding model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for h, r, t in triples:
                heads = torch.LongTensor([h])
                relations = torch.LongTensor([r])
                tails = torch.LongTensor([t])

                # Positive score
                pos_score = model(heads, relations, tails)

                # Negative sampling
                neg_h = np.random.randint(0, len(self.entity2id))
                neg_t = np.random.randint(0, len(self.entity2id))

                neg_heads = torch.LongTensor([neg_h])
                neg_tails = torch.LongTensor([neg_t])

                neg_score = model(neg_heads, relations, neg_tails)

                # Margin ranking loss
                loss = torch.clamp(model.margin + pos_score - neg_score, min=0).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def get_graph_embedding(self, model, triples):
        """Get graph-level embedding by averaging entity embeddings"""
        entity_embeddings = model.get_embeddings()

        # Get unique entities in this graph
        entities = set()
        for triple in triples:
            entities.add(triple[0])
            entities.add(triple[2])

        # Average embeddings
        embeddings = []
        for entity in entities:
            entity_id = self.entity2id.get(entity.lower(), 0)
            embeddings.append(entity_embeddings[entity_id])

        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim * 2 if isinstance(model, RotatE) else self.embedding_dim)

        return np.mean(embeddings, axis=0)

    def calculate_transe_similarity(self, triples1, triples2):
        """Calculate similarity using TransE embeddings"""
        num_entities, num_relations = self.build_vocabulary(triples1, triples2)

        if num_entities == 0 or num_relations == 0:
            return None

        # Convert to indices
        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        # Train TransE model
        model = TransE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        # Get graph embeddings
        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        # Cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def calculate_rotate_similarity(self, triples1, triples2):
        """Calculate similarity using RotatE embeddings"""
        num_entities, num_relations = self.build_vocabulary(triples1, triples2)

        if num_entities == 0 or num_relations == 0:
            return None

        # Convert to indices
        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        # Train RotatE model
        model = RotatE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        # Get graph embeddings
        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        # Cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)


# Main Processing Function
# def parse_triple(triple_str):
#     """Parse string representation of triple"""
#     try:
#         parsed = ast.literal_eval(triple_str)
#         if isinstance(parsed, list) and len(parsed) == 3:
#             return [parsed]
#         return []
#     except:
#         return []

def parse_triple(triple_str):
    """Parse string representation of triple(s)"""
    try:
        parsed = ast.literal_eval(triple_str)
        
        # Handle both formats:
        # Format 1: [['A', 'B', 'C'], ['D', 'E', 'F']] -> return as is
        # Format 2: ['A', 'B', 'C'] -> wrap to [['A', 'B', 'C']]
        
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                return parsed  # Already nested
            elif len(parsed) == 3:
                return [parsed]  # Wrap single triple
        return []
    except Exception as e:
        print(f"ERROR parsing: {e}")
        return []


def process_dataset(input_file, output_file):
    """Process dataset and calculate similarities using all methods

    Memory-optimized version: writes results incrementally and cleans up after each row
    """
    results = []
    fieldnames = ['question', 'gemini_answer_1', 'gemini_answer_2',
                #  'kea_similarity', 'kea_composite', 'kea_structural', 'kea_semantic',
                #  'transe_similarity', 'rotate_similarity', 'wl_kernel_similarity', 'gesml_similarity',
                 'aa_kea_similarity'
                #  
                ]

    # Read CSV file using csv.DictReader (now properly quoted)
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)

    # Open output file for incremental writing
    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            # Simple progress indicator
            print(f"Processing row {i+1}/{total_rows}...", end=" ", flush=True)

            question = row['question']
            answer1_str = row['gold_kg']
            answer2_str = row['llm_kg']

            # Parse triples
            triples1 = parse_triple(answer1_str)
            triples2 = parse_triple(answer2_str)

            result = {
                'question': question,
                'gemini_answer_1': answer1_str,
                'gemini_answer_2': answer2_str
            }

            if not triples1 or not triples2:
                print("skipped (empty triples)")
                result.update({
                    # 'kea_similarity': None,
                    # 'kea_composite': None,
                    # 'kea_structural': None,
                    # 'kea_semantic': None,
                    # 'transe_similarity': None,
                    # 'rotate_similarity': None,
                    # 'wl_kernel_similarity': None,
                    # 'gesml_similarity': None,
                    'aa_kea_similarity': None,
                    # 'aa_kea_neighborhood_similarity': None
                })
            else:
                # Create fresh calculators for each row to prevent memory accumulation
                embedding_calculator = GraphEmbeddingSimilarity(embedding_dim=50)
                gesml_calculator = GESMLNodeSimilarity(embedding_dim=50, top_k=5)

                # # 1. KEA method
                # try:
                #     kea_sim, _, _ = calculate_similarity(triples1, triples2)
                #     result['kea_similarity'] = kea_sim
                # except Exception:
                #     result['kea_similarity'] = None

                # # 1b. KEA Composite
                # try:
                #     composite_result = calculate_composite_similarity(triples1, triples2, alpha=0.1, sigma=1.0)
                #     result['kea_composite'] = composite_result['composite']
                #     result['kea_structural'] = composite_result['structural']
                #     result['kea_semantic'] = composite_result['semantic']
                # except Exception:
                #     result['kea_composite'] = None
                #     result['kea_structural'] = None
                #     result['kea_semantic'] = None

                # # 2. TransE method
                # try:
                #     result['transe_similarity'] = embedding_calculator.calculate_transe_similarity(triples1, triples2)
                # except Exception:
                #     result['transe_similarity'] = None

                # # 3. RotatE method
                # try:
                #     result['rotate_similarity'] = embedding_calculator.calculate_rotate_similarity(triples1, triples2)
                # except Exception:
                #     result['rotate_similarity'] = None

                # # 4. Pure WL Kernel method
                # try:
                #     result['wl_kernel_similarity'] = calculate_pure_wl_kernel_similarity(triples1, triples2)
                # except Exception:
                #     result['wl_kernel_similarity'] = None

                # # 5. GESML method
                # try:
                #     result['gesml_similarity'] = gesml_calculator.calculate_node_similarity(triples1, triples2)
                # except Exception:
                #     result['gesml_similarity'] = None

                # 6. AA-KEA (Attention-Augmented KEA) - replaces clustering with attention
                try:
                    result['aa_kea_similarity'] = calculate_aa_kea_similarity(triples1, triples2)
                except Exception:
                    result['aa_kea_similarity'] = None

                # # 7. AA-KEA with Neighborhood context
                # try:
                #     result['aa_kea_neighborhood_similarity'] = calculate_aa_kea_neighborhood_similarity(triples1, triples2)
                # except Exception:
                #     result['aa_kea_neighborhood_similarity'] = None

                # Clean up calculators
                del embedding_calculator
                del gesml_calculator

            # Write result immediately (don't accumulate in memory)
            writer.writerow(result)
            out_f.flush()  # Ensure it's written to disk
            results.append(result)

            # Force garbage collection every row to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("done")

    print(f"\n{'='*60}")
    print(f"Results written to: {output_file}")
    print(f"Total rows processed: {len(results)}")
    print(f"{'='*60}")

    return results


# Plotting Function
def plot_individual_method_results(results, output_file_base):
    """
    Create individual plots for each similarity method
    """
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(output_file_base), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df['Question_ID'] = range(1, len(df) + 1)

    # Filter valid data
    plot_df = df.dropna(subset=['aa_kea_similarity'])

    methods = {
        # 'KEA (Semantic Clustering + WL Kernel)': ('kea_similarity', '#2E86AB'),
        # 'KEA Composite (Gaussian + WL)': ('kea_composite', '#1E5F8C'),
        # 'KEA Semantic (Gaussian Only)': ('kea_semantic', '#5BA3D0'),
        # 'TransE (Translation Embedding)': ('transe_similarity', '#A23B72'),
        # 'RotatE (Rotation Embedding)': ('rotate_similarity', '#F18F01'),
        # 'Pure WL Kernel (Structural Only)': ('wl_kernel_similarity', '#6A994E'),
        # 'GESML (GAP + Top-k + ReLU)': ('gesml_similarity', '#E63946'),
        'AA-KEA (Attention + WL)': ('aa_kea_similarity', '#9B59B6')  # Purple
        # 'AA-KEA-Neighbor (Context + WL)': ('aa_kea_neighborhood_similarity', '#1ABC9C')  # Teal
    }

    print("\n" + "="*60)
    print("Generating individual plots for each method...")
    print(f"Saving plots to: {plots_dir}")
    print("="*60)

    # Create individual plot for each method
    for method_name, (col_name, color) in methods.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Similarity scores line plot
        scores = plot_df[col_name] * 100
        ax1.plot(plot_df['Question_ID'], scores,
                marker='o', linestyle='-', linewidth=2,
                markersize=6, color=color, alpha=0.8)
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect (100%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='No Match (0%)')

        ax1.set_title(f'{method_name}\nSimilarity Scores', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Similarity (%)', fontsize=10)
        ax1.set_xlabel('Question ID', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='best', fontsize=8)

        # Set y-axis limits based on data
        ymin = max(-10, scores.min() - 10)
        ymax = min(110, scores.max() + 10)
        ax1.set_ylim(ymin, ymax)

        # Plot 2: Distribution histogram
        ax2.hist(scores, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=scores.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {scores.mean():.1f}%')
        ax2.axvline(x=scores.median(), color='blue', linestyle='--',
                   linewidth=2, label=f'Median: {scores.median():.1f}%')

        ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Similarity (%)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Save plot in plots directory
        base_name = os.path.basename(output_file_base)
        filename = os.path.join(plots_dir, f"{base_name}_{col_name}_plot.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {os.path.basename(filename)}")
        plt.close()

    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(16, 10))

    
    # # Plot 1: All methods comparison
    # ax1 = plt.subplot(2, 2, 1)
    # for method_name, (col_name, color) in methods.items():
    #     scores = plot_df[col_name] * 100
    #     ax1.plot(plot_df['Question_ID'], scores,
    #             marker='o', linestyle='-', linewidth=1.5,
    #             markersize=4, color=color, label=method_name.split('(')[0].strip(),
    #             alpha=0.7)

    # ax1.set_title('All Methods Comparison', fontsize=12, fontweight='bold')
    # ax1.set_ylabel('Similarity (%)', fontsize=10)
    # ax1.set_xlabel('Question ID', fontsize=10)
    # ax1.legend(loc='best', fontsize=8)
    # ax1.grid(True, linestyle='--', alpha=0.3)

    # # Plot 2: Average scores comparison
    # ax2 = plt.subplot(2, 2, 2)
    # method_labels = [name.split('(')[0].strip() for name in methods.keys()]
    # averages = [plot_df[col] * 100 for _, (col, _) in methods.items()]
    # colors_list = [color for _, (_, color) in methods.items()]

    # bp = ax2.boxplot(averages, tick_labels=method_labels, patch_artist=True)
    # for patch, color in zip(bp['boxes'], colors_list):
    #     patch.set_facecolor(color)
    #     patch.set_alpha(0.6)

    # ax2.set_title('Score Distributions', fontsize=12, fontweight='bold')
    # ax2.set_ylabel('Similarity (%)', fontsize=10)
    # ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    # # Plot 3: Bar chart of averages
    # ax3 = plt.subplot(2, 2, 3)
    # avg_values = [plot_df[col].mean() * 100 for _, (col, _) in methods.items()]
    # bars = ax3.bar(method_labels, avg_values, color=colors_list, alpha=0.7, edgecolor='black')

    # for bar, avg in zip(bars, avg_values):
    #     height = bar.get_height()
    #     ax3.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{avg:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ax3.set_title('Average Similarity by Method', fontsize=12, fontweight='bold')
    # ax3.set_ylabel('Average Similarity (%)', fontsize=10)
    # ax3.set_ylim(0, 100)
    # ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    # plt.xticks(rotation=15, ha='right')

    # # Plot 4: KEA vs Pure WL
    # ax4 = plt.subplot(2, 2, 4)
    # kea_scores = plot_df['kea_similarity'] * 100
    # wl_scores = plot_df['wl_kernel_similarity'] * 100

    # ax4.plot(plot_df['Question_ID'], kea_scores,
    #         marker='o', linestyle='-', linewidth=2, markersize=5,
    #         color='#2E86AB', label='KEA (with semantics)', alpha=0.8)
    # ax4.plot(plot_df['Question_ID'], wl_scores,
    #         marker='d', linestyle='--', linewidth=2, markersize=5,
    #         color='#6A994E', label='Pure WL (structural)', alpha=0.8)

    # ax4.set_title('KEA vs Pure WL: Semantic Impact', fontsize=12, fontweight='bold')
    # ax4.set_ylabel('Similarity (%)', fontsize=10)
    # ax4.set_xlabel('Question ID', fontsize=10)
    # ax4.legend(loc='best', fontsize=9)
    # ax4.grid(True, linestyle='--', alpha=0.3)

    # plt.tight_layout()
    # base_name = os.path.basename(output_file_base)
    # comparison_file = os.path.join(plots_dir, f"{base_name}_comparison.png")
    # plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    # print(f"  ✓ Saved: {os.path.basename(comparison_file)}")
    # plt.close()

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for method_name, (col_name, _) in methods.items():
        scores = plot_df[col_name] * 100
        perfect = (plot_df[col_name] >= 0.99).sum()
        high = (plot_df[col_name] >= 0.8).sum()

        print(f"\n{method_name}:")
        print(f"  Average:  {scores.mean():.2f}%")
        print(f"  Median:   {scores.median():.2f}%")
        print(f"  Std Dev:  {scores.std():.2f}%")
        print(f"  Max:      {scores.max():.2f}%")
        print(f"  Min:      {scores.min():.2f}%")
        print(f"  Perfect (≈1.0): {perfect}/{len(plot_df)} ({perfect/len(plot_df)*100:.1f}%)")
        print(f"  High (≥0.8):    {high}/{len(plot_df)} ({high/len(plot_df)*100:.1f}%)")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Method Graph Similarity Comparison')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing). Default: process all rows')
    parser.add_argument('--input', type=str,
                        default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/final_dataset_opensrc_models/pubmed/pubmedqa_mistralai_Mistral-7B-Instruct-v0.2.csv",
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default="/Users/subu/Desktop/FYP/KGX-Graph-Similarity/results/pubmedqa_mistralai/multi_method_similarity_results.csv",
                        help='Output CSV file path')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots (saves memory)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    input_file = args.input
    output_file = args.output

    print("="*60)
    print("Multi-Method Graph Similarity Comparison")
    print("="*60)

    if args.limit:
        print(f"\n*** LIMITING TO {args.limit} ROWS (for testing) ***\n")

    # print("\nMethods:")
    # print("1. KEA (Semantic Clustering + WL Kernel)")
    # print("   - Uses SBERT + Agglomerative Clustering for semantic grouping")
    # print("   - Then applies Weisfeiler-Lehman kernel")
    # print("2. KEA Composite (Gaussian + WL Kernel)")
    # print("   - Combines structural (WL) + semantic (Gaussian) similarity")
    # print("   - Formula: alpha * structural + (1-alpha) * semantic")
    # print("3. KEA Semantic (Gaussian kernel on SBERT)")
    # print("   - Direct semantic similarity without clustering")
    # print("4. TransE (Translation-based embedding)")
    # print("   - h + r ≈ t in embedding space")
    # print("5. RotatE (Rotation-based embedding)")
    # print("   - Relations as rotations in complex space")
    # print("6. Pure WL Kernel (Structural similarity only)")
    # print("   - Raw Weisfeiler-Lehman kernel without semantic preprocessing")
    # print("7. GESML (Node-Level, No Training)")
    # print("   - Context-sensitive node embeddings with attention")
    # print("   - Top-k pooling for structural patterns")
    print("8. AA-KEA (Attention-Augmented KEA) - NEW!")
    print("   - Replaces semantic clustering with attention-based alignment")
    print("   - Uses softmax attention over SBERT embeddings")
    print("   - Eliminates dependency on clustering threshold (0.35)")
    # print("9. AA-KEA-Neighborhood (with context) - NEW!")
    # print("   - Enhanced version with neighborhood-aware embeddings")
    # print("   - Combines node + neighbor embeddings for alignment")
    print()

    # Read and optionally limit rows
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if args.limit:
        all_rows = all_rows[:args.limit]
        print(f"Processing {len(all_rows)} rows (limited from original dataset)\n")

    # Write limited rows to temp file if needed
    if args.limit:
        import tempfile
        temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        temp_writer = csv.DictWriter(temp_input, fieldnames=['question', 'Graph1', 'Graph2'])
        temp_writer.writeheader()
        temp_writer.writerows(all_rows)
        temp_input.close()
        input_file = temp_input.name

    # Process dataset
    results = process_dataset(input_file, output_file)

    # Clean up temp file
    if args.limit:
        os.remove(input_file)

    # Generate plots (optional - can be skipped to save memory)
    if not args.skip_plots:
        plot_base = output_file.replace('.csv', '')
        plot_individual_method_results(results, plot_base)
    else:
        print("\n[Skipping plot generation as requested]")
