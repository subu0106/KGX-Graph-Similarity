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
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=1.0):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        score = torch.norm(h + r - t, p=2, dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()


# RotatE Model
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=50, margin=6.0):
        super(RotatE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        h_re, h_im = torch.chunk(h, 2, dim=1)
        t_re, t_im = torch.chunk(t, 2, dim=1)

        r_re = torch.cos(r)
        r_im = torch.sin(r)

        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        score = torch.sqrt(torch.pow(hr_re - t_re, 2) + torch.pow(hr_im - t_im, 2)).sum(dim=1)
        return score

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.cpu().numpy()


# Direct Sentence Transformer Similarity
def calculate_direct_sbert_similarity(triples1, triples2):
    """
    Calculate graph similarity using sentence transformer embeddings directly.
    Uses the same SBERT model as KEA (paraphrase-MPNet-base-v2).

    Pipeline:
    1. Convert each triple to text representation
    2. Get SBERT embeddings for all triples in both graphs
    3. Compute graph-level embeddings via mean pooling
    4. Calculate cosine similarity between graph embeddings

    Args:
        triples1: List of [subject, predicate, object] triples
        triples2: List of [subject, predicate, object] triples

    Returns:
        similarity: float between -1 and 1 (cosine similarity)
    """
    try:
        # Filter valid triples
        valid_triples1 = [t for t in triples1 if len(t) == 3]
        valid_triples2 = [t for t in triples2 if len(t) == 3]

        if not valid_triples1 or not valid_triples2:
            return None

        # Convert triples to text representations
        def triple_to_text(triple):
            """Convert a triple [subject, predicate, object] to text"""
            return f"{triple[0]} {triple[1]} {triple[2]}"

        texts1 = [triple_to_text(t) for t in valid_triples1]
        texts2 = [triple_to_text(t) for t in valid_triples2]

        # Get SBERT embeddings for all triples
        embeddings1 = sbert_model.encode(texts1, convert_to_tensor=False)
        embeddings2 = sbert_model.encode(texts2, convert_to_tensor=False)

        # Compute graph-level embeddings via mean pooling
        graph_emb1 = np.mean(embeddings1, axis=0)
        graph_emb2 = np.mean(embeddings2, axis=0)

        # Calculate cosine similarity
        similarity = cosine_similarity([graph_emb1], [graph_emb2])[0][0]

        return float(similarity)

    except Exception as e:
        print(f"Error in direct SBERT similarity: {e}")
        return None


# Pure WL Kernel
def calculate_pure_wl_kernel_similarity(triples1, triples2):
    """Calculate pure WL kernel similarity without semantic clustering"""
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
        kg1_graph = create_networkx_graph(triples1)
        kg2_graph = create_networkx_graph(triples2)

        kg1_grakel = convert_to_grakel_graph(kg1_graph)
        kg2_grakel = convert_to_grakel_graph(kg2_graph)

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

                pos_score = model(heads, relations, tails)

                neg_h = np.random.randint(0, len(self.entity2id))
                neg_t = np.random.randint(0, len(self.entity2id))

                neg_heads = torch.LongTensor([neg_h])
                neg_tails = torch.LongTensor([neg_t])

                neg_score = model(neg_heads, relations, neg_tails)

                loss = torch.clamp(model.margin + pos_score - neg_score, min=0).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def get_graph_embedding(self, model, triples):
        """Get graph-level embedding by averaging entity embeddings"""
        entity_embeddings = model.get_embeddings()

        entities = set()
        for triple in triples:
            entities.add(triple[0])
            entities.add(triple[2])

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

        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        model = TransE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def calculate_rotate_similarity(self, triples1, triples2):
        """Calculate similarity using RotatE embeddings"""
        num_entities, num_relations = self.build_vocabulary(triples1, triples2)

        if num_entities == 0 or num_relations == 0:
            return None

        indexed1 = self.triples_to_indices(triples1)
        indexed2 = self.triples_to_indices(triples2)

        if len(indexed1) == 0 or len(indexed2) == 0:
            return None

        model = RotatE(num_entities, num_relations, self.embedding_dim)
        all_triples = indexed1 + indexed2
        self.train_model(model, all_triples, epochs=100)

        emb1 = self.get_graph_embedding(model, triples1)
        emb2 = self.get_graph_embedding(model, triples2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)


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

    print(f"\n{'='*60}")
    print(f"Results written to: {output_file}")
    print(f"Total rows processed: {total_rows}")
    print(f"{'='*60}")


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

    print("="*60)
    print("Semantic KG Multi-Method Graph Similarity Comparison")
    print("="*60)
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

    print("\nMethods:")
    print("1. KEA (Semantic Clustering + WL Kernel)")
    print("2. KEA Composite (Gaussian + WL Kernel)")
    print("3. KEA Semantic (Gaussian kernel only)")
    print("4. TransE (Translation-based embedding)")
    print("5. RotatE (Rotation-based embedding)")
    print("6. Pure WL Kernel (Structural similarity only)")
    print("7. AA-KEA (Attention-Augmented KEA)")
    print("8. Direct SBERT (paraphrase-MPNet-base-v2)")
    print()

    # Process dataset
    process_semantic_kg_dataset(input_file, output_file)

    # Clean up temp file
    if args.limit:
        os.remove(input_file)

    print("\n✓ Processing complete!")
    print(f"Results saved to: {output_file}")
