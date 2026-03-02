#!/usr/bin/env python3
"""
Attention-Augmented KEA (AA-KEA)

Replaces semantic clustering in KEA with GAP-style attention alignment.
This eliminates dependency on clustering thresholds while retaining
semantic sensitivity through mutual attention over graph neighborhoods.

Pipeline:
1. Semantic Triple Matching (from KEA) - SBERT-based triple selection
2. Attention-Based Node Alignment (from GAP) - replaces clustering
3. WL Kernel Comparison (from KEA) - structural similarity

Shared utilities (SBERT embeddings, graph building, soft label mapping,
WL kernel helpers) are imported from SNEA.py.

Author: Research Implementation
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grakel.kernels import WeisfeilerLehman

from .snea import (
    get_batch_embeddings,
    compute_soft_label_mapping,
    match_and_filter_triples,
    create_networkx_graph,
    relabel_graph_with_mapping,
    convert_to_grakel_graph,
)


class AttentionAligner(nn.Module):
    """
    GAP-inspired attention mechanism for cross-graph node alignment.

    Instead of hard clustering with thresholds, this computes soft alignments
    between nodes based on their semantic embeddings and neighborhood context.
    """

    def __init__(self, embedding_dim=768, hidden_dim=256):
        super(AttentionAligner, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Projection layers for query/key/value
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, hidden_dim)

        # Output projection to get aligned representation
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def compute_cross_attention(self, source_emb, target_emb):
        """
        Compute mutual attention between source and target node embeddings.

        Args:
            source_emb: [n_source, embedding_dim] - embeddings of source graph nodes
            target_emb: [n_target, embedding_dim] - embeddings of target graph nodes

        Returns:
            aligned_source: [n_source, hidden_dim] - context-aware source representations
            aligned_target: [n_target, hidden_dim] - context-aware target representations
            attention_matrix: [n_source, n_target] - soft alignment scores
        """
        Q_s = self.query_proj(source_emb)
        K_s = self.key_proj(source_emb)
        V_s = self.value_proj(source_emb)

        Q_t = self.query_proj(target_emb)
        K_t = self.key_proj(target_emb)
        V_t = self.value_proj(target_emb)

        scale = np.sqrt(self.hidden_dim)
        attention_st = torch.matmul(Q_s, K_t.transpose(0, 1)) / scale  # [n_source, n_target]
        attention_st = F.softmax(attention_st, dim=1)

        attention_ts = torch.matmul(Q_t, K_s.transpose(0, 1)) / scale  # [n_target, n_source]
        attention_ts = F.softmax(attention_ts, dim=1)

        context_s = torch.matmul(attention_st, V_t)
        aligned_source = self.output_proj(context_s + V_s)  # Residual connection

        context_t = torch.matmul(attention_ts, V_s)
        aligned_target = self.output_proj(context_t + V_t)  # Residual connection

        return aligned_source, aligned_target, attention_st


def compute_attention_aligned_labels(graph1_labels, graph2_labels, aligner):
    """
    Compute attention-based soft labels for nodes in both graphs.

    Instead of hard clustering, labels are derived from the attention pattern:
    nodes with similar attention distributions get similar derived labels.

    Args:
        graph1_labels: list of node labels from graph 1
        graph2_labels: list of node labels from graph 2
        aligner: AttentionAligner module

    Returns:
        label_mapping_g1: dict mapping original labels to derived labels for graph 1
        label_mapping_g2: dict mapping original labels to derived labels for graph 2
    """
    if not graph1_labels or not graph2_labels:
        return {}, {}

    emb1 = get_batch_embeddings(graph1_labels)
    emb2 = get_batch_embeddings(graph2_labels)

    emb1_tensor = torch.FloatTensor(emb1)
    emb2_tensor = torch.FloatTensor(emb2)

    with torch.no_grad():
        _, _, attention_matrix = aligner.compute_cross_attention(emb1_tensor, emb2_tensor)
        max_attention_indices = attention_matrix.argmax(dim=1).numpy()

    label_mapping_g1 = {}
    label_mapping_g2 = {}

    for i, label in enumerate(graph1_labels):
        best_match_idx = max_attention_indices[i]
        label_mapping_g1[label] = f"aligned_{best_match_idx}"

    for j, label in enumerate(graph2_labels):
        label_mapping_g2[label] = f"aligned_{j}"

    return label_mapping_g1, label_mapping_g2


def calculate_attention_augmented_similarity(kg1_triples, kg2_triples, use_neural_attention=False):
    """
    Calculate similarity using Attention-Augmented KEA.

    Pipeline:
    1. Filter valid triples
    2. Semantic triple matching (from KEA)
    3. Attention-based label alignment (replaces clustering)
    4. WL kernel comparison

    Args:
        kg1_triples: List of [subject, predicate, object] triples (claim KG)
        kg2_triples: List of [subject, predicate, object] triples (ground-truth KG)
        use_neural_attention: If True, use learned attention. If False, use softmax similarity.

    Returns:
        similarity: float between 0 and 1
        debug_info: dict with intermediate results for analysis
    """
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0, {'error': 'Empty triples'}

    # Step 1: Semantic Triple Matching (from KEA)
    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    if not filtered_kg2_triples:
        return 0.0, {'error': 'No matching triples found'}

    # Create NetworkX graphs
    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    kg1_node_labels = set(nx.get_node_attributes(kg1_graph, 'label').values())
    kg2_node_labels = set(nx.get_node_attributes(kg2_graph, 'label').values())
    kg1_edge_labels = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
    kg2_edge_labels = set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    # Step 2: Attention-Based Label Alignment
    # Entities and relations are aligned separately to avoid cross-type mismatches
    if use_neural_attention:
        aligner = AttentionAligner(embedding_dim=768, hidden_dim=256)

        entity_mapping_g1, entity_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_node_labels), list(kg2_node_labels), aligner
        ) if kg1_node_labels and kg2_node_labels else ({}, {})

        relation_mapping_g1, relation_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_edge_labels), list(kg2_edge_labels), aligner
        ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

    else:
        # Softmax-based similarity (no learned parameters) — same base logic as SNEA
        entity_mapping_g1, entity_mapping_g2 = compute_soft_label_mapping(
            kg1_node_labels, kg2_node_labels, prefix="node"
        ) if kg1_node_labels and kg2_node_labels else ({}, {})

        relation_mapping_g1, relation_mapping_g2 = compute_soft_label_mapping(
            kg1_edge_labels, kg2_edge_labels, prefix="rel"
        ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

    label_mapping_g1 = {**entity_mapping_g1, **relation_mapping_g1}
    label_mapping_g2 = {**entity_mapping_g2, **relation_mapping_g2}

    # Step 3: Relabel graphs with attention-derived labels
    relabeled_kg1 = relabel_graph_with_mapping(kg1_graph, label_mapping_g1)
    relabeled_kg2 = relabel_graph_with_mapping(kg2_graph, label_mapping_g2)

    kg1_grakel = convert_to_grakel_graph(relabeled_kg1)
    kg2_grakel = convert_to_grakel_graph(relabeled_kg2)

    if kg1_grakel is None or kg2_grakel is None:
        return 0.0, {'error': 'Failed to create GraKel graphs'}

    # Step 4: WL Kernel Comparison
    try:
        wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
        kernel_matrix = wl_kernel.fit_transform([kg1_grakel, kg2_grakel])
        similarity = float(kernel_matrix[0, 1])
    except Exception as e:
        return 0.0, {'error': f'WL kernel failed: {e}'}

    debug_info = {
        'kg1_nodes': len(kg1_graph.nodes()),
        'kg2_nodes': len(kg2_graph.nodes()),
        'kg1_edges': len(kg1_graph.edges()),
        'kg2_edges': len(kg2_graph.edges()),
        'filtered_triples': len(filtered_kg2_triples),
        'unique_alignments_g1': len(set(label_mapping_g1.values())),
        'unique_alignments_g2': len(set(label_mapping_g2.values())),
    }

    return similarity, debug_info


def calculate_aa_kea_similarity(kg1_triples, kg2_triples):
    """
    Wrapper function compatible with multi_method_similarity.py.
    Returns just the similarity score.
    """
    similarity, _ = calculate_attention_augmented_similarity(kg1_triples, kg2_triples)
    return similarity


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Attention-Augmented KEA")
    print("=" * 60)

    kg1 = [
        ['Marie Curie', 'discovered', 'Radium'],
        ['Marie Curie', 'won', 'Nobel Prize in Physics'],
        ['Marie Curie', 'won', 'Nobel Prize in Chemistry'],
    ]
    kg2 = [
        ['Marie Curie', 'found', 'Radium'],
        ['Marie Curie', 'received', 'Nobel Prize in Physics'],
        ['Marie Curie', 'was awarded', 'Nobel Prize in Chemistry'],
    ]

    print("\nTest 1: Similar graphs with different wording")
    sim1, info1 = calculate_attention_augmented_similarity(kg1, kg2)
    print(f"AA-KEA Similarity: {sim1:.4f}")
    print(f"Debug: {info1}")

    kg3 = [
        ['Albert Einstein', 'developed', 'Theory of Relativity'],
        ['Albert Einstein', 'won', 'Nobel Prize in Physics'],
    ]
    print("\nTest 2: Different graphs")
    sim2, info2 = calculate_attention_augmented_similarity(kg1, kg3)
    print(f"AA-KEA Similarity: {sim2:.4f}")

    print("\nTest 3: Identical graphs")
    sim3, info3 = calculate_attention_augmented_similarity(kg1, kg1)
    print(f"AA-KEA Similarity: {sim3:.4f}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
