#!/usr/bin/env python3
"""
SNEA - Semantic Node Edge Aligned Similarity

Performs entity/relationship-separated Knowledge Graph similarity using:
1. Semantic Triple Matching (SBERT-based triple selection, from KEA)
2. Soft Label Alignment - entities and relations aligned separately
   via cosine similarity with threshold=0.65
3. WL Kernel Comparison on relabelled graphs

The key design is treating node labels (entities) and edge labels (relations)
as separate alignment problems, preventing cross-type label collisions.
This file also provides shared utilities used by aa_kea.py.

Author: Research Implementation
"""

import networkx as nx
import numpy as np
import torch
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from sentence_transformers import SentenceTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2', device=DEVICE)


# ── shared utilities ───────────────────────────────────────────────────────────

def get_sbert_embedding(label):
    """Get SBERT embedding for a label."""
    embedding = sbert_model.encode(label, convert_to_tensor=True, device=DEVICE)
    return embedding.detach().cpu().numpy()


def get_batch_embeddings(labels):
    """Get SBERT embeddings for multiple labels efficiently."""
    if not labels:
        return np.array([])
    embeddings = sbert_model.encode(labels, convert_to_tensor=True, device=DEVICE)
    return embeddings.detach().cpu().numpy()


def cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def get_triple_embedding(triple):
    """Get embedding for a triple by concatenating its text."""
    triple_text = ' '.join(map(str, triple))
    return get_sbert_embedding(triple_text)


def match_and_filter_triples(kg1_triples, kg2_triples):
    """
    Match each triple in kg1 with the most similar triple in kg2.
    This is the Semantic Relation Selection step from KEA.
    """
    picked_triples = []
    for triple in kg1_triples:
        curr_kg1_embedding = get_triple_embedding(triple)
        similarities = []
        for other_triple in kg2_triples:
            curr_kg2_embedding = get_triple_embedding(other_triple)
            curr_similarity = cosine_sim(curr_kg1_embedding, curr_kg2_embedding)
            similarities.append((curr_similarity, other_triple))

        chosen_triple = max(similarities, key=lambda x: x[0])
        if chosen_triple[1] not in picked_triples:
            picked_triples.append(chosen_triple[1])
    return picked_triples


def compute_soft_label_mapping(graph1_labels, graph2_labels, similarity_threshold=0.65, prefix="anchor"):
    """
    Compute soft labels using direct embedding similarity.

    For each label in graph1, finds its best-matching label in graph2 via cosine
    similarity. If the best match exceeds the threshold, the label is replaced with
    an anchor derived from the graph2 index; otherwise the original label is kept.

    Args:
        graph1_labels: Labels from graph 1
        graph2_labels: Labels from graph 2
        similarity_threshold: Minimum cosine similarity for alignment (default: 0.65)
        prefix: Prefix for anchor labels (e.g., "node" or "rel" to distinguish types)

    Returns:
        label_mapping_g1: dict mapping graph1 labels -> aligned anchor labels
        label_mapping_g2: dict mapping graph2 labels -> anchor labels
    """
    if not graph1_labels or not graph2_labels:
        return {}, {}

    emb1 = get_batch_embeddings(list(graph1_labels))
    emb2 = get_batch_embeddings(list(graph2_labels))

    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(emb1_norm, emb2_norm.T)  # [n1, n2]

    label_mapping_g1 = {}
    label_mapping_g2 = {}

    graph1_labels_list = list(graph1_labels)
    graph2_labels_list = list(graph2_labels)

    for i, label in enumerate(graph1_labels_list):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_match_similarity = similarity_matrix[i, best_match_idx]
        if best_match_similarity >= similarity_threshold:
            label_mapping_g1[label] = f"{prefix}_{best_match_idx}"
        else:
            label_mapping_g1[label] = label

    for j, label in enumerate(graph2_labels_list):
        label_mapping_g2[label] = f"{prefix}_{j}"

    return label_mapping_g1, label_mapping_g2


def create_networkx_graph(triple_list):
    """Create a NetworkX graph from a list of (subject, predicate, object) triples."""
    G = nx.Graph()
    for triple in triple_list:
        if len(triple) == 3:
            subject, predicate, obj = triple
            G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
            G.nodes[subject.lower()]['label'] = subject.lower()
            G.nodes[obj.lower()]['label'] = obj.lower()
    return G


def relabel_graph_with_mapping(nx_graph, label_mapping):
    """
    Relabel graph nodes using a label mapping.
    Preserves graph structure while updating labels to reflect cross-graph alignments.
    """
    new_graph = nx.Graph()

    for node, data in nx_graph.nodes(data=True):
        original_label = data.get('label', node)
        aligned_label = label_mapping.get(original_label, original_label)
        new_graph.add_node(node, label=aligned_label)

    for u, v, data in nx_graph.edges(data=True):
        new_graph.add_edge(u, v, relation=data.get('relation', 'related'))

    return new_graph


def convert_to_grakel_graph(nx_graph):
    """Convert a NetworkX graph to GraKel format."""
    node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
    edge_labels = {(u, v): data.get('relation', 'default') for u, v, data in nx_graph.edges(data=True)}
    edges = {(u, v): 1 for u, v in nx_graph.edges()}

    if not edges:
        return None

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)


# ── SNEA method ────────────────────────────────────────────────────────────────

def calculate_snea_similarity(kg1_triples, kg2_triples):
    """
    Calculate graph similarity using Semantic Node Edge Aligned (SNEA) method.

    Pipeline:
    1. Filter valid triples
    2. Semantic triple matching (SBERT-based, from KEA)
    3. Soft label alignment — entities (nodes) and relations (edges) aligned
       separately using cosine similarity with threshold=0.65
    4. WL Kernel comparison on relabelled graphs

    Args:
        kg1_triples: List of [subject, predicate, object] triples
        kg2_triples: List of [subject, predicate, object] triples

    Returns:
        similarity: float in [0, 1]
        debug_info: dict with intermediate results
    """
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0, {'error': 'Empty triples'}

    # Step 1: Semantic Triple Matching
    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    if not filtered_kg2_triples:
        return 0.0, {'error': 'No matching triples found'}

    # Build NetworkX graphs
    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    kg1_node_labels = set(nx.get_node_attributes(kg1_graph, 'label').values())
    kg2_node_labels = set(nx.get_node_attributes(kg2_graph, 'label').values())
    kg1_edge_labels = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
    kg2_edge_labels = set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    # Step 2: Separate alignment for entities and relations
    entity_mapping_g1, entity_mapping_g2 = compute_soft_label_mapping(
        kg1_node_labels, kg2_node_labels, prefix="node"
    ) if kg1_node_labels and kg2_node_labels else ({}, {})

    relation_mapping_g1, relation_mapping_g2 = compute_soft_label_mapping(
        kg1_edge_labels, kg2_edge_labels, prefix="rel"
    ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

    label_mapping_g1 = {**entity_mapping_g1, **relation_mapping_g1}
    label_mapping_g2 = {**entity_mapping_g2, **relation_mapping_g2}

    # Step 3: Relabel graphs
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


def calculate_snea_similarity_score(kg1_triples, kg2_triples):
    """Wrapper returning only the similarity score (no debug info)."""
    similarity, _ = calculate_snea_similarity(kg1_triples, kg2_triples)
    return similarity


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SNEA - Semantic Node Edge Aligned Similarity")
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
    sim1, info1 = calculate_snea_similarity(kg1, kg2)
    print(f"SNEA Similarity: {sim1:.4f}")
    print(f"Debug: {info1}")

    kg3 = [
        ['Albert Einstein', 'developed', 'Theory of Relativity'],
        ['Albert Einstein', 'won', 'Nobel Prize in Physics'],
    ]
    print("\nTest 2: Different graphs")
    sim2, info2 = calculate_snea_similarity(kg1, kg3)
    print(f"SNEA Similarity: {sim2:.4f}")

    print("\nTest 3: Identical graphs")
    sim3, info3 = calculate_snea_similarity(kg1, kg1)
    print(f"SNEA Similarity: {sim3:.4f}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
