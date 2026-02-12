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

Author: Research Implementation
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from sentence_transformers import SentenceTransformer

# Load SBERT model once (shared with KEA for fair comparison)
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2')


def get_sbert_embedding(label):
    """Get SBERT embedding for a label"""
    embedding = sbert_model.encode(label, convert_to_tensor=True)
    return embedding.detach().cpu().numpy()


def get_batch_embeddings(labels):
    """Get SBERT embeddings for multiple labels efficiently"""
    if not labels:
        return np.array([])
    embeddings = sbert_model.encode(labels, convert_to_tensor=True)
    return embeddings.detach().cpu().numpy()


# =============================================================================
# STEP 1: Semantic Triple Matching (kept from KEA)
# =============================================================================

def cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def get_triple_embedding(triple):
    """Get embedding for a triple by concatenating its text"""
    triple_text = ' '.join(map(str, triple))
    return get_sbert_embedding(triple_text)


def match_and_filter_triples(kg1_triples, kg2_triples):
    """
    Match each triple in kg1 with most similar triple in kg2.
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


# =============================================================================
# STEP 2: GAP-Style Attention Alignment (replaces semantic clustering)
# =============================================================================

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

        # Initialize weights
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
        # Project to query/key/value spaces
        Q_s = self.query_proj(source_emb)  # [n_source, hidden_dim]
        K_s = self.key_proj(source_emb)
        V_s = self.value_proj(source_emb)

        Q_t = self.query_proj(target_emb)  # [n_target, hidden_dim]
        K_t = self.key_proj(target_emb)
        V_t = self.value_proj(target_emb)

        # Compute cross-attention: source attending to target
        # A_st[i,j] = softmax(Q_s[i] · K_t[j]^T / sqrt(d))
        scale = np.sqrt(self.hidden_dim)
        attention_st = torch.matmul(Q_s, K_t.transpose(0, 1)) / scale  # [n_source, n_target]
        attention_st = F.softmax(attention_st, dim=1)

        # Compute cross-attention: target attending to source
        attention_ts = torch.matmul(Q_t, K_s.transpose(0, 1)) / scale  # [n_target, n_source]
        attention_ts = F.softmax(attention_ts, dim=1)

        # Get context-aware representations
        # Source nodes enriched with target context
        context_s = torch.matmul(attention_st, V_t)  # [n_source, hidden_dim]
        aligned_source = self.output_proj(context_s + V_s)  # Residual connection

        # Target nodes enriched with source context
        context_t = torch.matmul(attention_ts, V_s)  # [n_target, hidden_dim]
        aligned_target = self.output_proj(context_t + V_t)  # Residual connection

        return aligned_source, aligned_target, attention_st


def compute_attention_aligned_labels(graph1_labels, graph2_labels, aligner):
    """
    Compute attention-based soft labels for nodes in both graphs.

    Instead of hard clustering, we derive labels from the attention pattern:
    - Nodes with similar attention distributions get similar derived labels
    - This preserves fine-grained distinctions that clustering would blur

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

    # Get SBERT embeddings for all labels
    emb1 = get_batch_embeddings(graph1_labels)
    emb2 = get_batch_embeddings(graph2_labels)

    # Convert to tensors
    emb1_tensor = torch.FloatTensor(emb1)
    emb2_tensor = torch.FloatTensor(emb2)

    with torch.no_grad():
        # Compute attention-aligned representations
        aligned1, aligned2, attention_matrix = aligner.compute_cross_attention(emb1_tensor, emb2_tensor)

        # Convert aligned representations to numpy
        aligned1_np = aligned1.numpy()
        aligned2_np = aligned2.numpy()

        # Derive soft labels by quantizing the aligned representations
        # We use k-means-style assignment to a fixed set of "virtual clusters"
        # But instead of clustering, we use the attention pattern directly

        # Method: Use max attention target as the "alignment anchor"
        # This gives each source node a label based on its best-matching target
        max_attention_indices = attention_matrix.argmax(dim=1).numpy()

        # Create label mappings
        label_mapping_g1 = {}
        label_mapping_g2 = {}

        # For graph1: label based on which graph2 node it aligns to most
        for i, label in enumerate(graph1_labels):
            best_match_idx = max_attention_indices[i]
            best_match_label = graph2_labels[best_match_idx]
            # Create a combined label that preserves both identity and alignment
            label_mapping_g1[label] = f"aligned_{best_match_idx}"

        # For graph2: each node becomes an alignment anchor
        for j, label in enumerate(graph2_labels):
            label_mapping_g2[label] = f"aligned_{j}"

    return label_mapping_g1, label_mapping_g2


def compute_soft_label_mapping(graph1_labels, graph2_labels, similarity_threshold=0.65, prefix="anchor"):
    """
    Alternative: Compute soft labels using direct embedding similarity with learned weighting.

    This version uses a similarity threshold - only aligns labels if their cosine similarity
    is above the threshold.

    Args:
        graph1_labels: Labels from graph 1
        graph2_labels: Labels from graph 2
        similarity_threshold: Minimum cosine similarity required for alignment (default: 0.65)
        prefix: Prefix for anchor labels (e.g., "node" or "rel" to distinguish types)
    """
    if not graph1_labels or not graph2_labels:
        return {}, {}

    # Get embeddings
    emb1 = get_batch_embeddings(list(graph1_labels))
    emb2 = get_batch_embeddings(list(graph2_labels))

    # Compute pairwise similarity matrix
    # sim[i,j] = cosine_similarity(emb1[i], emb2[j])
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(emb1_norm, emb2_norm.T)  # [n1, n2]

    # For each label in graph1, find its best alignment in graph2
    label_mapping_g1 = {}
    label_mapping_g2 = {}

    graph1_labels_list = list(graph1_labels)
    graph2_labels_list = list(graph2_labels)

    # Graph1 labels: map to their best-matching graph2 anchor only if similarity exceeds threshold
    for i, label in enumerate(graph1_labels_list):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_match_similarity = similarity_matrix[i, best_match_idx]
        
        # Only create alignment if similarity exceeds threshold
        if best_match_similarity >= similarity_threshold:
            label_mapping_g1[label] = f"{prefix}_{best_match_idx}"
        else:
            # Keep original label if no good match found
            label_mapping_g1[label] = label

    # Graph2 labels: each becomes its own anchor
    for j, label in enumerate(graph2_labels_list):
        label_mapping_g2[label] = f"{prefix}_{j}"

    return label_mapping_g1, label_mapping_g2


# =============================================================================
# STEP 3: Graph Construction and WL Kernel (adapted from KEA)
# =============================================================================

def create_networkx_graph(triple_list):
    """Create NetworkX graph from triples"""
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
    Relabel graph nodes using the attention-derived label mapping.

    Unlike KEA's clustering which merges nodes, this preserves graph structure
    while changing labels to reflect cross-graph alignments.
    """
    new_graph = nx.Graph()

    # Map nodes to their aligned labels
    node_to_aligned = {}
    for node, data in nx_graph.nodes(data=True):
        original_label = data.get('label', node)
        # Use mapping if available, otherwise keep original
        aligned_label = label_mapping.get(original_label, original_label)
        node_to_aligned[node] = aligned_label
        new_graph.add_node(node, label=aligned_label)

    # Copy edges with original relations
    for u, v, data in nx_graph.edges(data=True):
        new_graph.add_edge(u, v, relation=data.get('relation', 'related'))

    return new_graph


def convert_to_grakel_graph(nx_graph):
    """Convert NetworkX graph to GraKel format"""
    node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
    edge_labels = {(u, v): data.get('relation', 'default') for u, v, data in nx_graph.edges(data=True)}
    edges = {(u, v): 1 for u, v in nx_graph.edges()}

    if not edges:
        # Handle empty graph
        return None

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)


# =============================================================================
# MAIN SIMILARITY FUNCTION
# =============================================================================

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
    # Filter valid triples
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

    # Collect all labels from both graphs
    kg1_node_labels = set(nx.get_node_attributes(kg1_graph, 'label').values())
    kg2_node_labels = set(nx.get_node_attributes(kg2_graph, 'label').values())
    kg1_edge_labels = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
    kg2_edge_labels = set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    # Step 2: Attention-Based Label Alignment (replaces clustering)
    # IMPORTANT: Align entities and relations separately to avoid mismatches

    if use_neural_attention:
        aligner = AttentionAligner(embedding_dim=768, hidden_dim=256)

        # Align entities (nodes) separately
        entity_mapping_g1, entity_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_node_labels), list(kg2_node_labels), aligner
        ) if kg1_node_labels and kg2_node_labels else ({}, {})

        # Align relations (edges) separately
        relation_mapping_g1, relation_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_edge_labels), list(kg2_edge_labels), aligner
        ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

        # Combine mappings
        label_mapping_g1 = {**entity_mapping_g1, **relation_mapping_g1}
        label_mapping_g2 = {**entity_mapping_g2, **relation_mapping_g2}
    else:
        # Use softmax-based similarity (no learned parameters)

        # Align entities (nodes) separately with "node" prefix
        entity_mapping_g1, entity_mapping_g2 = compute_soft_label_mapping(
            kg1_node_labels, kg2_node_labels, prefix="node"
        ) if kg1_node_labels and kg2_node_labels else ({}, {})

        # Align relations (edges) separately with "rel" prefix
        relation_mapping_g1, relation_mapping_g2 = compute_soft_label_mapping(
            kg1_edge_labels, kg2_edge_labels, prefix="rel"
        ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

        # Combine mappings
        label_mapping_g1 = {**entity_mapping_g1, **relation_mapping_g1}
        label_mapping_g2 = {**entity_mapping_g2, **relation_mapping_g2}

    # Step 3: Relabel graphs with attention-derived labels
    relabeled_kg1 = relabel_graph_with_mapping(kg1_graph, label_mapping_g1)
    relabeled_kg2 = relabel_graph_with_mapping(kg2_graph, label_mapping_g2)

    # Convert to GraKel format
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


# =============================================================================
# ENHANCED VERSION: Neighborhood-Aware Attention (NOT USED IN PROGRESS EVALUATION)
# =============================================================================

# def get_node_neighborhood_embedding(graph, node, hop=1):
#     """
#     Get a neighborhood-aware embedding for a node.
#     Combines the node's own embedding with its neighbors' embeddings.
#     """
#     # Get node's own label
#     node_label = graph.nodes[node].get('label', node)
#     node_emb = get_sbert_embedding(node_label)
#
#     # Get neighbor labels
#     neighbors = list(graph.neighbors(node))
#     if not neighbors:
#         return node_emb
#
#     neighbor_labels = [graph.nodes[n].get('label', n) for n in neighbors]
#     neighbor_embs = get_batch_embeddings(neighbor_labels)
#
#     # Combine: node embedding + mean of neighbor embeddings
#     neighbor_mean = neighbor_embs.mean(axis=0)
#     combined = 0.7 * node_emb + 0.3 * neighbor_mean  # Weighted combination
#
#     return combined
#
#
# def calculate_neighborhood_attention_similarity(kg1_triples, kg2_triples):
#     """
#     Enhanced version that considers neighborhood structure in attention computation.
#
#     This better captures the graph context when computing alignments.
#     """
#     # Filter valid triples
#     kg1_triples = [t for t in kg1_triples if len(t) == 3]
#     kg2_triples = [t for t in kg2_triples if len(t) == 3]
#
#     if not kg1_triples or not kg2_triples:
#         return 0.0, {'error': 'Empty triples'}
#
#     # Semantic triple matching
#     filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)
#
#     if not filtered_kg2_triples:
#         return 0.0, {'error': 'No matching triples found'}
#
#     # Create graphs
#     kg1_graph = create_networkx_graph(kg1_triples)
#     kg2_graph = create_networkx_graph(filtered_kg2_triples)
#
#     # Get neighborhood-aware embeddings for all nodes
#     kg1_nodes = list(kg1_graph.nodes())
#     kg2_nodes = list(kg2_graph.nodes())
#
#     kg1_embs = np.array([get_node_neighborhood_embedding(kg1_graph, n) for n in kg1_nodes])
#     kg2_embs = np.array([get_node_neighborhood_embedding(kg2_graph, n) for n in kg2_nodes])
#
#     # Compute attention-based alignment using neighborhood embeddings
#     kg1_embs_norm = kg1_embs / (np.linalg.norm(kg1_embs, axis=1, keepdims=True) + 1e-8)
#     kg2_embs_norm = kg2_embs / (np.linalg.norm(kg2_embs, axis=1, keepdims=True) + 1e-8)
#
#     similarity_matrix = np.dot(kg1_embs_norm, kg2_embs_norm.T)
#
#     # Softmax attention
#     temperature = 0.1
#     attention = np.exp(similarity_matrix / temperature)
#     attention = attention / attention.sum(axis=1, keepdims=True)
#
#     # Create label mappings
#     label_mapping_g1 = {}
#     label_mapping_g2 = {}
#
#     for i, node in enumerate(kg1_nodes):
#         label = kg1_graph.nodes[node].get('label', node)
#         best_match = np.argmax(attention[i])
#         label_mapping_g1[label] = f"aligned_{best_match}"
#
#     for j, node in enumerate(kg2_nodes):
#         label = kg2_graph.nodes[node].get('label', node)
#         label_mapping_g2[label] = f"aligned_{j}"
#
#     # Also map edge labels (relations) using simple matching
#     kg1_relations = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
#     kg2_relations = set(nx.get_edge_attributes(kg2_graph, 'relation').values())
#
#     rel_mapping_g1, rel_mapping_g2 = compute_soft_label_mapping(kg1_relations, kg2_relations)
#     label_mapping_g1.update(rel_mapping_g1)
#     label_mapping_g2.update(rel_mapping_g2)
#
#     # Relabel and compare
#     relabeled_kg1 = relabel_graph_with_mapping(kg1_graph, label_mapping_g1)
#     relabeled_kg2 = relabel_graph_with_mapping(kg2_graph, label_mapping_g2)
#
#     kg1_grakel = convert_to_grakel_graph(relabeled_kg1)
#     kg2_grakel = convert_to_grakel_graph(relabeled_kg2)
#
#     if kg1_grakel is None or kg2_grakel is None:
#         return 0.0, {'error': 'Failed to create graphs'}
#
#     try:
#         wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
#         kernel_matrix = wl_kernel.fit_transform([kg1_grakel, kg2_grakel])
#         similarity = float(kernel_matrix[0, 1])
#     except Exception as e:
#         return 0.0, {'error': f'WL kernel failed: {e}'}
#
#     return similarity, {
#         'method': 'neighborhood_attention',
#         'kg1_nodes': len(kg1_nodes),
#         'kg2_nodes': len(kg2_nodes)
#     }


# =============================================================================
# WRAPPER FOR COMPARISON SCRIPT
# =============================================================================

def calculate_aa_kea_similarity(kg1_triples, kg2_triples):
    """
    Wrapper function compatible with multi_method_similarity.py
    Returns just the similarity score.
    """
    similarity, _ = calculate_attention_augmented_similarity(kg1_triples, kg2_triples)
    return similarity


# def calculate_aa_kea_neighborhood_similarity(kg1_triples, kg2_triples):
#     """
#     Wrapper for neighborhood-aware version. (NOT USED IN PROGRESS EVALUATION)
#     """
#     similarity, _ = calculate_neighborhood_attention_similarity(kg1_triples, kg2_triples)
#     return similarity


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Attention-Augmented KEA")
    print("=" * 60)

    # Test case 1: Similar graphs with different wording
    kg1 = [
        ['Marie Curie', 'discovered', 'Radium'],
        ['Marie Curie', 'won', 'Nobel Prize in Physics'],
        ['Marie Curie', 'won', 'Nobel Prize in Chemistry']
    ]

    kg2 = [
        ['Marie Curie', 'found', 'Radium'],
        ['Marie Curie', 'received', 'Nobel Prize in Physics'],
        ['Marie Curie', 'was awarded', 'Nobel Prize in Chemistry']
    ]

    print("\nTest 1: Similar graphs with different wording")
    print(f"KG1: {kg1}")
    print(f"KG2: {kg2}")

    sim1, info1 = calculate_attention_augmented_similarity(kg1, kg2)
    print(f"AA-KEA Similarity: {sim1:.4f}")
    print(f"Debug: {info1}")

    sim1_neigh, _ = calculate_neighborhood_attention_similarity(kg1, kg2)
    print(f"AA-KEA (Neighborhood): {sim1_neigh:.4f}")

    # Test case 2: Different graphs
    kg3 = [
        ['Albert Einstein', 'developed', 'Theory of Relativity'],
        ['Albert Einstein', 'won', 'Nobel Prize in Physics']
    ]

    print("\nTest 2: Different graphs")
    print(f"KG1: {kg1}")
    print(f"KG3: {kg3}")

    sim2, info2 = calculate_attention_augmented_similarity(kg1, kg3)
    print(f"AA-KEA Similarity: {sim2:.4f}")

    # Test case 3: Identical graphs
    print("\nTest 3: Identical graphs")
    sim3, info3 = calculate_attention_augmented_similarity(kg1, kg1)
    print(f"AA-KEA Similarity: {sim3:.4f}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
