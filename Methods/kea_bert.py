import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2')
sbert_model = sbert_model.to(DEVICE)


def get_sbert_embedding(label):
    """Get SBERT embedding for a single label"""
    embedding = sbert_model.encode(label, convert_to_tensor=True, device=DEVICE)
    return embedding.detach().cpu().numpy()


def get_batch_embeddings(labels):
    """Get SBERT embeddings for multiple labels efficiently"""
    if not labels:
        return np.array([])
    embeddings = sbert_model.encode(labels, convert_to_tensor=True, device=DEVICE)
    return embeddings.detach().cpu().numpy()


def get_triple_embedding(triple):
    """Get embedding for a triple by concatenating its text"""
    triple_text = ' '.join(map(str, triple))
    return get_sbert_embedding(triple_text)


def cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def match_and_filter_triples(kg1_triples, kg2_triples):
    """Match each triple in kg1 with most similar triple in kg2"""
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
# Attention-Based Alignment (from AA-KEA)
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
    emb1_tensor = torch.FloatTensor(emb1).to(DEVICE)
    emb2_tensor = torch.FloatTensor(emb2).to(DEVICE)

    with torch.no_grad():
        # Compute attention-aligned representations
        _, _, attention_matrix = aligner.compute_cross_attention(emb1_tensor, emb2_tensor)

        # Use max attention target as the "alignment anchor"
        max_attention_indices = attention_matrix.argmax(dim=1).cpu().numpy()

        # Create label mappings
        label_mapping_g1 = {}
        label_mapping_g2 = {}

        # For graph1: label based on which graph2 node it aligns to most
        for i, label in enumerate(graph1_labels):
            best_match_idx = max_attention_indices[i]
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


def extract_graph_components(nx_graph):
    """Extract nodes and edges from NetworkX graph"""
    nodes = list(nx.get_node_attributes(nx_graph, 'label').values())
    edges = list(nx.get_edge_attributes(nx_graph, 'relation').values())
    return nodes, edges


def compute_bertscore_similarity(source_labels, target_labels, idf_weights=None):
    """
    Compute BERTScore-inspired similarity between two sets of labels.

    Returns:
        precision: How well target covers source
        recall: How well source covers target
        f1: Harmonic mean of precision and recall
    """
    if not source_labels or not target_labels:
        return 0.0, 0.0, 0.0

    # Get embeddings
    source_embs = get_batch_embeddings(source_labels)
    target_embs = get_batch_embeddings(target_labels)

    # Normalize embeddings for cosine similarity
    source_embs_norm = source_embs / (np.linalg.norm(source_embs, axis=1, keepdims=True) + 1e-8)
    target_embs_norm = target_embs / (np.linalg.norm(target_embs, axis=1, keepdims=True) + 1e-8)

    # Compute pairwise similarity matrix
    similarity_matrix = np.dot(source_embs_norm, target_embs_norm.T)

    # Apply IDF weights if provided
    if idf_weights is not None:
        source_weights = np.array([idf_weights.get(label, 1.0) for label in source_labels])
        target_weights = np.array([idf_weights.get(label, 1.0) for label in target_labels])
    else:
        source_weights = np.ones(len(source_labels))
        target_weights = np.ones(len(target_labels))

    # Precision: For each target label, find best matching source label
    target_max_sims = np.max(similarity_matrix, axis=0)  # Max over source for each target
    precision = np.average(target_max_sims, weights=target_weights)

    # Recall: For each source label, find best matching target label
    source_max_sims = np.max(similarity_matrix, axis=1)  # Max over target for each source
    recall = np.average(source_max_sims, weights=source_weights)

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def compute_graph_bertscore(kg1_graph, kg2_graph):
    """
    Compute BERTScore-inspired similarity for entire graphs.

    Separately computes scores for:
    - Entity nodes (subjects and objects)
    - Relation edges (predicates)
    - Combines them into overall graph similarity
    """
    kg1_nodes, kg1_edges = extract_graph_components(kg1_graph)
    kg2_nodes, kg2_edges = extract_graph_components(kg2_graph)

    # Compute node-level similarity
    if kg1_nodes and kg2_nodes:
        node_precision, node_recall, node_f1 = compute_bertscore_similarity(kg1_nodes, kg2_nodes)
    else:
        node_precision, node_recall, node_f1 = 0.0, 0.0, 0.0

    # Compute edge-level similarity
    if kg1_edges and kg2_edges:
        edge_precision, edge_recall, edge_f1 = compute_bertscore_similarity(kg1_edges, kg2_edges)
    else:
        edge_precision, edge_recall, edge_f1 = 0.0, 0.0, 0.0

    # Combine node and edge scores (weighted average)
    # Give more weight to nodes as they represent entities
    node_weight = 0.6
    edge_weight = 0.4

    overall_precision = node_weight * node_precision + edge_weight * edge_precision
    overall_recall = node_weight * node_recall + edge_weight * edge_recall
    overall_f1 = node_weight * node_f1 + edge_weight * edge_f1

    return {
        'overall_f1': float(overall_f1),
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'node_f1': float(node_f1),
        'node_precision': float(node_precision),
        'node_recall': float(node_recall),
        'edge_f1': float(edge_f1),
        'edge_precision': float(edge_precision),
        'edge_recall': float(edge_recall),
    }


def compute_triple_level_bertscore(kg1_triples, kg2_triples):
    """
    Compute BERTScore at the triple level.
    Each triple is treated as a semantic unit (subject-predicate-object).
    """
    if not kg1_triples or not kg2_triples:
        return 0.0, 0.0, 0.0

    # Get embeddings for entire triples
    kg1_triple_texts = [' '.join(map(str, t)) for t in kg1_triples]
    kg2_triple_texts = [' '.join(map(str, t)) for t in kg2_triples]

    kg1_embs = get_batch_embeddings(kg1_triple_texts)
    kg2_embs = get_batch_embeddings(kg2_triple_texts)

    # Normalize
    kg1_embs_norm = kg1_embs / (np.linalg.norm(kg1_embs, axis=1, keepdims=True) + 1e-8)
    kg2_embs_norm = kg2_embs / (np.linalg.norm(kg2_embs, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix
    similarity_matrix = np.dot(kg1_embs_norm, kg2_embs_norm.T)

    # Precision and Recall
    precision = np.mean(np.max(similarity_matrix, axis=0))
    recall = np.mean(np.max(similarity_matrix, axis=1))

    # F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def calculate_kea_bert_similarity(kg1_triples, kg2_triples, return_detailed=False):
    """
    Main function: KEA-BERT similarity using pure semantic scoring.

    This replaces the Weisfeiler-Lehman kernel with BERTScore-inspired
    semantic similarity based on SBERT embeddings.

    Args:
        kg1_triples: List of triples from first knowledge graph
        kg2_triples: List of triples from second knowledge graph
        return_detailed: If True, return detailed metrics; otherwise just F1 score

    Returns:
        similarity score (float) or detailed metrics (dict)
    """
    # Filter valid triples
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0 if not return_detailed else {
            'overall_f1': 0.0,
            'graph_level': {},
            'triple_level': {},
            'error': 'Empty triples'
        }

    # Step 1: Match and filter triples (align KG2 to KG1)
    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    if not filtered_kg2_triples:
        return 0.0 if not return_detailed else {
            'overall_f1': 0.0,
            'graph_level': {},
            'triple_level': {},
            'error': 'No matching triples found'
        }

    # Step 2: Create graphs
    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    # Step 3: Compute graph-level BERTScore (nodes + edges)
    graph_metrics = compute_graph_bertscore(kg1_graph, kg2_graph)

    # Step 4: Compute triple-level BERTScore
    triple_precision, triple_recall, triple_f1 = compute_triple_level_bertscore(
        kg1_triples, filtered_kg2_triples
    )

    # Step 5: Combine graph-level and triple-level scores
    # Graph-level captures entity/relation alignment
    # Triple-level captures semantic coherence of complete statements
    graph_weight = 0.5
    triple_weight = 0.5

    final_f1 = graph_weight * graph_metrics['overall_f1'] + triple_weight * triple_f1

    if not return_detailed:
        return float(final_f1)

    # Return detailed metrics
    detailed_metrics = {
        'overall_f1': float(final_f1),
        'graph_level': graph_metrics,
        'triple_level': {
            'precision': float(triple_precision),
            'recall': float(triple_recall),
            'f1': float(triple_f1)
        },
        'metadata': {
            'kg1_triples': len(kg1_triples),
            'kg2_triples': len(kg2_triples),
            'filtered_kg2_triples': len(filtered_kg2_triples),
            'kg1_nodes': len(kg1_graph.nodes()),
            'kg2_nodes': len(kg2_graph.nodes()),
            'kg1_edges': len(kg1_graph.edges()),
            'kg2_edges': len(kg2_graph.edges()),
        }
    }

    return detailed_metrics


def calculate_kea_bert_with_attention(kg1_triples, kg2_triples, use_neural_attention=False, return_detailed=False):
    """
    Enhanced KEA-BERT with Attention-Based Alignment (from AA-KEA).

    Combines:
    1. Semantic triple matching (from KEA)
    2. Attention-based label alignment (from AA-KEA) - separate for entities and relations
    3. BERTScore-inspired semantic similarity (graph + triple level)

    This integrates the best of AA-KEA (attention alignment) with KEA-BERT (semantic scoring).

    Args:
        kg1_triples: List of triples from first knowledge graph
        kg2_triples: List of triples from second knowledge graph
        use_neural_attention: If True, use learned attention. If False, use softmax similarity.
        return_detailed: If True, return detailed metrics; otherwise just F1 score

    Returns:
        similarity score (float) or detailed metrics (dict)
    """
    # Filter valid triples
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0 if not return_detailed else {
            'overall_f1': 0.0,
            'graph_level': {},
            'triple_level': {},
            'error': 'Empty triples'
        }

    # Step 1: Match and filter triples (align KG2 to KG1)
    filtered_kg2_triples = match_and_filter_triples(kg1_triples, kg2_triples)

    if not filtered_kg2_triples:
        return 0.0 if not return_detailed else {
            'overall_f1': 0.0,
            'graph_level': {},
            'triple_level': {},
            'error': 'No matching triples found'
        }

    # Step 2: Create graphs
    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2_triples)

    # Step 3: Extract labels for attention alignment (separate entities and relations)
    kg1_node_labels = set(nx.get_node_attributes(kg1_graph, 'label').values())
    kg2_node_labels = set(nx.get_node_attributes(kg2_graph, 'label').values())
    kg1_edge_labels = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
    kg2_edge_labels = set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    # Step 4: Compute attention-based alignment (separate for entities and relations)
    if use_neural_attention:
        aligner = AttentionAligner(embedding_dim=768, hidden_dim=256)
        aligner = aligner.to(DEVICE)

        # Align entities (nodes) separately
        entity_mapping_g1, entity_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_node_labels), list(kg2_node_labels), aligner
        ) if kg1_node_labels and kg2_node_labels else ({}, {})

        # Align relations (edges) separately
        relation_mapping_g1, relation_mapping_g2 = compute_attention_aligned_labels(
            list(kg1_edge_labels), list(kg2_edge_labels), aligner
        ) if kg1_edge_labels and kg2_edge_labels else ({}, {})
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

    # Step 5: Relabel graphs with attention-derived labels
    relabeled_kg1 = relabel_graph_with_mapping(kg1_graph, label_mapping_g1)
    relabeled_kg2 = relabel_graph_with_mapping(kg2_graph, label_mapping_g2)

    # Step 6: Compute graph-level BERTScore (after attention alignment)
    graph_metrics = compute_graph_bertscore(relabeled_kg1, relabeled_kg2)

    # Step 7: Compute triple-level BERTScore
    triple_precision, triple_recall, triple_f1 = compute_triple_level_bertscore(
        kg1_triples, filtered_kg2_triples
    )

    # Step 8: Combine graph-level and triple-level scores
    # Graph-level captures entity/relation alignment
    # Triple-level captures semantic coherence of complete statements
    graph_weight = 0.5
    triple_weight = 0.5

    final_f1 = graph_weight * graph_metrics['overall_f1'] + triple_weight * triple_f1

    if not return_detailed:
        return float(final_f1)

    # Return detailed metrics
    detailed_metrics = {
        'overall_f1': float(final_f1),
        'graph_level': graph_metrics,
        'triple_level': {
            'precision': float(triple_precision),
            'recall': float(triple_recall),
            'f1': float(triple_f1)
        },
        'metadata': {
            'kg1_triples': len(kg1_triples),
            'kg2_triples': len(kg2_triples),
            'filtered_kg2_triples': len(filtered_kg2_triples),
            'kg1_nodes': len(kg1_graph.nodes()),
            'kg2_nodes': len(kg2_graph.nodes()),
            'kg1_edges': len(kg1_graph.edges()),
            'kg2_edges': len(kg2_graph.edges()),
            'unique_alignments_g1': len(set(label_mapping_g1.values())),
            'unique_alignments_g2': len(set(label_mapping_g2.values())),
        }
    }

    return detailed_metrics


# Wrapper function for compatibility with multi_method_similarity.py
def calculate_similarity(kg1_triples, kg2_triples):
    """Compatibility wrapper that returns just the F1 score"""
    return calculate_kea_bert_similarity(kg1_triples, kg2_triples, return_detailed=False)
