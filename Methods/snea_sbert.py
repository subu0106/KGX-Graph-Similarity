#!/usr/bin/env python3
"""
SNEA-SBERT — SNEA + SBERT Semantic Blend

Extends SNEA (snea.py) by adding a continuous SBERT semantic component to
fix the discrete/quantized score problem produced by the WL kernel alone.

Why SNEA produces stepped/flat ROC curves:
    WL kernel counts integer matches of discrete label patterns on small KGs
    (4-12 nodes). After soft label alignment collapses labels into a handful
    of strings (node_0, node_1, ...), only ~5-6 distinct WL scores are
    possible → stepped ROC / poor threshold resolution.

Fix:
    Keep the full SNEA pipeline unchanged (triple matching, entity/relation
    soft label alignment, WL kernel). After computing the WL score, also
    compute the cosine similarity between mean-pooled SBERT embeddings of
    both KGs, then average the two:

        score = 0.5 * wl_score + 0.5 * sbert_score

    The SBERT component is smooth and continuously distributed, which
    dilutes the discrete WL values and produces a smooth output range.

Pipeline:
    1. Semantic Triple Matching   — same as snea.py
    2. Soft Label Alignment       — same as snea.py (entity + relation separated)
    3. WL Kernel                  — same as snea.py  → wl_score
    4. SBERT Mean-Pool Cosine     — new              → sbert_score
    5. Blend at alpha=0.5         — new              → final score
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from sentence_transformers import SentenceTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sbert_model = SentenceTransformer('paraphrase-MPNet-base-v2', device=DEVICE)

ALPHA = 0.5   # weight for WL score; (1 - ALPHA) goes to SBERT score


# ---------------------------------------------------------------------------
# Embedding helpers  (unchanged from aa_kea.py)
# ---------------------------------------------------------------------------

def get_sbert_embedding(label):
    embedding = sbert_model.encode(label, convert_to_tensor=True, device=DEVICE)
    return embedding.detach().cpu().numpy()


def get_batch_embeddings(labels):
    if not labels:
        return np.array([])
    embeddings = sbert_model.encode(labels, convert_to_tensor=True, device=DEVICE)
    return embeddings.detach().cpu().numpy()


def cosine_sim(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def get_triple_embedding(triple):
    triple_text = ' '.join(map(str, triple))
    return get_sbert_embedding(triple_text)


# ---------------------------------------------------------------------------
# Step 1 — Semantic Triple Matching  (unchanged from aa_kea.py)
# ---------------------------------------------------------------------------

def match_and_filter_triples(kg1_triples, kg2_triples):
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


# ---------------------------------------------------------------------------
# Step 2 — Soft Label Alignment  (unchanged from aa_kea.py)
# ---------------------------------------------------------------------------

def compute_soft_label_mapping(graph1_labels, graph2_labels, similarity_threshold=0.65, prefix="anchor"):
    if not graph1_labels or not graph2_labels:
        return {}, {}

    emb1 = get_batch_embeddings(list(graph1_labels))
    emb2 = get_batch_embeddings(list(graph2_labels))

    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(emb1_norm, emb2_norm.T)

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


# ---------------------------------------------------------------------------
# Graph construction helpers  (unchanged from aa_kea.py)
# ---------------------------------------------------------------------------

def create_networkx_graph(triple_list):
    G = nx.Graph()
    for triple in triple_list:
        if len(triple) == 3:
            subject, predicate, obj = triple
            G.add_edge(subject.lower(), obj.lower(), relation=predicate.lower())
            G.nodes[subject.lower()]['label'] = subject.lower()
            G.nodes[obj.lower()]['label'] = obj.lower()
    return G


def relabel_graph_with_mapping(nx_graph, label_mapping):
    new_graph = nx.Graph()
    node_to_aligned = {}
    for node, data in nx_graph.nodes(data=True):
        original_label = data.get('label', node)
        aligned_label = label_mapping.get(original_label, original_label)
        node_to_aligned[node] = aligned_label
        new_graph.add_node(node, label=aligned_label)
    for u, v, data in nx_graph.edges(data=True):
        new_graph.add_edge(u, v, relation=data.get('relation', 'related'))
    return new_graph


def convert_to_grakel_graph(nx_graph):
    node_labels = {node: data.get('label', node) for node, data in nx_graph.nodes(data=True)}
    edge_labels = {(u, v): data.get('relation', 'default') for u, v, data in nx_graph.edges(data=True)}
    edges = {(u, v): 1 for u, v in nx_graph.edges()}
    if not edges:
        return None
    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)


# ---------------------------------------------------------------------------
# Step 4 — SBERT mean-pool cosine similarity  (new)
# ---------------------------------------------------------------------------

def _sbert_mean_cosine(kg1_triples, kg2_triples):
    """
    Compute cosine similarity between the mean-pooled SBERT embeddings of
    both KGs. Each triple is embedded as a single sentence (subject + predicate
    + object concatenated). Returns a float in [-1, 1], typically [0, 1].
    """
    texts1 = [' '.join(map(str, t)) for t in kg1_triples]
    texts2 = [' '.join(map(str, t)) for t in kg2_triples]

    emb1 = get_batch_embeddings(texts1).mean(axis=0)   # [768]
    emb2 = get_batch_embeddings(texts2).mean(axis=0)   # [768]

    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


# ---------------------------------------------------------------------------
# Main similarity function
# ---------------------------------------------------------------------------

def calculate_snea_sbert_similarity(kg1_triples, kg2_triples, alpha=ALPHA):
    """
    Compute similarity between two KGs using the SNEA-SBERT method.

    Steps 1-3 are identical to snea.py (triple matching → entity/relation
    soft label alignment → WL kernel). Step 4 adds a continuous SBERT
    component, and Step 5 blends the two at the given alpha.

    Args:
        kg1_triples : list of [subject, predicate, object] — gold / reference KG
        kg2_triples : list of [subject, predicate, object] — LLM-generated KG
        alpha       : weight for the WL kernel score (default 0.5).
                      (1 - alpha) is the weight for the SBERT cosine score.

    Returns:
        similarity  : float in [0.0, 1.0]
        debug_info  : dict with wl_score, sbert_score, and intermediate details
    """
    # -- Validate --
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0, {'error': 'Empty triples'}

    # -- Step 1: Semantic triple matching --
    filtered_kg2 = match_and_filter_triples(kg1_triples, kg2_triples)
    if not filtered_kg2:
        return 0.0, {'error': 'No matching triples found'}

    # -- Step 2: Build graphs --
    kg1_graph = create_networkx_graph(kg1_triples)
    kg2_graph = create_networkx_graph(filtered_kg2)

    kg1_node_labels = set(nx.get_node_attributes(kg1_graph, 'label').values())
    kg2_node_labels = set(nx.get_node_attributes(kg2_graph, 'label').values())
    kg1_edge_labels = set(nx.get_edge_attributes(kg1_graph, 'relation').values())
    kg2_edge_labels = set(nx.get_edge_attributes(kg2_graph, 'relation').values())

    # -- Step 2: Soft label alignment --
    entity_mapping_g1, entity_mapping_g2 = compute_soft_label_mapping(
        kg1_node_labels, kg2_node_labels, prefix="node"
    ) if kg1_node_labels and kg2_node_labels else ({}, {})

    relation_mapping_g1, relation_mapping_g2 = compute_soft_label_mapping(
        kg1_edge_labels, kg2_edge_labels, prefix="rel"
    ) if kg1_edge_labels and kg2_edge_labels else ({}, {})

    label_mapping_g1 = {**entity_mapping_g1, **relation_mapping_g1}
    label_mapping_g2 = {**entity_mapping_g2, **relation_mapping_g2}

    # -- Step 3: Relabel and run WL kernel --
    relabeled_kg1 = relabel_graph_with_mapping(kg1_graph, label_mapping_g1)
    relabeled_kg2 = relabel_graph_with_mapping(kg2_graph, label_mapping_g2)

    kg1_grakel = convert_to_grakel_graph(relabeled_kg1)
    kg2_grakel = convert_to_grakel_graph(relabeled_kg2)

    if kg1_grakel is None or kg2_grakel is None:
        # No edges — fall back to SBERT only (clipped to [0, 1])
        sbert_score = float(np.clip(_sbert_mean_cosine(kg1_triples, filtered_kg2), 0.0, 1.0))
        return sbert_score, {
            'wl_score': None,
            'sbert_score': sbert_score,
            'blended_score': sbert_score,
            'note': 'WL skipped (empty graph) — SBERT only',
        }

    try:
        wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True)
        kernel_matrix = wl_kernel.fit_transform([kg1_grakel, kg2_grakel])
        wl_score = float(kernel_matrix[0, 1])
    except Exception as e:
        wl_score = 0.0

    # -- Step 4: SBERT mean-pool cosine similarity --
    sbert_score         = _sbert_mean_cosine(kg1_triples, filtered_kg2)
    sbert_score_clipped = float(np.clip(sbert_score, 0.0, 1.0))

    # -- Step 5: Blend — score = α * wl_score + (1 - α) * sbert_score_clipped --
    similarity = float(alpha * wl_score + (1.0 - alpha) * sbert_score_clipped)

    debug_info = {
        'wl_score': wl_score,
        'sbert_score_raw': sbert_score,
        'sbert_score_clipped': sbert_score_clipped,
        'blended_score': similarity,
        'alpha': alpha,
        'kg1_nodes': len(kg1_graph.nodes()),
        'kg2_nodes': len(kg2_graph.nodes()),
        'kg1_edges': len(kg1_graph.edges()),
        'kg2_edges': len(kg2_graph.edges()),
        'filtered_triples': len(filtered_kg2),
    }

    return similarity, debug_info


def calculate_snea_sbert_similarity_score(kg1_triples, kg2_triples):
    """
    Drop-in replacement for calculate_aa_kea_similarity() in aa_kea.py.
    Returns just the blended similarity score as a float in [0, 1].
    """
    score, _ = calculate_snea_sbert_similarity(kg1_triples, kg2_triples)
    return score


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('SNEA-SBERT — SNEA + SBERT Blend (alpha=0.5)')
    print('=' * 60)

    kg1 = [
        ['Marie Curie', 'discovered', 'Radium'],
        ['Marie Curie', 'won', 'Nobel Prize in Physics'],
        ['Marie Curie', 'won', 'Nobel Prize in Chemistry'],
    ]

    # Test 1: Similar graphs, different wording
    kg2_similar = [
        ['Marie Curie', 'found', 'Radium'],
        ['Marie Curie', 'received', 'Nobel Prize in Physics'],
        ['Marie Curie', 'was awarded', 'Nobel Prize in Chemistry'],
    ]
    sim, info = calculate_snea_sbert_similarity(kg1, kg2_similar)
    print(f'\nTest 1 — Similar graphs, different wording')
    print(f'  WL score    : {info["wl_score"]:.4f}')
    print(f'  SBERT score : {info["sbert_score"]:.4f}')
    print(f'  Blended     : {sim:.4f}')

    # Test 2: Different graphs
    kg3_different = [
        ['Albert Einstein', 'developed', 'Theory of Relativity'],
        ['Albert Einstein', 'won', 'Nobel Prize in Physics'],
    ]
    sim2, info2 = calculate_snea_sbert_similarity(kg1, kg3_different)
    print(f'\nTest 2 — Different graphs')
    print(f'  WL score    : {info2["wl_score"]:.4f}')
    print(f'  SBERT score : {info2["sbert_score"]:.4f}')
    print(f'  Blended     : {sim2:.4f}')

    # Test 3: Identical graphs
    sim3, info3 = calculate_snea_sbert_similarity(kg1, kg1)
    print(f'\nTest 3 — Identical graphs')
    print(f'  WL score    : {info3["wl_score"]:.4f}')
    print(f'  SBERT score : {info3["sbert_score"]:.4f}')
    print(f'  Blended     : {sim3:.4f}')

    print('\n' + '=' * 60)
    print('Tests complete.')
    print('=' * 60)
