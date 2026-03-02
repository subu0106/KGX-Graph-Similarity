#!/usr/bin/env python3
"""
Semantic Weisfeiler-Lehman (WL) Kernel for Knowledge Graph Similarity

Implements a semantic variant of the WL graph kernel that uses SBERT embeddings
instead of integer hashing for node relabelling.

Key differences from the standard WL kernel (snea.py / aa_kea.py):
- Directed graph (DiGraph) instead of undirected
- WL relabelling embeds the growing neighbourhood description with SBERT
- Convergence via Agglomerative Clustering stability check instead of
  a fixed iteration count
- Multiple graph pooling strategies (mean, max, degree-weighted, attention)
- Edge (relation) embeddings blended into the final graph vector
- Optional cross-graph node alignment via the Hungarian algorithm

Pipeline:
1. Build directed graph from triples
2. Initialise node labels / embeddings (SBERT of entity name)
3. WL relabelling iterations — embed neighbourhood text with SBERT
4. Node grouping convergence check (Agglomerative Clustering, cosine dist)
5. Graph embedding construction (pooling + optional edge blend)
6. Cosine similarity ± cross-graph alignment

Author: Chamath
"""

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Set, Optional

# ── SBERT model setup ──────────────────────────────────────────────────────────

SBERT_MODELS = {
    'default':      'all-MiniLM-L6-v2',
    'accurate':     'all-mpnet-base-v2',
    'biomedical':   'pritamdeka/S-PubMedBert-MS-MARCO',
}

MODEL_DIMENSIONS = {
    'all-MiniLM-L6-v2':                         384,
    'all-mpnet-base-v2':                         768,
    'pritamdeka/S-PubMedBert-MS-MARCO':          768,
    'paraphrase-multilingual-MiniLM-L12-v2':     384,
}

_current_model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer(_current_model_name)

_embedding_cache: Dict[str, np.ndarray] = {}
_cache_enabled = True


# ── model / cache helpers ──────────────────────────────────────────────────────

def set_sbert_model(model_name: str = 'default') -> None:
    global sbert_model, _current_model_name, _embedding_cache
    if model_name in SBERT_MODELS:
        model_name = SBERT_MODELS[model_name]
    _current_model_name = model_name
    sbert_model = SentenceTransformer(model_name)
    _embedding_cache = {}


def get_embedding_dim() -> int:
    return MODEL_DIMENSIONS.get(_current_model_name, 384)


def clear_embedding_cache() -> None:
    global _embedding_cache
    _embedding_cache = {}


def set_cache_enabled(enabled: bool) -> None:
    global _cache_enabled
    _cache_enabled = enabled


# ── embedding helpers ──────────────────────────────────────────────────────────

def get_sbert_embedding(text: str, use_cache: bool = True) -> np.ndarray:
    global _embedding_cache
    if use_cache and _cache_enabled and text in _embedding_cache:
        return _embedding_cache[text]
    embedding = sbert_model.encode(text, convert_to_tensor=False)
    if use_cache and _cache_enabled:
        _embedding_cache[text] = embedding
    return embedding


def get_batch_embeddings(texts: List[str], use_cache: bool = True) -> Dict[str, np.ndarray]:
    global _embedding_cache
    result = {}
    texts_to_encode = []
    for text in texts:
        if use_cache and _cache_enabled and text in _embedding_cache:
            result[text] = _embedding_cache[text]
        else:
            texts_to_encode.append(text)
    if texts_to_encode:
        embeddings = sbert_model.encode(
            texts_to_encode, convert_to_tensor=False,
            batch_size=32, show_progress_bar=False
        )
        for text, emb in zip(texts_to_encode, embeddings):
            result[text] = emb
            if use_cache and _cache_enabled:
                _embedding_cache[text] = emb
    return result


def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


# ── graph construction ─────────────────────────────────────────────────────────

def build_directed_graph(triplets: List[List[str]]) -> nx.DiGraph:
    """Step 1: Build a directed graph from (subject, predicate, object) triples."""
    G = nx.DiGraph()
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        subject, predicate, obj = triplet
        G.add_node(subject)
        G.add_node(obj)
        G.add_edge(subject, obj, relation=predicate)
    return G


# ── WL relabelling ─────────────────────────────────────────────────────────────

def initialize_node_labels(G: nx.DiGraph) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Step 2: Assign each node its entity name as initial label; embed with SBERT."""
    node_labels = {node: str(node) for node in G.nodes()}
    embeddings_dict = get_batch_embeddings(list(node_labels.values()))
    node_embeddings = {node: embeddings_dict[label] for node, label in node_labels.items()}
    return node_labels, node_embeddings


def get_neighbor_descriptions(G: nx.DiGraph, node: str, node_labels: Dict[str, str]) -> List[str]:
    """Collect semantic neighbourhood descriptions for a node (in + out edges)."""
    descriptions = []
    current_label = node_labels[node]
    for _, neighbor, data in G.out_edges(node, data=True):
        relation = data.get('relation', 'related_to')
        desc = f"{current_label} --{relation}--> {node_labels[neighbor]}"
        descriptions.append(desc)
    for neighbor, _, data in G.in_edges(node, data=True):
        relation = data.get('relation', 'related_to')
        desc = f"{node_labels[neighbor]} --{relation}--> {current_label}"
        descriptions.append(desc)
    return sorted(descriptions)


def wl_weighted_aggregation(G: nx.DiGraph,
                             node_labels: Dict[str, str],
                             node_embeddings: Dict[str, np.ndarray],
                             self_weight: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Weighted neighbour aggregation: aggregate neighbour embeddings directly
    instead of encoding concatenated strings.
    """
    all_relations = {data.get('relation', 'related_to') for _, _, data in G.edges(data=True)}
    relation_embeddings = get_batch_embeddings(list(all_relations))
    new_embeddings = {}
    for node in G.nodes():
        current_emb = node_embeddings[node]
        neighbor_embs = []
        for _, neighbor, data in G.out_edges(node, data=True):
            rel_emb = relation_embeddings[data.get('relation', 'related_to')]
            neighbor_embs.append((rel_emb + node_embeddings[neighbor]) / 2)
        for neighbor, _, data in G.in_edges(node, data=True):
            rel_emb = relation_embeddings[data.get('relation', 'related_to')]
            neighbor_embs.append((rel_emb + node_embeddings[neighbor]) / 2)
        if neighbor_embs:
            neighbor_agg = np.mean(neighbor_embs, axis=0)
            new_embeddings[node] = self_weight * current_emb + (1 - self_weight) * neighbor_agg
        else:
            new_embeddings[node] = current_emb
    return new_embeddings


def wl_relabel_iteration(G: nx.DiGraph,
                          node_labels: Dict[str, str],
                          node_embeddings: Dict[str, np.ndarray],
                          use_weighted_aggregation: bool = False) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Step 3: One WL relabelling iteration — update labels by incorporating neighbourhood."""
    new_labels = {}
    for node in G.nodes():
        neighbor_descs = get_neighbor_descriptions(G, node, node_labels)
        if neighbor_descs:
            new_labels[node] = node_labels[node] + " | " + " | ".join(neighbor_descs)
        else:
            new_labels[node] = node_labels[node]

    if use_weighted_aggregation:
        new_embeddings = wl_weighted_aggregation(G, node_labels, node_embeddings)
    else:
        embeddings_dict = get_batch_embeddings(list(new_labels.values()))
        new_embeddings = {node: embeddings_dict[label] for node, label in new_labels.items()}

    return new_labels, new_embeddings


# ── node grouping ──────────────────────────────────────────────────────────────

def compute_node_groups(node_embeddings: Dict[str, np.ndarray],
                         similarity_threshold: float = 0.90) -> Dict[str, int]:
    """Step 4: Group nodes via Agglomerative Clustering on cosine distance."""
    nodes = list(node_embeddings.keys())
    if len(nodes) == 0:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: 0}
    embeddings = np.vstack([node_embeddings[node] for node in nodes])
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    cluster_labels = clustering.fit_predict(embeddings)
    return {node: int(label) for node, label in zip(nodes, cluster_labels)}


def groups_are_stable(prev_groups: Dict[str, int], curr_groups: Dict[str, int]) -> bool:
    """Check whether node group partitioning has stabilised between iterations."""
    if set(prev_groups.keys()) != set(curr_groups.keys()):
        return False

    def get_partition(groups: Dict[str, int]) -> Set[frozenset]:
        partition: Dict[int, set] = {}
        for node, gid in groups.items():
            partition.setdefault(gid, set()).add(node)
        return {frozenset(s) for s in partition.values()}

    return get_partition(prev_groups) == get_partition(curr_groups)


def semantic_wl_iterations(G: nx.DiGraph,
                            max_iterations: int = 5,
                            similarity_threshold: float = 0.90,
                            use_weighted_aggregation: bool = False,
                            verbose: bool = False) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Run Semantic WL iterations until group structure stabilises or max_iterations."""
    if len(G.nodes()) == 0:
        return {}, {}
    node_labels, node_embeddings = initialize_node_labels(G)
    prev_groups = compute_node_groups(node_embeddings, similarity_threshold)
    for iteration in range(max_iterations):
        node_labels, node_embeddings = wl_relabel_iteration(
            G, node_labels, node_embeddings, use_weighted_aggregation
        )
        curr_groups = compute_node_groups(node_embeddings, similarity_threshold)
        if verbose:
            print(f"  Iteration {iteration + 1}: groups={curr_groups}")
        if groups_are_stable(prev_groups, curr_groups):
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break
        prev_groups = curr_groups
    return node_labels, node_embeddings


# ── graph embedding ────────────────────────────────────────────────────────────

def compute_graph_embedding(node_embeddings: Dict[str, np.ndarray],
                             G: nx.DiGraph = None,
                             pooling: str = 'mean') -> np.ndarray:
    """Step 5: Pool node embeddings into a single graph vector."""
    dim = get_embedding_dim()
    if not node_embeddings:
        return np.zeros(dim)
    nodes = list(node_embeddings.keys())
    embeddings = np.vstack([node_embeddings[n] for n in nodes])
    if pooling == 'max':
        return np.max(embeddings, axis=0)
    elif pooling == 'weighted' and G is not None:
        centrality = nx.degree_centrality(G)
        weights = np.array([centrality.get(node, 1.0) for node in nodes])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        return np.average(embeddings, axis=0, weights=weights)
    elif pooling == 'attention':
        query = np.mean(embeddings, axis=0, keepdims=True)
        attn = sklearn_cosine_similarity(query, embeddings)[0]
        attn_weights = np.exp(attn) / np.sum(np.exp(attn))
        return np.average(embeddings, axis=0, weights=attn_weights)
    else:  # mean (default)
        return np.mean(embeddings, axis=0)


def compute_edge_embedding(G: nx.DiGraph) -> np.ndarray:
    """Compute mean embedding over all relation (predicate) labels."""
    dim = get_embedding_dim()
    relations = [data.get('relation', 'related_to') for _, _, data in G.edges(data=True)]
    if not relations:
        return np.zeros(dim)
    rel_embs = get_batch_embeddings(relations)
    return np.mean(np.vstack([rel_embs[r] for r in relations]), axis=0)


def compute_combined_graph_embedding(G: nx.DiGraph,
                                      node_embeddings: Dict[str, np.ndarray],
                                      node_weight: float = 0.7,
                                      pooling: str = 'weighted') -> np.ndarray:
    """Blend node pooling with edge (relation) embedding."""
    node_emb = compute_graph_embedding(node_embeddings, G, pooling)
    edge_emb = compute_edge_embedding(G)
    return node_weight * node_emb + (1 - node_weight) * edge_emb


# ── cross-graph alignment ──────────────────────────────────────────────────────

def compute_cross_graph_alignment_similarity(emb1_dict: Dict[str, np.ndarray],
                                              emb2_dict: Dict[str, np.ndarray],
                                              method: str = 'hungarian') -> float:
    """Optimal node alignment between two graphs (Hungarian or greedy)."""
    if not emb1_dict or not emb2_dict:
        return 0.0
    nodes1, nodes2 = list(emb1_dict.keys()), list(emb2_dict.keys())
    emb1 = np.vstack([emb1_dict[n] for n in nodes1])
    emb2 = np.vstack([emb2_dict[n] for n in nodes2])
    sim_matrix = sklearn_cosine_similarity(emb1, emb2)
    if method == 'hungarian':
        row_ind, col_ind = linear_sum_assignment(1 - sim_matrix)
        return float(np.mean(sim_matrix[row_ind, col_ind]))
    elif method == 'greedy':
        matched, used = [], set()
        source = range(len(nodes1)) if len(nodes1) <= len(nodes2) else range(len(nodes2))
        axis = 1 if len(nodes1) <= len(nodes2) else 0
        for i in source:
            row = sim_matrix[i] if axis == 1 else sim_matrix[:, i]
            for j in np.argsort(-row):
                if j not in used:
                    matched.append(row[j])
                    used.add(j)
                    break
        return float(np.mean(matched)) if matched else 0.0
    else:
        return float(np.mean(np.max(sim_matrix, axis=1)))


# ── preset configurations ──────────────────────────────────────────────────────

def get_preset_config(preset: str = 'balanced') -> Dict:
    """
    Return a configuration dict for the similarity function.

    Presets:
      fast     — minimal computation, suitable for large datasets
      balanced — good speed/accuracy trade-off (default)
      accurate — maximum accuracy, slower
      legacy   — original basic behaviour
    """
    presets = {
        'fast': {
            'max_iterations': 3,
            'pooling': 'mean',
            'include_edges': False,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': False,
        },
        'balanced': {
            'max_iterations': 5,
            'pooling': 'weighted',
            'include_edges': True,
            'node_weight': 0.7,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': True,
            'alignment_method': 'greedy',
            'similarity_combination': 'weighted',
        },
        'accurate': {
            'max_iterations': 7,
            'pooling': 'attention',
            'include_edges': True,
            'node_weight': 0.7,
            'use_weighted_aggregation': True,
            'use_cross_graph_alignment': True,
            'alignment_method': 'hungarian',
            'similarity_combination': 'weighted',
        },
        'legacy': {
            'max_iterations': 5,
            'pooling': 'mean',
            'include_edges': False,
            'use_weighted_aggregation': False,
            'use_cross_graph_alignment': False,
        },
    }
    return presets.get(preset, presets['balanced'])


# ── full configurable pipeline ─────────────────────────────────────────────────

def semantic_wl_kernel_similarity(kg1_triplets: List[List[str]],
                                   kg2_triplets: List[List[str]],
                                   max_iterations: int = 5,
                                   similarity_threshold: float = 0.90,
                                   pooling: str = 'weighted',
                                   include_edges: bool = True,
                                   node_weight: float = 0.7,
                                   use_weighted_aggregation: bool = False,
                                   use_cross_graph_alignment: bool = False,
                                   alignment_method: str = 'hungarian',
                                   similarity_combination: str = 'mean',
                                   verbose: bool = False) -> Dict:
    """
    Full configurable Semantic WL Kernel similarity pipeline.

    Returns a dict with keys:
      similarity, embedding_similarity, graph1_labels, graph2_labels,
      graph1_embedding, graph2_embedding, [alignment_similarity]
    """
    kg1_triplets = [t for t in kg1_triplets if len(t) == 3]
    kg2_triplets = [t for t in kg2_triplets if len(t) == 3]
    dim = get_embedding_dim()

    if not kg1_triplets or not kg2_triplets:
        result = {
            'similarity': 0.0,
            'embedding_similarity': 0.0,
            'graph1_labels': {},
            'graph2_labels': {},
            'graph1_embedding': np.zeros(dim),
            'graph2_embedding': np.zeros(dim),
        }
        if use_cross_graph_alignment:
            result['alignment_similarity'] = 0.0
        return result

    G1 = build_directed_graph(kg1_triplets)
    G2 = build_directed_graph(kg2_triplets)

    labels1, embeddings1 = semantic_wl_iterations(
        G1, max_iterations, similarity_threshold, use_weighted_aggregation, verbose
    )
    labels2, embeddings2 = semantic_wl_iterations(
        G2, max_iterations, similarity_threshold, use_weighted_aggregation, verbose
    )

    if include_edges:
        graph1_embedding = compute_combined_graph_embedding(G1, embeddings1, node_weight, pooling)
        graph2_embedding = compute_combined_graph_embedding(G2, embeddings2, node_weight, pooling)
    else:
        graph1_embedding = compute_graph_embedding(embeddings1, G1, pooling)
        graph2_embedding = compute_graph_embedding(embeddings2, G2, pooling)

    embedding_similarity = cosine_sim(graph1_embedding, graph2_embedding)

    alignment_similarity = None
    if use_cross_graph_alignment:
        alignment_similarity = compute_cross_graph_alignment_similarity(
            embeddings1, embeddings2, alignment_method
        )
        if similarity_combination == 'max':
            similarity = max(embedding_similarity, alignment_similarity)
        elif similarity_combination == 'weighted':
            similarity = 0.6 * embedding_similarity + 0.4 * alignment_similarity
        else:  # mean
            similarity = (embedding_similarity + alignment_similarity) / 2
    else:
        similarity = embedding_similarity

    result = {
        'similarity': similarity,
        'embedding_similarity': embedding_similarity,
        'graph1_labels': labels1,
        'graph2_labels': labels2,
        'graph1_embedding': graph1_embedding,
        'graph2_embedding': graph2_embedding,
    }
    if alignment_similarity is not None:
        result['alignment_similarity'] = alignment_similarity
    return result


def semantic_wl_kernel_nodewise_similarity(kg1_triplets: List[List[str]],
                                            kg2_triplets: List[List[str]],
                                            max_iterations: int = 5,
                                            similarity_threshold: float = 0.90,
                                            use_weighted_aggregation: bool = False,
                                            verbose: bool = False) -> Dict:
    """
    Compare node group embeddings pairwise (no global pooling).

    For every group ID present in either graph, compute the mean embedding
    from that graph's nodes in that group; then take the cosine similarity
    between the two mean embeddings and average across all groups.
    """
    G1 = build_directed_graph(kg1_triplets)
    G2 = build_directed_graph(kg2_triplets)

    labels1, embeddings1 = semantic_wl_iterations(G1, max_iterations, similarity_threshold, use_weighted_aggregation, verbose)
    labels2, embeddings2 = semantic_wl_iterations(G2, max_iterations, similarity_threshold, use_weighted_aggregation, verbose)

    groups1 = compute_node_groups(embeddings1, similarity_threshold)
    groups2 = compute_node_groups(embeddings2, similarity_threshold)

    all_group_ids = sorted(set(groups1.values()) | set(groups2.values()))
    dim = get_embedding_dim()

    def group_mean_embedding(embeddings, groups, group_id):
        nodes = [n for n, gid in groups.items() if gid == group_id]
        if not nodes:
            return np.zeros(dim)
        return np.mean([embeddings[n] for n in nodes], axis=0)

    similarities = [
        cosine_sim(
            group_mean_embedding(embeddings1, groups1, gid),
            group_mean_embedding(embeddings2, groups2, gid)
        )
        for gid in all_group_ids
    ]

    return {
        'similarity': float(np.mean(similarities)) if similarities else 0.0,
        'groupwise_similarities': similarities,
        'group_ids': all_group_ids,
        'graph1_labels': labels1,
        'graph2_labels': labels2,
    }


# ── public API (matches pattern of other Methods files) ───────────────────────

def calculate_semantic_wl_similarity(kg1_triples: List[List[str]],
                                      kg2_triples: List[List[str]]) -> Tuple[float, Dict]:
    """
    Calculate Semantic WL Kernel similarity between two KGs.

    Uses the balanced preset (weighted pooling, edge blend, greedy cross-graph
    alignment) as default. Returns (similarity, debug_info).

    Args:
        kg1_triples: list of [subject, predicate, object]
        kg2_triples: list of [subject, predicate, object]

    Returns:
        similarity : float in [0, 1]
        debug_info : dict with embedding_similarity, alignment_similarity, etc.
    """
    config = get_preset_config('balanced')
    result = semantic_wl_kernel_similarity(kg1_triples, kg2_triples, **config)
    debug_info = {k: v for k, v in result.items()
                  if k not in ('graph1_embedding', 'graph2_embedding',
                               'graph1_labels', 'graph2_labels')}
    return result['similarity'], debug_info


def calculate_semantic_wl_similarity_score(kg1_triples: List[List[str]],
                                            kg2_triples: List[List[str]]) -> float:
    """Wrapper returning only the similarity score (no debug info)."""
    similarity, _ = calculate_semantic_wl_similarity(kg1_triples, kg2_triples)
    return similarity


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    kg1 = [
        ['Marie Curie', 'discovered', 'Radium'],
        ['Marie Curie', 'won', 'Nobel Prize in Physics'],
        ['Marie Curie', 'won', 'Nobel Prize in Chemistry'],
    ]
    kg2_similar = [
        ['Marie Curie', 'found', 'Radium'],
        ['Marie Curie', 'received', 'Nobel Prize in Physics'],
        ['Marie Curie', 'was awarded', 'Nobel Prize in Chemistry'],
    ]
    kg3_different = [
        ['Albert Einstein', 'developed', 'Theory of Relativity'],
        ['Albert Einstein', 'won', 'Nobel Prize in Physics'],
    ]

    print('=' * 60)
    print('Semantic WL Kernel — smoke test')
    print('=' * 60)

    for name, kg_b in [('Similar (different wording)', kg2_similar),
                        ('Different graphs', kg3_different),
                        ('Identical graphs', kg1)]:
        sim, info = calculate_semantic_wl_similarity(kg1, kg_b)
        print(f'\n{name}')
        print(f'  similarity          : {sim:.4f}')
        print(f'  embedding_similarity: {info.get("embedding_similarity", "N/A"):.4f}')
        if 'alignment_similarity' in info:
            print(f'  alignment_similarity: {info["alignment_similarity"]:.4f}')

    print('\n' + '=' * 60)
    print('Testing complete.')
    print('=' * 60)
