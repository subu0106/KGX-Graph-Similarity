#!/usr/bin/env python3
"""
GNN-based KG Similarity (Option A — GraphCheck-inspired)

Adapts GraphCheck's GNN graph encoding approach for unsupervised KG
similarity scoring. No pre-trained weights or LLM required.

Pipeline:
1. Parse KG triples into a PyG Data object
   - Nodes  = unique entities (subject / object)
   - Edges  = directed triple edges (subject → object)
   - Node features (x) = SBERT embeddings of entity labels
2. Propagate through a 2-layer GAT
   - Each node accumulates neighbourhood context via attention
   - Captures both semantic content (SBERT) and local structure (GAT)
3. Mean-pool node embeddings → single graph-level vector per KG
4. Cosine similarity between the two graph vectors → score in [0, 1]

Reference: GraphCheck (ACL 2025) — graph_build.py + model/gnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SBERT_MODEL_NAME = "paraphrase-MPNet-base-v2"  # 768-D, consistent with KEA/AA-KEA
EMBED_DIM = 768
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 2
GNN_NUM_HEADS = 4
GNN_DROPOUT = 0.0          # no dropout at inference
GNN_RANDOM_SEED = 42       # fixed seed so the encoder is reproducible

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoids reloading on every call)
# ---------------------------------------------------------------------------

_sbert_model: SentenceTransformer | None = None
_gnn_encoder: "KGEncoder | None" = None


def _get_sbert() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
    return _sbert_model


def _get_encoder() -> "KGEncoder":
    global _gnn_encoder
    if _gnn_encoder is None:
        torch.manual_seed(GNN_RANDOM_SEED)
        _gnn_encoder = KGEncoder(
            in_channels=EMBED_DIM,
            hidden_channels=GNN_HIDDEN_DIM,
            out_channels=GNN_HIDDEN_DIM,
            num_heads=GNN_NUM_HEADS,
            dropout=GNN_DROPOUT,
        ).to(DEVICE)
        _gnn_encoder.eval()
    return _gnn_encoder


# ---------------------------------------------------------------------------
# GNN encoder (lightweight 2-layer GAT, same family as GraphCheck's gnn.py)
# ---------------------------------------------------------------------------

class KGEncoder(nn.Module):
    """
    2-layer Graph Attention Network that converts per-node SBERT embeddings
    into structure-aware node embeddings.

    Architecture mirrors GraphCheck's GAT (gnn.py) but is lightweight:
      Input  → GATConv(heads=4, concat=False) → BN → ReLU
             → GATConv(heads=4, concat=False)
    Mean-pooling over all nodes gives the graph-level embedding.
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=num_heads, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels,
                             heads=num_heads, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, in_channels]  — SBERT node features
            edge_index: [2, E]            — directed edges

        Returns:
            [N, out_channels] — structure-aware node embeddings
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# KG → PyG Data conversion  (adapted from GraphCheck's textualize_graph)
# ---------------------------------------------------------------------------

def _triples_to_pyg(triples: list, sbert: SentenceTransformer) -> Data | None:
    """
    Convert a list of (subject, predicate, object) triples into a PyG Data
    object ready for the GAT encoder.

    Node features x are SBERT embeddings of the entity label strings.
    Edge attributes are NOT passed to the GAT (basic GATConv ignores them);
    they are stored in data.edge_attr for optional future use.
    """
    nodes: dict[str, int] = {}
    edges_src: list[int] = []
    edges_dst: list[int] = []
    edge_labels: list[str] = []

    for triple in triples:
        if len(triple) != 3:
            continue
        subj, pred, obj = triple
        subj = str(subj).lower().strip() if subj else " "
        pred = str(pred).lower().strip() if pred else "related"
        obj  = str(obj).lower().strip()  if obj  else " "

        if subj not in nodes:
            nodes[subj] = len(nodes)
        if obj not in nodes:
            nodes[obj] = len(nodes)

        edges_src.append(nodes[subj])
        edges_dst.append(nodes[obj])
        edge_labels.append(pred)

    if not nodes:
        return None

    node_labels = list(nodes.keys())

    # Encode node labels with SBERT
    x = sbert.encode(
        node_labels, convert_to_tensor=True, device=DEVICE, show_progress_bar=False
    ).float()  # [N, 768]

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst],
                                  dtype=torch.long, device=DEVICE)
        edge_attr = sbert.encode(
            edge_labels, convert_to_tensor=True, device=DEVICE, show_progress_bar=False
        ).float()  # [E, 768]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        edge_attr  = torch.zeros((0, EMBED_DIM), dtype=torch.float, device=DEVICE)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes),
    )


# ---------------------------------------------------------------------------
# Graph-level encoding
# ---------------------------------------------------------------------------

def _encode_kg(
    triples: list,
    encoder: KGEncoder,
    sbert: SentenceTransformer,
) -> tuple[torch.Tensor | None, bool]:
    """
    Encode a KG into a single graph-level embedding vector.

    Returns:
        (graph_emb [out_channels], fallback_used)
        fallback_used=True means GNN was bypassed (e.g. single node).
    """
    data = _triples_to_pyg(triples, sbert)
    if data is None:
        return None, True

    # Single-node graphs: skip GNN (no edges to aggregate)
    if data.num_nodes == 1:
        return data.x.squeeze(0), True

    with torch.no_grad():
        node_embs = encoder(data.x, data.edge_index)  # [N, GNN_HIDDEN_DIM]
        graph_emb = node_embs.mean(dim=0)              # [GNN_HIDDEN_DIM]

    return graph_emb, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_gnn_similarity(
    kg1_triples: list,
    kg2_triples: list,
) -> float:
    """
    Compute KG similarity using a GraphCheck-inspired GNN encoder.

    Steps:
      1. Build PyG Data for each KG (SBERT node features, directed edges)
      2. Encode through a shared 2-layer GAT → structure-aware node embs
      3. Mean-pool nodes → graph-level embedding
      4. Cosine similarity, mapped to [0, 1]

    Args:
        kg1_triples: list of [subject, predicate, object] triples
        kg2_triples: list of [subject, predicate, object] triples

    Returns:
        similarity: float in [0, 1]
    """
    similarity, _ = calculate_gnn_similarity_with_info(kg1_triples, kg2_triples)
    return similarity


def calculate_gnn_similarity_with_info(
    kg1_triples: list,
    kg2_triples: list,
) -> tuple[float, dict]:
    """
    Same as calculate_gnn_similarity but also returns debug info.

    Returns:
        (similarity, debug_info)
    """
    kg1_triples = [t for t in kg1_triples if len(t) == 3]
    kg2_triples = [t for t in kg2_triples if len(t) == 3]

    if not kg1_triples or not kg2_triples:
        return 0.0, {"error": "empty triples"}

    sbert   = _get_sbert()
    encoder = _get_encoder()

    emb1, fallback1 = _encode_kg(kg1_triples, encoder, sbert)
    emb2, fallback2 = _encode_kg(kg2_triples, encoder, sbert)

    if emb1 is None or emb2 is None:
        return 0.0, {"error": "could not encode one or both KGs"}

    # Cosine similarity ∈ [-1, 1]  →  map to [0, 1]
    e1 = F.normalize(emb1.unsqueeze(0), p=2, dim=1)
    e2 = F.normalize(emb2.unsqueeze(0), p=2, dim=1)
    cosine_raw = torch.mm(e1, e2.t()).item()
    similarity = (cosine_raw + 1.0) / 2.0

    # Build PyG Data again just to count nodes/edges for debug info
    d1 = _triples_to_pyg(kg1_triples, sbert)
    d2 = _triples_to_pyg(kg2_triples, sbert)

    debug_info = {
        "kg1_nodes":      d1.num_nodes if d1 else 0,
        "kg1_edges":      d1.edge_index.shape[1] if d1 else 0,
        "kg2_nodes":      d2.num_nodes if d2 else 0,
        "kg2_edges":      d2.edge_index.shape[1] if d2 else 0,
        "cosine_raw":     round(cosine_raw, 4),
        "gnn_fallback":   fallback1 or fallback2,
        "embed_dim_in":   EMBED_DIM,
        "embed_dim_gnn":  GNN_HIDDEN_DIM,
    }

    return float(similarity), debug_info


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("GNN Similarity — smoke test")
    print("=" * 60)

    kg1 = [
        ["Marie Curie", "discovered", "Radium"],
        ["Marie Curie", "won", "Nobel Prize in Physics"],
        ["Marie Curie", "won", "Nobel Prize in Chemistry"],
    ]
    kg2 = [
        ["Marie Curie", "found", "Radium"],
        ["Marie Curie", "received", "Nobel Prize in Physics"],
        ["Marie Curie", "was awarded", "Nobel Prize in Chemistry"],
    ]
    kg3 = [
        ["Albert Einstein", "developed", "Theory of Relativity"],
        ["Albert Einstein", "won", "Nobel Prize in Physics"],
    ]

    sim, info = calculate_gnn_similarity_with_info(kg1, kg2)
    print(f"\nTest 1 (same topic, different wording): {sim:.4f}")
    print(f"  debug: {info}")

    sim, info = calculate_gnn_similarity_with_info(kg1, kg3)
    print(f"\nTest 2 (different entities):             {sim:.4f}")
    print(f"  debug: {info}")

    sim, info = calculate_gnn_similarity_with_info(kg1, kg1)
    print(f"\nTest 3 (identical graphs):               {sim:.4f}")
    print(f"  debug: {info}")

    print("\n" + "=" * 60)
