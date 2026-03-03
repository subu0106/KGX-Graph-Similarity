"""
Methods package for graph similarity calculation
"""

from .transe import TransE
from .rotate import RotatE
from .wl_kernel import calculate_wl_kernel_similarity
from .graph_embeddings import GraphEmbeddingSimilarity
from .kea import (
    calculate_kea_similarity,
    calculate_kea_composite_similarity,
    calculate_gaussian_feature_similarity
)
from .snea import (
    calculate_snea_similarity,
    calculate_snea_similarity_score,
)
from .aa_kea import (
    calculate_aa_kea_similarity,
    calculate_attention_augmented_similarity
)
from .snea_sbert import (
    calculate_snea_sbert_similarity,
    calculate_snea_sbert_similarity_score,
)
from .semantic_wl_kernel_chamath import (
    calculate_semantic_wl_similarity,
    calculate_semantic_wl_similarity_score,
)
from .kea_bert import (
    calculate_kea_bert_similarity,
    calculate_kea_bert_similarity_score,
)
from .gnn_similarity import (
    calculate_gnn_similarity,
    calculate_gnn_similarity_with_info,
)

__all__ = [
    'TransE',
    'RotatE',
    'calculate_wl_kernel_similarity',
    'GraphEmbeddingSimilarity',
    'calculate_kea_similarity',
    'calculate_kea_composite_similarity',
    'calculate_gaussian_feature_similarity',
    'calculate_snea_similarity',
    'calculate_snea_similarity_score',
    'calculate_aa_kea_similarity',
    'calculate_attention_augmented_similarity',
    'calculate_snea_sbert_similarity',
    'calculate_snea_sbert_similarity_score',
    'calculate_kea_bert_similarity',
    'calculate_kea_bert_similarity_score',
    'calculate_gnn_similarity',
    'calculate_gnn_similarity_with_info',
    'calculate_semantic_wl_similarity',
    'calculate_semantic_wl_similarity_score',
]
