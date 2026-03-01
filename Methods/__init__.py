"""
Methods package for graph similarity calculation
"""

from .transe import TransE
from .rotate import RotatE
from .wl_kernel import calculate_pure_wl_kernel_similarity
from .graph_embeddings import GraphEmbeddingSimilarity
from .kea import (
    calculate_similarity,
    calculate_composite_similarity,
    calculate_gaussian_feature_similarity
)
from .aa_kea import (
    calculate_aa_kea_similarity,
    calculate_attention_augmented_similarity
)
from .enhance_aa_kea import (
    calculate_enhanced_aa_kea_similarity,
    calculate_enhanced_aa_kea_similarity_score
)
from .kea_bert import calculate_kea_bert_similarity
from .gnn_similarity import (
    calculate_gnn_similarity,
    calculate_gnn_similarity_with_info,
)

__all__ = [
    'TransE',
    'RotatE',
    'calculate_pure_wl_kernel_similarity',
    'GraphEmbeddingSimilarity',
    'calculate_similarity',
    'calculate_composite_similarity',
    'calculate_gaussian_feature_similarity',
    'calculate_aa_kea_similarity',
    'calculate_attention_augmented_similarity',
    'calculate_enhanced_aa_kea_similarity',
    'calculate_enhanced_aa_kea_similarity_score',
    'calculate_kea_bert_similarity',
    'calculate_gnn_similarity',
    'calculate_gnn_similarity_with_info',
]
