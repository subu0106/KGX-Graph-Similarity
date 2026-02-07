"""
KEA (Knowledge Graph Entity Alignment) method
Semantic clustering + Weisfeiler-Lehman kernel approach
Enhanced with Gaussian kernel and Composite similarity
"""
from .comparison import (
    calculate_similarity,
    calculate_composite_similarity,
    calculate_gaussian_feature_similarity
)

__all__ = [
    'calculate_similarity',
    'calculate_composite_similarity',
    'calculate_gaussian_feature_similarity'
]
