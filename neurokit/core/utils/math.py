"""
neurokit.core.utils.math

Mathematical utility functions for vector operations.
"""


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
        (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine distance (1 - similarity), range 0 to 2
        (0 = identical, 1 = orthogonal, 2 = opposite)
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (L2 norm of difference)
    """
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5


def normalize(vec: list[float]) -> list[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector with L2 norm = 1
    """
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]
