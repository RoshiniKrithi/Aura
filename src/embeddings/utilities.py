"""Embedding Vector Space Utilities for Aura LLM Pipeline.

Provides pairwise cosine similarity computation, vector normalization,
and top-k nearest neighbors retrieval across token embedding space.
"""

import logging
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F

from src.embeddings.embedding_layer import EmbeddingLayer

logger = logging.getLogger(__name__)


class EmbeddingUtilities:
    """Vector mathematical utility functions for inspecting token embedding representations.

    Design Decisions:
        - Vectorized PyTorch operations for fast cosine similarity and top-k retrieval.
        - Supports query by token ID (int) or target vector (Tensor).

    Time Complexity:
        O(V * d) for computing cosine similarity across entire vocabulary of size V.

    Space Complexity:
        O(V) to hold similarities array.
    """

    @staticmethod
    def compute_cosine_similarity(
        vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """Computes cosine similarity between two 1D vector representations: sim = (a . b) / (||a|| * ||b||).

        Args:
            vec_a: Tensor vector A of shape (d,).
            vec_b: Tensor vector B of shape (d,).
            eps: Epsilon parameter for numerical stability.

        Returns:
            Cosine similarity float value in range [-1.0, 1.0].
        """
        if vec_a.ndim != 1 or vec_b.ndim != 1:
            raise ValueError(
                f"Expected 1D vectors for cosine similarity, got vec_a: {tuple(vec_a.shape)}, vec_b: {tuple(vec_b.shape)}"
            )

        sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), eps=eps)
        return float(sim.item())

    @classmethod
    def get_token_similarity(
        cls, layer: EmbeddingLayer, token_id_a: int, token_id_b: int
    ) -> float:
        """Calculates cosine similarity between two token IDs in embedding space.

        Args:
            layer: EmbeddingLayer instance.
            token_id_a: First token index.
            token_id_b: Second token index.

        Returns:
            Cosine similarity float value.
        """
        weight = layer.weight.detach()
        vec_a = weight[token_id_a]
        vec_b = weight[token_id_b]
        return cls.compute_cosine_similarity(vec_a, vec_b)

    @classmethod
    def top_k_nearest_neighbors(
        cls,
        layer: EmbeddingLayer,
        query: Union[int, torch.Tensor],
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> List[Tuple[int, float]]:
        """Retrieves top-k nearest neighbor tokens in embedding space based on cosine similarity.

        Args:
            layer: EmbeddingLayer module instance.
            query: Integer token ID OR 1D query vector tensor of shape (d_model,).
            top_k: Number of nearest neighbors to return.
            exclude_self: If True, excludes the query token itself from results.

        Returns:
            List of tuples: [(token_id, similarity_score), ...].
        """
        weight = layer.weight.detach()  # Shape: (V, d)

        if isinstance(query, int):
            query_id = query
            query_vec = weight[query_id]  # Shape: (d,)
        else:
            query_id = None
            query_vec = query.detach()
            if query_vec.ndim > 1:
                query_vec = query_vec.squeeze(0)

        # Normalize query and weight matrix for fast cosine similarity: sim = (W_norm @ q_norm)
        weight_norm = F.normalize(weight, p=2, dim=1)  # (V, d)
        query_norm = F.normalize(query_vec, p=2, dim=0)   # (d,)

        similarities = torch.mv(weight_norm, query_norm)  # (V,)

        # Top k retrieval
        k_fetch = top_k + (1 if exclude_self and query_id is not None else 0)
        top_scores, top_indices = torch.topk(similarities, k=min(k_fetch, weight.size(0)))

        results: List[Tuple[int, float]] = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if exclude_self and query_id is not None and idx == query_id:
                continue
            results.append((int(idx), float(score)))
            if len(results) >= top_k:
                break

        return results
