"""Ранжирование рекламных каналов методом Аналитических Иерархий (AHP).

Реализует метод Саати:
  1. Построение матриц парных сравнений для каждого критерия
  2. Расчёт вектора приоритетов (собственный вектор)
"""

import logging
from functools import lru_cache

import numpy as np
from core.models import ChannelData, CriterionDef, ChannelRating

logger = logging.getLogger("marketings.ahp")

def _build_pairwise_matrix(values: np.ndarray) -> np.ndarray:
    n = len(values)
    P = np.ones((n, n))
    eps = 1e-12
    for i in range(n):
        for j in range(i + 1, n):
            vi = values[i] + eps
            vj = values[j] + eps
            ratio = vi / vj
            P[i, j] = ratio
            P[j, i] = 1.0 / ratio
    return P


def _priority_vector(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    lambda_max = np.max(np.real(eigenvalues))
    idx = np.argmax(np.real(eigenvalues))
    principal = np.real(eigenvectors[:, idx])
    return principal / principal.sum()


@lru_cache(maxsize=16)
def _rate_channels_impl(
    channels_tuple: tuple,
    criteria_tuple: tuple,
    weights_tuple: tuple,
) -> list[ChannelRating]:
    channels = [
        ChannelData(name=t[0], reach=t[1], cac=t[2], relevance=t[3])
        for t in channels_tuple
    ]
    criteria = [
        CriterionDef(name=t[0], key=t[1], higher_is_better=t[2])
        for t in criteria_tuple
    ]
    weights = np.array(list(weights_tuple))

    n_alts = len(channels)
    n_crit = len(criteria)

    priority_vectors = np.zeros((n_alts, n_crit))

    for c_idx, criterion in enumerate(criteria):
        raw = np.array([getattr(ch, criterion.key) for ch in channels], dtype=float)

        if not criterion.higher_is_better:
            raw = 1.0 / raw if raw.max() != raw.min() else np.ones_like(raw)

        P = _build_pairwise_matrix(raw)
        priority_vectors[:, c_idx] = _priority_vector(P)

    scores = priority_vectors @ weights
    scores /= scores.sum()

    rankings = np.argsort(-scores)

    ratings = []
    for rank_pos, idx in enumerate(rankings):
        ratings.append(ChannelRating(
            name=channels[idx].name,
            score=float(scores[idx]),
            rank=rank_pos + 1,
        ))

    weights_norm = weights / weights.sum()
    logger.info(
        "AHP-ранжирование завершено: %d каналов, веса=%s, лидер=%s",
        n_alts,
        [round(w, 3) for w in weights_norm],
        ratings[0].name,
    )

    return ratings


def rate_channels(
    channels: list[ChannelData],
    criteria: list[CriterionDef],
    weights: list[float],
) -> list[ChannelRating]:
    channels_key = tuple(
        (ch.name, ch.reach, ch.cac, ch.relevance) for ch in channels
    )
    criteria_key = tuple(
        (c.name, c.key, c.higher_is_better) for c in criteria
    )
    weights_key = tuple(weights)

    logger.debug("rate_channels вызван: %d каналов, %d критериев",
                 len(channels), len(criteria))
    return _rate_channels_impl(channels_key, criteria_key, weights_key)
