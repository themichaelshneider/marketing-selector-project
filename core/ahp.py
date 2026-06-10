"""Ранжирование рекламных каналов методом взвешенной суммы (АИИ).

Использует scikit-criteria для построения матрицы решений и
вычисления интегральных оценок альтернатив.
"""

import logging
from functools import lru_cache

import numpy as np
from skcriteria import mkdm, Objective
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.preprocessing.scalers import SumScaler

from core.models import ChannelData, CriterionDef, ChannelRating

logger = logging.getLogger("marketings.ahp")


@lru_cache(maxsize=16)
def _rate_channels_impl(
    channels_tuple: tuple,
    criteria_tuple: tuple,
    weights_tuple: tuple,
) -> list[ChannelRating]:
    """Внутренняя функция с кэшированием (hashable-аргументы)."""
    channels = [
        ChannelData(name=t[0], reach=t[1], cac=t[2], relevance=t[3])
        for t in channels_tuple
    ]
    criteria = [
        CriterionDef(name=t[0], key=t[1], higher_is_better=t[2])
        for t in criteria_tuple
    ]
    weights = list(weights_tuple)

    matrix = np.array([
        [getattr(ch, c.key) for c in criteria]
        for ch in channels
    ], dtype=float)

    objectives = []
    for c in criteria:
        objectives.append(Objective.MAX if c.higher_is_better else Objective.MIN)

    inverted = matrix.copy()
    for col_idx, obj in enumerate(objectives):
        if obj == Objective.MIN:
            col = inverted[:, col_idx]
            col_max = np.max(col)
            col_min = np.min(col)
            if col_max != col_min:
                inverted[:, col_idx] = col_max - col
            else:
                inverted[:, col_idx] = 1.0

    objectives_max = [Objective.MAX] * len(objectives)

    dm = mkdm(inverted, objectives_max, weights=weights)

    scaler = SumScaler("matrix")
    dm_scaled = scaler.transform(dm)

    wsm = WeightedSumModel()
    result = wsm.evaluate(dm_scaled)

    raw_scores = result.e_["score"]
    scores = np.atleast_1d(raw_scores)

    ranks = np.atleast_1d(result.rank_)
    ratings = []
    for i in range(len(channels)):
        ratings.append(ChannelRating(
            name=channels[i].name,
            score=float(scores[i]),
            rank=int(ranks[i]),
        ))

    ratings.sort(key=lambda r: r.rank)
    return ratings


def rate_channels(
    channels: list[ChannelData],
    criteria: list[CriterionDef],
    weights: list[float],
) -> list[ChannelRating]:
    """Ранжирует каналы по взвешенной сумме нормализованных критериев."""
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
