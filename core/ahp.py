"""Ранжирование рекламных каналов методом взвешенной суммы (АИИ).

Использует scikit-criteria для построения матрицы решений и
вычисления интегральных оценок альтернатив.
"""

import numpy as np
from skcriteria import mkdm, Objective
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.preprocessing.scalers import SumScaler
from core.models import ChannelData, CriterionDef, ChannelRating


def rate_channels(
    channels: list[ChannelData],
    criteria: list[CriterionDef],
    weights: list[float],
) -> list[ChannelRating]:
    """Ранжирует каналы по взвешенной сумме нормализованных критериев.

    channels: список каналов с их характеристиками
    criteria: критерии оценки (с направлением MAX или MIN)
    weights:  веса критериев (должны суммироваться в 1)

    Возвращает список ChannelRating, отсортированный по рангу.
    """
    # Строим матрицу решений: строки = каналы, столбцы = критерии
    matrix = np.array([
        [getattr(ch, c.key) for c in criteria]
        for ch in channels
    ], dtype=float)

    # Определяем направления критериев
    objectives = []
    for c in criteria:
        objectives.append(Objective.MAX if c.higher_is_better else Objective.MIN)

    # WeightedSumModel работает только с MAX-направлением,
    # поэтому инвертируем MIN-критерии: x → max(col) - x
    inverted = matrix.copy()
    for col_idx, obj in enumerate(objectives):
        if obj == Objective.MIN:
            inverted[:, col_idx] = np.max(inverted[:, col_idx]) - inverted[:, col_idx]

    # Все критерии теперь MAX
    objectives_max = [Objective.MAX] * len(objectives)

    # Создаём матрицу решений scikit-criteria
    dm = mkdm(inverted, objectives_max, weights=weights)

    # Нормализация: суммарное масштабирование (деление на сумму по столбцу)
    scaler = SumScaler("matrix")
    dm_scaled = scaler.transform(dm)

    # Метод взвешенной суммы
    wsm = WeightedSumModel()
    result = wsm.evaluate(dm_scaled)

    # Извлекаем интегральные оценки (может быть массив или скаляр для 1 альтернативы)
    raw_scores = result.e_["score"]
    scores = np.atleast_1d(raw_scores)

    # Формируем результат с оценками и рангами
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
