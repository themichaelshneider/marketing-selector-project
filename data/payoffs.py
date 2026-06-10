"""Построение платёжных матриц для биматричной игры на основе данных каналов."""

import logging
from functools import lru_cache

import numpy as np
from core.models import ChannelData

logger = logging.getLogger("marketings.payoffs")


@lru_cache(maxsize=16)
def _build_payoff_matrix_impl(
    channels_tuple: tuple,
    budget: float,
    margin: float,
    conversion: float,
    market_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Внутренняя функция с кэшированием (hashable-аргументы)."""
    channels = [
        ChannelData(name=t[0], reach=t[1], cac=t[2], relevance=t[3])
        for t in channels_tuple
    ]

    n = len(channels)
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                reach_1 = channels[i].reach / 2.0
            else:
                reach_1 = channels[i].reach

            A[i, j] = (
                (reach_1 / 100.0)
                * conversion
                * margin
                * market_size
                - budget
            )

            if i == j:
                reach_2 = channels[j].reach / 2.0
            else:
                reach_2 = channels[j].reach

            B[i, j] = (
                (reach_2 / 100.0)
                * conversion
                * margin
                * market_size
                - budget
            )

    return A, B


def build_payoff_matrix(
    channels: list[ChannelData],
    budget: float,
    margin: float,
    conversion: float,
    market_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит платёжные матрицы A и B для двух игроков."""
    channels_key = tuple(
        (ch.name, ch.reach, ch.cac, ch.relevance) for ch in channels
    )

    logger.debug("build_payoff_matrix вызван: %d каналов, budget=%.0f",
                 len(channels), budget)
    return _build_payoff_matrix_impl(
        channels_key, budget, margin, conversion, market_size,
    )
