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
    competitor_budget: float,
    competitor_margin: float,
    competition_beta: float,
    avg_revenue_per_client: float,
) -> tuple[np.ndarray, np.ndarray]:
    channels = [
        ChannelData(name=t[0], reach=t[1], cac=t[2], relevance=t[3])
        for t in channels_tuple
    ]

    n = len(channels)
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ri, rj = channels[i].reach, channels[j].reach
            cac_i, cac_j = channels[i].cac, channels[j].cac

            if i == j:
                eff_i = ri / 2.0
                eff_j = rj / 2.0
            else:
                eff_i = ri * (1 - competition_beta * rj / 100.0)
                eff_j = rj * (1 - competition_beta * ri / 100.0)

            clients_1 = eff_i / 100.0 * conversion * market_size
            clients_2 = eff_j / 100.0 * conversion * market_size

            net_per_client_1 = avg_revenue_per_client - cac_i
            net_per_client_2 = avg_revenue_per_client - cac_j

            A[i, j] = clients_1 * net_per_client_1 * margin - budget
            B[i, j] = clients_2 * net_per_client_2 * competitor_margin - competitor_budget

    return A, B


def build_payoff_matrix(
    channels: list[ChannelData],
    budget: float,
    margin: float,
    conversion: float,
    market_size: float,
    competitor_budget: float = 800000,
    competitor_margin: float = 0.30,
    competition_beta: float = 0.5,
    avg_revenue_per_client: float = 15000,
) -> tuple[np.ndarray, np.ndarray]:
    channels_key = tuple(
        (ch.name, ch.reach, ch.cac, ch.relevance) for ch in channels
    )

    logger.debug("build_payoff_matrix вызван: %d каналов, budget=%.0f",
                 len(channels), budget)
    return _build_payoff_matrix_impl(
        channels_key, budget, margin, conversion, market_size,
        competitor_budget, competitor_margin, competition_beta, avg_revenue_per_client,
    )
