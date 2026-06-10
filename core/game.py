"""Поиск равновесия Нэша в биматричной игре с помощью Nashpy."""

import nashpy as nash
import numpy as np
from core.models import NashResult


def solve_game(
    A: np.ndarray,
    B: np.ndarray,
    channels: list[str],
) -> NashResult:
    """Находит равновесие Нэша в смешанных стратегиях для биматричной игры.

    A — платёжная матрица игрока 1 (строки — его стратегии)
    B — платёжная матрица игрока 2 (столбцы — его стратегии)
    channels — названия стратегий (каналов) для отображения

    Если равновесий несколько, берётся первое из найденных.
    """
    game = nash.Game(A, B)
    equilibria = list(game.support_enumeration())

    if not equilibria:
        # Страховочный случай: равновесий нет -> равномерное распределение
        n = len(channels)
        uniform = [1.0 / n] * n
        return NashResult(
            player1_strategy=uniform,
            player2_strategy=uniform,
            channels=channels,
            expected_payoff_player1=0.0,
            expected_payoff_player2=0.0,
        )

    # Берём первое равновесие
    p_strat, q_strat = equilibria[0]

    # Ожидаемый выигрыш = p^T * A * q
    exp1 = float(p_strat @ A @ q_strat)
    exp2 = float(p_strat @ B @ q_strat)

    return NashResult(
        player1_strategy=[float(x) for x in p_strat],
        player2_strategy=[float(x) for x in q_strat],
        channels=channels,
        expected_payoff_player1=exp1,
        expected_payoff_player2=exp2,
    )
