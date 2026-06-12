"""Поиск равновесия Нэша в биматричной игре с помощью Nashpy."""

import logging

import nashpy as nash
import numpy as np
from core.models import NashResult

logger = logging.getLogger("marketings.game")


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
        n = len(channels)
        uniform = [1.0 / n] * n
        logger.warning("Равновесий не найдено — возвращено равномерное распределение")
        return NashResult(
            player1_strategy=uniform,
            player2_strategy=uniform,
            channels=channels,
            expected_payoff_player1=0.0,
            expected_payoff_player2=0.0,
        )

    def _is_pure(vec):
        return all(abs(p - round(p)) < 1e-6 for p in vec)

    # Предпочитаем смешанное равновесие чистому
    mixed = [(p, q) for p, q in equilibria if not _is_pure(p)]
    p_strat, q_strat = mixed[0] if mixed else equilibria[0]

    exp1 = float(p_strat @ A @ q_strat)
    exp2 = float(p_strat @ B @ q_strat)

    if _is_pure(p_strat) and _is_pure(q_strat):
        logger.warning(
            "Игра вырождена: найдено чисто-стратегическое равновесие %s / %s",
            [float(x) for x in p_strat],
            [float(x) for x in q_strat],
        )

    return NashResult(
        player1_strategy=[float(x) for x in p_strat],
        player2_strategy=[float(x) for x in q_strat],
        channels=channels,
        expected_payoff_player1=exp1,
        expected_payoff_player2=exp2,
    )
