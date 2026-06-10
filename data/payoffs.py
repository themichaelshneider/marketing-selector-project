"""Построение платёжных матриц для биматричной игры на основе данных каналов."""

import numpy as np
from core.models import ChannelData


def build_payoff_matrix(
    channels: list[ChannelData],
    budget: float,
    margin: float,
    conversion: float,
    market_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит платёжные матрицы A и B для двух игроков.

    Игрок 1 выбирает строку (канал i), игрок 2 выбирает столбец (канал j).
    Если оба выбрали один канал — аудитория делится поровну между ними.
    Выигрыш = прибыль от канала минус бюджет на кампанию.

    Возвращает (A, B), где:
        A[i,j] — выигрыш игрока 1 при его стратегии i и стратегии j оппонента
        B[i,j] — выигрыш игрока 2 при его стратегии j и стратегии i оппонента
    """
    n = len(channels)
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Выигрыш игрока 1 (выбрал канал i)
            if i == j:
                # Оба в одном канале — делим аудиторию пополам
                reach_1 = channels[i].reach / 2.0
            else:
                # В разных каналах — каждый получает всю свою аудиторию
                reach_1 = channels[i].reach

            # Прибыль = (охват/100) * конверсия * маржа * рынок - бюджет
            A[i, j] = (
                (reach_1 / 100.0)
                * conversion
                * margin
                * market_size
                - budget
            )

            # Выигрыш игрока 2 (выбрал канал j)
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
