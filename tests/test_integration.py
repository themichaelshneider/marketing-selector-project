"""Интеграционные тесты — проверка полного конвейера: данные → АИИ → игра."""

import pytest
from data.channels import load_config
from data.payoffs import build_payoff_matrix
from core.ahp import rate_channels
from core.game import solve_game


def test_full_pipeline_returns_valid_result():
    """Полный цикл должен давать осмысленный результат."""
    config = load_config()

    # Стандартные веса
    weights = [0.4, 0.3, 0.3]

    # Шаг 1: АИИ-ранжирование
    ratings = rate_channels(config.channels, config.criteria, weights)
    assert len(ratings) == len(config.channels)
    assert ratings[0].rank == 1

    # Шаг 2: берём топ-3 канала и строим платёжную матрицу
    top_names = {r.name for r in ratings[:3]}
    top_channels = [ch for ch in config.channels if ch.name in top_names]
    A, B = build_payoff_matrix(
        top_channels,
        config.budget,
        config.margin,
        config.conversion,
        config.market_size,
    )
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)

    # Шаг 3: равновесие Нэша
    channel_names = [ch.name for ch in top_channels]
    result = solve_game(A, B, channel_names)

    # Проверки
    assert len(result.player1_strategy) == 3
    assert len(result.player2_strategy) == 3
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
    assert abs(sum(result.player2_strategy) - 1.0) < 1e-6


def test_full_pipeline_different_weights():
    """Конвейер работает с разными наборами весов (все веса на CAC)."""
    config = load_config()
    weights = [0.0, 1.0, 0.0]  # только CAC

    ratings = rate_channels(config.channels, config.criteria, weights)
    # Канал с минимальным CAC должен быть на первом месте
    assert ratings[0].name == "Telegram"  # CAC = 300

    top_names = {r.name for r in ratings[:3]}
    top_channels = [ch for ch in config.channels if ch.name in top_names]
    A, B = build_payoff_matrix(
        top_channels,
        config.budget,
        config.margin,
        config.conversion,
        config.market_size,
    )

    result = solve_game(A, B, [ch.name for ch in top_channels])
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
