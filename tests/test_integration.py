"""Интеграционный тест — полный конвейер: данные → АИИ → игра."""

from data.channels import load_config
from data.payoffs import build_payoff_matrix
from core.ahp import rate_channels
from core.game import solve_game


def test_full_pipeline_returns_valid_result():
    config = load_config()
    weights = [0.4, 0.3, 0.3]

    ratings = rate_channels(config.channels, config.criteria, weights)
    assert len(ratings) == len(config.channels)
    assert ratings[0].rank == 1

    top_names = {r.name for r in ratings[:2]}
    top_channels = [ch for ch in config.channels if ch.name in top_names]
    A, B = build_payoff_matrix(
        top_channels, config.budget, config.margin, config.conversion,
        config.market_size, config.competitor_budget, config.competitor_margin,
        config.competition_beta, config.avg_revenue_per_client)
    assert A.shape == (2, 2)
    assert B.shape == (2, 2)

    result = solve_game(A, B, [ch.name for ch in top_channels])
    assert len(result.player1_strategy) == 2
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
