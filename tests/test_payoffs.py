"""Тесты модуля data/payoffs.py — построение платёжных матриц."""

import pytest
import numpy as np
from data.payoffs import build_payoff_matrix
from core.models import ChannelData

KW = dict(competitor_budget=800000, competitor_margin=0.30,
           competition_beta=0.5, avg_revenue_per_client=15000)


def test_same_channel_split():
    channels = [ChannelData(name="A", reach=100, cac=500, relevance=10)]
    A, B = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000, **KW)
    assert A[0, 0] == pytest.approx(500 * 14500 * 1.0)
    assert B[0, 0] == pytest.approx(500 * 14500 * 0.30 - 800000)


def test_different_channels():
    channels = [
        ChannelData(name="A", reach=80, cac=1000, relevance=5),
        ChannelData(name="B", reach=40, cac=2000, relevance=5),
    ]
    A, B = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000, **KW)
    # eff_i = 80 * (1 - 0.5 * 40/100) = 64, clients = 640, net = 14000
    assert A[0, 1] == pytest.approx(640 * 14000 * 1.0)
    assert B[0, 1] == pytest.approx(240 * 13000 * 0.30 - 800000)


def test_asymmetric_players():
    channels = [
        ChannelData(name="A", reach=80, cac=1000, relevance=5),
        ChannelData(name="B", reach=40, cac=2000, relevance=5),
    ]
    A, B = build_payoff_matrix(
        channels, budget=500_000, margin=0.25, conversion=0.03, market_size=1_000_000,
        competitor_budget=400_000, competitor_margin=0.20,
        competition_beta=0.5, avg_revenue_per_client=15000)
    assert not np.allclose(B, A.T), "B == A.T — игра вырождена"


def test_cac_affects_payoff():
    channels = [
        ChannelData(name="Дешёвый", reach=50, cac=500, relevance=5),
        ChannelData(name="Дорогой", reach=50, cac=3000, relevance=5),
    ]
    A, _ = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000, **KW)
    assert A[0, 1] > A[1, 0], "Дешёвый канал должен давать больший выигрыш"
