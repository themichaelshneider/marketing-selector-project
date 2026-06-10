"""Тесты модуля data/payoffs.py — построение платёжных матриц."""

import pytest
import numpy as np
from data.payoffs import build_payoff_matrix
from core.models import ChannelData


def test_same_channel_split():
    """Если оба выбрали один канал — аудитория делится поровну."""
    channels = [
        ChannelData(name="A", reach=100, cac=0, relevance=10),
    ]
    A, B = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000
    )
    # reach=100% -> 1000 чел, пополам = 500 чел -> 500 прибыли
    expected = 500.0
    assert A[0, 0] == pytest.approx(expected)
    assert B[0, 0] == pytest.approx(expected)


def test_different_channels():
    """Если игроки выбрали разные каналы — пересечения нет."""
    channels = [
        ChannelData(name="A", reach=80, cac=0, relevance=5),
        ChannelData(name="B", reach=40, cac=0, relevance=5),
    ]
    A, B = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000
    )
    # A выбирает A (0), B выбирает B (1)
    profit_a = (80 / 100.0) * 1000  # 800
    profit_b = (40 / 100.0) * 1000  # 400
    assert A[0, 1] == pytest.approx(profit_a)
    assert B[0, 1] == pytest.approx(profit_b)


def test_budget_deducted():
    """Бюджет должен вычитаться из прибыли."""
    channels = [
        ChannelData(name="A", reach=50, cac=0, relevance=5),
    ]
    A, B = build_payoff_matrix(
        channels, budget=200, margin=1.0, conversion=1.0, market_size=1000
    )
    # Прибыль до вычета: 500, после вычета бюджета: 300
    # Но т.к. канал один, аудитория делится: 250 - 200 = 50
    expected = (50 / 100.0) * 1.0 * 1.0 * 1000 / 2.0 - 200  # 250 - 200 = 50
    assert A[0, 0] == pytest.approx(expected)


def test_zero_market():
    """При нулевом рынке прибыль = -бюджет."""
    channels = [
        ChannelData(name="A", reach=80, cac=0, relevance=5),
    ]
    A, _ = build_payoff_matrix(
        channels, budget=500, margin=1.0, conversion=1.0, market_size=0
    )
    assert A[0, 0] == pytest.approx(-500.0)


def test_matrix_shape():
    """Размер матрицы должен соответствовать числу каналов."""
    channels = [
        ChannelData(name="A", reach=80, cac=0, relevance=5),
        ChannelData(name="B", reach=60, cac=0, relevance=5),
        ChannelData(name="C", reach=40, cac=0, relevance=5),
    ]
    A, B = build_payoff_matrix(
        channels, budget=0, margin=1.0, conversion=1.0, market_size=1000
    )
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)
