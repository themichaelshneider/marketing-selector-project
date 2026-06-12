"""Тесты модуля core/game.py — поиск равновесия Нэша."""

import pytest
import numpy as np
from core.game import solve_game


def test_matching_pennies():
    A = np.array([[1, -1], [-1, 1]])
    B = np.array([[-1, 1], [1, -1]])
    result = solve_game(A, B, ["Орёл", "Решка"])
    assert abs(result.player1_strategy[0] - 0.5) < 1e-6
    assert abs(result.player2_strategy[0] - 0.5) < 1e-6


def test_prisoners_dilemma():
    A = np.array([[-1, -10], [0, -8]])
    B = np.array([[-1, 0], [-10, -8]])
    result = solve_game(A, B, ["Молчать", "Предать"])
    assert abs(result.player1_strategy[1] - 1.0) < 1e-6
    assert abs(result.player2_strategy[1] - 1.0) < 1e-6


def test_probabilities_sum_to_one():
    A = np.array([[3, 1], [0, 2]])
    B = np.array([[2, 0], [1, 3]])
    result = solve_game(A, B, ["A", "B"])
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
    assert abs(sum(result.player2_strategy) - 1.0) < 1e-6


def test_no_equilibrium_fallback(monkeypatch):
    import nashpy as nash

    class MockGame:
        def __init__(self, A, B):
            pass
        def support_enumeration(self):
            return []

    monkeypatch.setattr(nash, "Game", MockGame)

    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    result = solve_game(A, B, ["A", "B"])
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
    assert result.expected_payoff_player1 == 0.0
