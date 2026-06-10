"""Тесты модуля math/game.py — поиск равновесия Нэша."""

import pytest
import numpy as np
from core.game import solve_game


def test_matching_pennies():
    """Орлянка: симметричная игра, равновесие в смешанных стратегиях (0.5, 0.5)."""
    A = np.array([[1, -1], [-1, 1]])
    B = np.array([[-1, 1], [1, -1]])
    result = solve_game(A, B, ["Орёл", "Решка"])
    assert abs(result.player1_strategy[0] - 0.5) < 1e-6
    assert abs(result.player1_strategy[1] - 0.5) < 1e-6
    assert abs(result.player2_strategy[0] - 0.5) < 1e-6
    assert abs(result.player2_strategy[1] - 0.5) < 1e-6


def test_pure_equilibrium():
    """Игра с доминирующей стратегией: равновесие в чистых стратегиях."""
    A = np.array([[5, 5], [0, 0]])
    B = np.array([[5, 0], [5, 0]])
    result = solve_game(A, B, ["Верх", "Низ"])
    # Для обоих игроков стратегия "Верх" доминирует
    assert abs(result.player1_strategy[0] - 1.0) < 1e-6
    assert abs(result.player1_strategy[1]) < 1e-6
    assert abs(result.player2_strategy[0] - 1.0) < 1e-6
    assert abs(result.player2_strategy[1]) < 1e-6


def test_prisoners_dilemma():
    """Дилемма заключённого: (предать, предать) — единственное равновесие."""
    # (молчать, молчать) = (-1, -1), (предать, предать) = (-8, -8)
    # (молчать, предать) = (-10, 0), (предать, молчать) = (0, -10)
    A = np.array([[-1, -10], [0, -8]])
    B = np.array([[-1, 0], [-10, -8]])
    result = solve_game(A, B, ["Молчать", "Предать"])
    # Равновесие: оба предают (стратегия 1, т.е. индекс 1)
    assert abs(result.player1_strategy[1] - 1.0) < 1e-6
    assert abs(result.player2_strategy[1] - 1.0) < 1e-6


def test_probabilities_sum_to_one():
    """Сумма вероятностей каждого игрока должна быть 1."""
    A = np.array([[3, 1], [0, 2]])
    B = np.array([[2, 0], [1, 3]])
    result = solve_game(A, B, ["A", "B"])
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
    assert abs(sum(result.player2_strategy) - 1.0) < 1e-6


def test_three_strategies():
    """Игра 3x3 должна находить равновесие."""
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    B = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    result = solve_game(A, B, ["A", "B", "C"])
    assert len(result.player1_strategy) == 3
    assert abs(sum(result.player1_strategy) - 1.0) < 1e-6
