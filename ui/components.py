"""Переиспользуемые UI-компоненты для отображения результатов."""

import importlib.metadata  # noqa: необходим до streamlit для совместимости с Python 3.13

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from typing import List


def render_payoff_matrix(A: np.ndarray, B: np.ndarray, channels: List[str]) -> None:
    """Отображает платёжные матрицы в виде таблицы с парами (выигрыш1, выигрыш2).

    Строки — стратегии игрока 1, столбцы — стратегии игрока 2.
    """
    n = len(channels)
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(f"({A[i, j]:,.0f}, {B[i, j]:,.0f})")
        data.append(row)

    df = pd.DataFrame(
        data,
        index=[f"Игрок 1: {ch}" for ch in channels],
        columns=[f"Игрок 2: {ch}" for ch in channels],
    )
    st.table(df)


def render_strategy_pie(probabilities: List[float], channels: List[str], title: str) -> None:
    """Отображает вероятностное распределение стратегий в виде круговой диаграммы."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Фильтруем каналы с ненулевой вероятностью
    nonzero = [(p, ch) for p, ch in zip(probabilities, channels) if p > 0.001]
    if not nonzero:
        st.info("Все вероятности близки к нулю.")
        return

    probs, labels = zip(*nonzero)
    wedges, texts, autotexts = ax.pie(
        probs,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.axis("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    st.pyplot(fig)


def render_rankings_table(rankings) -> None:
    """Отображает таблицу рейтинга каналов (ранг, название, оценка)."""
    df = pd.DataFrame([
        {"Ранг": r.rank, "Канал": r.name, "Интегральная оценка": f"{r.score:.4f}"}
        for r in rankings
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)
