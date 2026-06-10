"""Streamlit-приложение MarketingSelector.

Позволяет пользователю задавать веса критериев, параметры расчёта
и получать рекомендацию по выбору рекламного канала.
"""

import importlib.metadata  # noqa: необходим до streamlit для совместимости с Python 3.13

import streamlit as st
import sys
from pathlib import Path

# Добавляем корень проекта в путь импорта
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from data.channels import load_config
from data.payoffs import build_payoff_matrix
from core.ahp import rate_channels
from core.game import solve_game
from ui.components import (
    render_payoff_matrix,
    render_strategy_pie,
    render_rankings_table,
)
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()


def main() -> None:
    st.set_page_config(
        page_title="MarketingSelector",
        page_icon="📊",
        layout="wide",
    )

    st.title("MarketingSelector: выбор рекламной кампании")
    st.markdown(
        "**Этап 1:** оценка каналов методом анализа иерархий (АИИ)  ·  "
        "**Этап 2:** моделирование реакции конкурента как матричной игры"
    )

    # ------------------------------------------------------------------
    # Загрузка конфигурации
    # ------------------------------------------------------------------
    config = load_config()

    # ------------------------------------------------------------------
    # Боковая панель: веса критериев
    # ------------------------------------------------------------------
    st.sidebar.header("Веса критериев")
    weights_raw = {}
    for crit in config.criteria:
        w = st.sidebar.slider(
            f"**{crit.name}**",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help=f"Чем выше вес, тем важнее критерий «{crit.name}» при выборе канала",
        )
        weights_raw[crit.key] = w

    # Нормализуем веса в долях от суммы
    total = sum(weights_raw.values())
    if total > 0:
        weights_norm = [weights_raw[c.key] / total for c in config.criteria]
    else:
        # Если все веса нулевые — равномерное распределение
        weights_norm = [1.0 / len(config.criteria)] * len(config.criteria)

    st.sidebar.caption(
        f"Нормализованные веса: "
        + ", ".join(f"{c.name}={w:.2f}" for c, w in zip(config.criteria, weights_norm))
    )

    # ------------------------------------------------------------------
    # Боковая панель: параметры расчёта
    # ------------------------------------------------------------------
    st.sidebar.header("Параметры расчёта")
    budget = st.sidebar.number_input(
        "Бюджет (руб)",
        min_value=100_000,
        max_value=10_000_000,
        value=int(config.budget),
        step=100_000,
        format="%d",
    )
    margin_pct = st.sidebar.slider("Маржа (%)", 5, 50, int(config.margin * 100))
    conversion_pct = st.sidebar.slider("Конверсия (%)", 1, 20, int(config.conversion * 100))

    margin = margin_pct / 100.0
    conversion = conversion_pct / 100.0

    # ------------------------------------------------------------------
    # Кнопка запуска расчёта
    # ------------------------------------------------------------------
    if st.sidebar.button("Рассчитать", type="primary"):
        logger.info("Запуск расчёта. Веса: %s", weights_norm)

        # ---- Шаг 1: АИИ-ранжирование каналов -------------------------
        ratings = rate_channels(config.channels, config.criteria, weights_norm)

        st.header("Шаг 1: оценка каналов (АИИ)")
        render_rankings_table(ratings)

        # Берём топ-3 канала для построения игры
        top_names = {r.name for r in ratings[:3]}
        top_channels = [ch for ch in config.channels if ch.name in top_names]

        # ---- Шаг 2: платёжная матрица ---------------------------------
        A, B = build_payoff_matrix(
            top_channels,
            budget,
            margin,
            conversion,
            config.market_size,
        )
        channel_names = [ch.name for ch in top_channels]

        st.header("Шаг 2: платёжная матрица")
        st.caption(
            "Формат ячейки: **(прибыль игрока 1, прибыль игрока 2)**, руб"
        )
        render_payoff_matrix(A, B, channel_names)

        # ---- Шаг 3: равновесие Нэша -----------------------------------
        result = solve_game(A, B, channel_names)

        st.header("Шаг 3: равновесие Нэша")

        col1, col2 = st.columns(2)
        with col1:
            render_strategy_pie(
                result.player1_strategy,
                result.channels,
                "Стратегия игрока 1 (мы)",
            )
        with col2:
            render_strategy_pie(
                result.player2_strategy,
                result.channels,
                "Стратегия игрока 2 (конкурент)",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Ожидаемая прибыль игрока 1",
                f"{result.expected_payoff_player1:,.0f} руб",
            )
        with col2:
            st.metric(
                "Ожидаемая прибыль игрока 2",
                f"{result.expected_payoff_player2:,.0f} руб",
            )

        # ---- Рекомендация ---------------------------------------------
        st.header("Рекомендация")
        best_idx = max(
            range(len(result.player1_strategy)),
            key=lambda i: result.player1_strategy[i],
        )
        best_channel = result.channels[best_idx]
        best_prob = result.player1_strategy[best_idx] * 100

        st.success(
            f"Рекомендуется направить **{best_prob:.1f}%** бюджета "
            f"на канал **«{best_channel}»**"
        )

        # Распечатываем детали распределения
        st.table(
            pd.DataFrame([
                {
                    "Канал": ch,
                    "Доля бюджета": f"{p * 100:.1f}%",
                    "Сумма, руб": f"{p * budget:,.0f}",
                }
                for ch, p in zip(result.channels, result.player1_strategy)
                if p > 0.001
            ])
        )


if __name__ == "__main__":
    main()
