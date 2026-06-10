"""Pydantic-схемы для валидации входных и выходных данных."""

from pydantic import BaseModel
from typing import List


class ChannelData(BaseModel):
    """Исходные данные рекламного канала."""
    name: str          # название канала
    reach: float       # охват целевой аудитории, %
    cac: float         # стоимость привлечения клиента, руб
    relevance: float   # релевантность аудитории (1-10)


class CriterionDef(BaseModel):
    """Описание критерия для АИИ-оценки."""
    name: str              # название критерия
    key: str               # ключ для доступа к ChannelData
    higher_is_better: bool # направление оптимизации


class ChannelRating(BaseModel):
    """Результат оценки канала методом АИИ."""
    name: str    # название канала
    score: float # интегральная оценка
    rank: int    # место в рейтинге (1 = лучшее)


class NashResult(BaseModel):
    """Результат поиска равновесия Нэша."""
    player1_strategy: List[float]  # вероятности для игрока 1
    player2_strategy: List[float]  # вероятности для игрока 2
    channels: List[str]            # названия каналов-стратегий
    expected_payoff_player1: float # ожидаемый выигрыш игрока 1
    expected_payoff_player2: float # ожидаемый выигрыш игрока 2


class GameConfig(BaseModel):
    """Полная конфигурация для расчёта."""
    channels: List[ChannelData]
    criteria: List[CriterionDef]
    budget: float
    margin: float
    conversion: float
    market_size: float
