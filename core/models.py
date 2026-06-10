"""Pydantic-схемы для валидации входных и выходных данных."""

from pydantic import BaseModel, Field, field_validator
from typing import List


class ChannelData(BaseModel):
    """Исходные данные рекламного канала."""
    name: str
    reach: float = Field(gt=0, le=100, description="Охват целевой аудитории, %")
    cac: float = Field(ge=0, description="Стоимость привлечения клиента, руб")
    relevance: float = Field(ge=1, le=10, description="Релевантность аудитории (1-10)")


class CriterionDef(BaseModel):
    """Описание критерия для АИИ-оценки."""
    name: str
    key: str
    higher_is_better: bool


class ChannelRating(BaseModel):
    """Результат оценки канала методом АИИ."""
    name: str
    score: float
    rank: int = Field(ge=1)


class NashResult(BaseModel):
    """Результат поиска равновесия Нэша."""
    player1_strategy: List[float]
    player2_strategy: List[float]
    channels: List[str]
    expected_payoff_player1: float
    expected_payoff_player2: float


class GameConfig(BaseModel):
    """Полная конфигурация для расчёта."""
    channels: List[ChannelData]
    criteria: List[CriterionDef]
    budget: float = Field(gt=0, description="Бюджет на рекламную кампанию, руб")
    margin: float = Field(gt=0, le=1, description="Маржа (доля прибыли от выручки)")
    conversion: float = Field(gt=0, le=1, description="Конверсия (доля от охвата)")
    market_size: float = Field(gt=0, description="Размер целевого рынка, чел")
