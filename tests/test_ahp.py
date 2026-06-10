"""Тесты модуля math/ahp.py — ранжирование каналов."""

import pytest
from core.ahp import rate_channels
from core.models import ChannelData, CriterionDef


def test_basic_ranking():
    """Канал с лучшими показателями по всем критериям должен быть на 1-м месте."""
    channels = [
        ChannelData(name="Лучший", reach=95, cac=100, relevance=10),
        ChannelData(name="Худший", reach=20, cac=900, relevance=2),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
        CriterionDef(name="CAC", key="cac", higher_is_better=False),
        CriterionDef(name="Релевантность", key="relevance", higher_is_better=True),
    ]
    weights = [0.4, 0.3, 0.3]
    ratings = rate_channels(channels, criteria, weights)
    assert ratings[0].name == "Лучший"
    assert ratings[0].rank == 1


def test_reverse_weights():
    """Если вес CAC=0, а Охват=10, то канал с бОльшим охватом выигрывает,
    даже если у него высокий CAC."""
    channels = [
        ChannelData(name="Широкий", reach=90, cac=2000, relevance=5),
        ChannelData(name="Узкий", reach=30, cac=200, relevance=8),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
        CriterionDef(name="CAC", key="cac", higher_is_better=False),
        CriterionDef(name="Релевантность", key="relevance", higher_is_better=True),
    ]
    # Весь вес на охват, CAC и релевантность не важны
    weights = [1.0, 0.0, 0.0]
    ratings = rate_channels(channels, criteria, weights)
    assert ratings[0].name == "Широкий"


def test_all_weights_equal():
    """При равных весах порядок должен определяться нормализованными значениями."""
    channels = [
        ChannelData(name="A", reach=80, cac=400, relevance=6),
        ChannelData(name="B", reach=60, cac=300, relevance=8),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
        CriterionDef(name="CAC", key="cac", higher_is_better=False),
        CriterionDef(name="Релевантность", key="relevance", higher_is_better=True),
    ]
    weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    ratings = rate_channels(channels, criteria, weights)
    # Просто проверяем, что оба получили оценку и ранг
    assert len(ratings) == 2
    assert all(isinstance(r.score, float) for r in ratings)
    assert ratings[0].rank == 1
    assert ratings[1].rank == 2


def test_all_weights_zero():
    """Если все веса = 0, оценки должны быть одинаковыми (равномерное распределение)."""
    channels = [
        ChannelData(name="A", reach=90, cac=100, relevance=9),
        ChannelData(name="B", reach=50, cac=500, relevance=5),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
        CriterionDef(name="CAC", key="cac", higher_is_better=False),
        CriterionDef(name="Релевантность", key="relevance", higher_is_better=True),
    ]
    weights = [0.0, 0.0, 0.0]
    ratings = rate_channels(channels, criteria, weights)
    # Оценки должны быть равны
    assert abs(ratings[0].score - ratings[1].score) < 1e-6


def test_single_channel():
    """Если всего один канал, он всегда на 1-м месте."""
    channels = [
        ChannelData(name="Единственный", reach=50, cac=500, relevance=5),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
    ]
    weights = [1.0]
    ratings = rate_channels(channels, criteria, weights)
    assert len(ratings) == 1
    assert ratings[0].rank == 1
    assert ratings[0].name == "Единственный"
