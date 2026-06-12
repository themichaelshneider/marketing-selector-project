"""Тесты модуля core/ahp.py — метод анализа иерархий (AHP)."""

from core.ahp import rate_channels
from core.models import ChannelData, CriterionDef


def test_basic_ranking():
    channels = [
        ChannelData(name="A", reach=80, cac=100, relevance=9),
        ChannelData(name="B", reach=60, cac=500, relevance=5),
    ]
    criteria = [CriterionDef(name="Охват", key="reach", higher_is_better=True)]
    ratings = rate_channels(channels, criteria, weights=[1.0])
    assert ratings[0].name == "A"
    assert ratings[0].rank == 1


def test_reverse_weights():
    channels = [
        ChannelData(name="TV", reach=80, cac=1500, relevance=5),
        ChannelData(name="Telegram", reach=30, cac=300, relevance=9),
    ]
    criteria = [
        CriterionDef(name="Охват", key="reach", higher_is_better=True),
        CriterionDef(name="CAC", key="cac", higher_is_better=False),
    ]
    ratings = rate_channels(channels, criteria, weights=[1.0, 0.0])
    assert ratings[0].name == "TV"


def test_single_channel():
    channels = [ChannelData(name="A", reach=50, cac=500, relevance=5)]
    criteria = [CriterionDef(name="Охват", key="reach", higher_is_better=True)]
    ratings = rate_channels(channels, criteria, weights=[1.0])
    assert len(ratings) == 1
    assert ratings[0].rank == 1


def test_min_criterion_handling():
    channels = [
        ChannelData(name="A", reach=50, cac=300, relevance=5),
        ChannelData(name="B", reach=50, cac=500, relevance=5),
    ]
    criteria = [CriterionDef(name="CAC", key="cac", higher_is_better=False)]
    ratings = rate_channels(channels, criteria, weights=[1.0])
    assert ratings[0].name == "A"
