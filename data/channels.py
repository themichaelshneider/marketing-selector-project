"""Загрузка конфигурации каналов и критериев из YAML-файла."""

import yaml
from pathlib import Path
from core.models import ChannelData, CriterionDef, GameConfig


def load_config(path: str = "config/params.yaml") -> GameConfig:
    """Загружает и валидирует конфигурацию из YAML-файла.

    path: путь к файлу конфигурации (по умолчанию config/params.yaml)
    """
    # Путь относительно корня проекта
    config_path = Path(__file__).resolve().parent.parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Парсим каналы
    channels = []
    for ch_data in raw["channels"].values():
        channels.append(ChannelData(**ch_data))

    # Парсим критерии
    criteria = [CriterionDef(**c) for c in raw["criteria"]]

    return GameConfig(
        channels=channels,
        criteria=criteria,
        budget=raw["budget"],
        margin=raw["margin"],
        conversion=raw["conversion"],
        market_size=raw["market_size"],
    )
