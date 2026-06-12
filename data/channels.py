"""Загрузка конфигурации каналов и критериев из YAML-файла."""

import yaml
from pathlib import Path
from pydantic import ValidationError
from core.models import ChannelData, CriterionDef, GameConfig
from utils.logger import setup_logger

logger = setup_logger("marketings.data")


def load_config(path: str = "config/params.yaml") -> GameConfig:
    """Загружает и валидирует конфигурацию из YAML-файла.

    path: путь к файлу конфигурации (по умолчанию config/params.yaml)

    Возвращает GameConfig.

    Исключения:
        FileNotFoundError — файл конфигурации не найден
        yaml.YAMLError — синтаксическая ошибка в YAML
        KeyError — отсутствуют обязательные поля
        ValidationError — данные не прошли Pydantic-валидацию
    """
    config_path = Path(__file__).resolve().parent.parent / path

    if not config_path.exists():
        msg = f"Файл конфигурации не найден: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Ошибка синтаксиса YAML в {config_path}: {e}"
        logger.error(msg)
        raise

    if raw is None:
        msg = f"Файл конфигурации пуст: {config_path}"
        logger.error(msg)
        raise ValueError(msg)

    required_keys = {"channels", "criteria", "budget", "margin", "conversion", "market_size",
                     "competitor_budget", "competitor_margin", "competition_beta", "avg_revenue_per_client"}
    missing = required_keys - set(raw.keys())
    if missing:
        msg = f"В конфигурации отсутствуют обязательные поля: {missing}"
        logger.error(msg)
        raise KeyError(msg)

    try:
        channels = [ChannelData(**ch_data) for ch_data in raw["channels"].values()]
        criteria = [CriterionDef(**c) for c in raw["criteria"]]
        config = GameConfig(
            channels=channels,
            criteria=criteria,
            budget=raw["budget"],
            margin=raw["margin"],
            conversion=raw["conversion"],
            market_size=raw["market_size"],
            competitor_budget=raw["competitor_budget"],
            competitor_margin=raw["competitor_margin"],
            competition_beta=raw["competition_beta"],
            avg_revenue_per_client=raw["avg_revenue_per_client"],
        )
    except ValidationError as e:
        logger.error("Ошибка валидации данных конфигурации: %s", e)
        raise

    logger.info("Конфигурация загружена: %d каналов, %d критериев",
                len(config.channels), len(config.criteria))
    return config
