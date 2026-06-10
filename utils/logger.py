"""Настройка логирования проекта."""

import logging
import sys


def setup_logger(name: str = "marketings") -> logging.Logger:
    """Создаёт и настраивает логгер с выводом в stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Добавляем обработчик только если ещё не добавлен
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
