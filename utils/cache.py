"""Декораторы для кэширования результатов тяжёлых вычислений."""

from functools import lru_cache
import logging

logger = logging.getLogger("marketings.cache")


def cached_computation(maxsize: int = 32):
    """Декоратор для кэширования результатов ресурсоёмких расчётов.

    Использует lru_cache под капотом. Для работы с numpy-массивами
    необходимо передавать hashable-аргументы (например, tuple).
    """
    def decorator(func):
        @lru_cache(maxsize=maxsize)
        def wrapper(*args, **kwargs):
            logger.debug(f"Кэш промах для {func.__name__}, вычисляю...")
            return func(*args, **kwargs)

        def clear_cache():
            wrapper.cache_clear()
            logger.info(f"Кэш {func.__name__} очищен")

        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator
