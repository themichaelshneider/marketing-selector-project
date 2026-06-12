"""Тесты модуля utils/cache.py — декоратор кэширования."""

from utils.cache import cached_computation


def test_cache_hit():
    call_count = 0

    @cached_computation(maxsize=4)
    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    assert compute(3) == 9
    assert call_count == 1
    assert compute(3) == 9
    assert call_count == 1


def test_clear_cache():
    call_count = 0

    @cached_computation(maxsize=4)
    def compute(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    assert compute(5) == 10
    assert call_count == 1
    compute.clear_cache()
    assert compute(5) == 10
    assert call_count == 2
