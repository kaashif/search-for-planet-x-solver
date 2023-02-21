# 12 sectors is standard, 18 sectors is expert
from typing import Iterable

NUM_SECTORS = 12

NUM_VISIBLE_SECTORS = NUM_SECTORS // 2


def next_sector(i: int) -> int:
    return (i + 1) % NUM_SECTORS


def prev_sector(i: int) -> int:
    return (i - 1) % NUM_SECTORS


def get_range_cyclic(start: int, length: int) -> Iterable[int]:
    current = start
    num_yielded = 0

    while num_yielded < length:
        yield current
        num_yielded += 1
        current = next_sector(current)
