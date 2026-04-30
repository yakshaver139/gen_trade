"""Wrapper library for async calls using Futures.

When implementing the function being called, place the iterable element as the
LAST positional argument. The constants get bound via ``functools.partial``
ahead of the iterable.

Example
-------
A function that takes three constants and one iterable element::

    def func(a, b, c, item):
        return a + b + c + item

    process_future_caller(func, [1, 2, 3], a, b, c)
"""

import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

CPU_COUNT = multiprocessing.cpu_count()


def process_future_caller(func: Callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ProcessPoolExecutor, *constants)


def threaded_future_caller(func: Callable, iterable: Iterable, *constants):
    return future_caller(func, iterable, ThreadPoolExecutor, *constants)


def future_caller(func: Callable, iterable: Iterable, executor_cls=ThreadPoolExecutor, *constants):
    """Map ``func`` over ``iterable`` with ``constants`` bound as the leading args."""
    with executor_cls(max_workers=CPU_COUNT) as executor:
        bound = partial(func, *constants)
        results = executor.map(bound, iterable)
    return tuple(results)
