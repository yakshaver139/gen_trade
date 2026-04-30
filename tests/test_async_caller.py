from concurrent.futures import ThreadPoolExecutor

from gentrade.async_caller import future_caller


def _sum_four(a, b, c, d):
    return a + b + c + d


def test_future_caller_binds_constants_then_iterable():
    res = future_caller(_sum_four, [1, 2, 3, 4], ThreadPoolExecutor, 1, 2, 3)
    assert res == (7, 8, 9, 10)
