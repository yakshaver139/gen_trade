from unittest.mock import patch

import pytest

from gentrade import load_strategy

RSI = {
    "indicator": "RSI",
    "type": "momentum",
    "absolute": True,
    "op": ">=",
    "abs_value": 70,
    "rel_value": None,
}
EMA_1 = {
    "indicator": "EMA",
    "type": "trend",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "MA",
}
EMA_2 = {
    "indicator": "EMA",
    "type": "trend",
    "absolute": False,
    "op": ">=",
    "abs_value": None,
    "rel_value": "PREVIOUS_PERIOD",
}


@pytest.mark.parametrize(
    "strategy,callback_count,expected",
    [
        (dict(indicators=[RSI], conjunctions=[]), 0, "RSI >= 70"),
        (dict(indicators=[EMA_1], conjunctions=[]), 1, "EMA >= Z"),
        (
            dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "OR"]),
            2,
            "RSI >= 70 AND EMA >= Z OR EMA >= Z",
        ),
    ],
)
@patch("gentrade.load_strategy.get_callback")
def test_load_strategy(get_callback, strategy, callback_count, expected):
    get_callback.return_value = "EMA >= Z"
    assert load_strategy.load_from_object(strategy) == expected
    assert get_callback.call_count == callback_count


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (dict(indicators=[RSI], conjunctions=[]), "RSI >= 70"),
        (
            dict(indicators=[RSI, EMA_1, EMA_2], conjunctions=["AND", "OR"]),
            "RSI >= 70 AND (EMA >= Z OR EMA >= Z)",
        ),
        (
            dict(indicators=[RSI, EMA_1, EMA_2, RSI], conjunctions=["AND", "OR", "AND"]),
            "RSI >= 70 AND (EMA >= Z OR EMA >= Z AND RSI >= 70)",
        ),
        (
            dict(indicators=[RSI, EMA_1, EMA_2, RSI], conjunctions=["AND", "OR", "OR"]),
            "RSI >= 70 AND (EMA >= Z OR (EMA >= Z OR RSI >= 70))",
        ),
    ],
)
@patch("gentrade.load_strategy.get_callback")
def test_load_strategy_parenthesised(get_callback, strategy, expected):
    get_callback.return_value = "EMA >= Z"
    assert load_strategy.load_from_object_parenthesised(strategy) == expected
