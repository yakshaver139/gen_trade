from unittest.mock import patch

from gentrade import generate_strategy


def _ind(name: str, ind_type: str) -> dict:
    return {
        "indicator": name,
        "type": ind_type,
        "absolute": True,
        "op": ">=",
        "abs_value": 0.5,
        "rel_value": None,
    }


def test_choose_indicator_respects_max_same_class():
    momentum_a = _ind("a", "momentum")
    momentum_b = _ind("b", "momentum")
    trend_a = _ind("c", "trend")

    # already at the cap for momentum: must return the trend indicator.
    indicators = [momentum_a, momentum_b, trend_a]
    same_class = {"momentum": 2}
    chosen = generate_strategy.choose_indicator(indicators, max_same_class=1, same_class_indicators=same_class)
    assert chosen is trend_a


@patch("gentrade.generate_strategy.choose_num_indicators")
@patch("gentrade.generate_strategy.choose_indicator")
def test_generate_returns_strategy_dict(choose_indicator, choose_num_indicators):
    choose_num_indicators.return_value = 3
    base = _ind("base", "momentum")
    second = _ind("second", "trend")
    third = _ind("third", "volatility")
    choose_indicator.side_effect = [second, third]

    strategy = generate_strategy.generate(
        base, [base, second, third], conjunctions=["and"], max_indicators=3
    )

    assert strategy["indicators"] == [base, second, third]
    assert len(strategy["conjunctions"]) == 2
    assert strategy["conjunctions"][0] == "and"
    assert "id" in strategy
