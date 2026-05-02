"""Static description / formula / example for every indicator in the
trusted catalogue (``signals/{absolute,relative}_signals.json``).

Each entry has:
  description: one-paragraph plain-English summary
  formula:     terse mathematical statement (rendered as Python-style code)
  example:     concrete numeric example showing how to read the value

Used by the New Run page's seed-strategy builder, where each row's
``?`` button opens a modal showing the entry.

The descriptions are intentionally short and trader-oriented — they
explain what the number *means* and how to act on it, not the full
``ta`` library docs. Anything not in this dict falls back to a generic
"see the ``ta`` library docs" message.
"""
from __future__ import annotations

from typing import TypedDict


class Doc(TypedDict, total=False):
    description: str
    formula: str
    example: str


INDICATOR_DOCS: dict[str, Doc] = {
    # ----- momentum -----
    "momentum_ao": {
        "description": (
            "Awesome Oscillator. Measures market momentum by comparing a fast "
            "and slow simple moving average of bar midpoints. Crossovers above "
            "or below zero are read as momentum shifts."
        ),
        "formula": "AO = SMA(midpoint, 5) - SMA(midpoint, 34)\n  where midpoint = (high + low) / 2",
        "example": "AO > 0 → bullish momentum; AO crossing below 0 is a common bearish signal.",
    },
    "momentum_kama": {
        "description": (
            "Kaufman Adaptive Moving Average. A trend-following line that speeds up in "
            "trending markets and slows down in noisy ones. Compared as a level rather "
            "than via a fixed threshold."
        ),
        "formula": "KAMA_t = KAMA_{t-1} + SC × (price_t - KAMA_{t-1})\n  SC depends on the efficiency ratio (signal / noise).",
        "example": "Price > KAMA → up-trend bias; price < KAMA → down-trend bias.",
    },
    "momentum_ppo": {
        "description": (
            "Percentage Price Oscillator. Like MACD but expressed as a percentage of "
            "the slow EMA, so it's directly comparable across assets at different price "
            "scales."
        ),
        "formula": "PPO = ((EMA_fast(close) - EMA_slow(close)) / EMA_slow(close)) × 100",
        "example": "PPO > 0 → fast EMA above slow EMA (up-trend); a positive cross of zero is a buy signal.",
    },
    "momentum_pvo": {
        "description": (
            "Percentage Volume Oscillator. The PPO formula applied to volume, not price. "
            "Used to confirm price moves with corresponding volume momentum."
        ),
        "formula": "PVO = ((EMA_fast(volume) - EMA_slow(volume)) / EMA_slow(volume)) × 100",
        "example": "PVO > 0 alongside a price breakout is a confirmation that the move has volume behind it.",
    },
    "momentum_roc": {
        "description": (
            "Rate of Change. The percentage change between the current close and the "
            "close N periods ago. Pure momentum reading — sign indicates direction, "
            "magnitude indicates strength."
        ),
        "formula": "ROC = (close_t - close_{t-N}) / close_{t-N} × 100\n  N defaults to 12 in `ta`.",
        "example": "ROC = +5 → price is 5% above where it was 12 bars ago.",
    },
    "momentum_rsi": {
        "description": (
            "Relative Strength Index. Bounded oscillator in [0, 100] measuring the "
            "ratio of recent gains to losses. Classic mean-reversion signal: extreme "
            "highs are 'overbought', extreme lows are 'oversold'."
        ),
        "formula": "RSI = 100 - 100 / (1 + RS)\n  RS = avg(gains_N) / avg(losses_N)\n  N defaults to 14.",
        "example": "RSI ≥ 70 commonly means overbought (consider exit / short); RSI ≤ 30 oversold (consider entry).",
    },
    "momentum_stoch": {
        "description": (
            "Stochastic Oscillator (%K). Bounded in [0, 100] showing where the close "
            "sits within the recent N-bar range. Fast oscillator that flags reversals."
        ),
        "formula": "%K = (close - lowest_low_N) / (highest_high_N - lowest_low_N) × 100\n  N defaults to 14.",
        "example": "%K ≥ 80 → close near the recent high (overbought); ≤ 20 → near the low (oversold).",
    },
    "momentum_stoch_rsi": {
        "description": (
            "Stochastic of RSI. Applies the stochastic transform to the RSI series, "
            "amplifying reversals in already-bounded RSI moves."
        ),
        "formula": "stoch_rsi = (RSI - min(RSI, N)) / (max(RSI, N) - min(RSI, N))",
        "example": "Use the same overbought/oversold reading as RSI but with sharper turns.",
    },
    "momentum_stoch_rsi_d": {
        "description": (
            "%D component of the Stochastic RSI — a 3-period SMA of the stoch_rsi line. "
            "The slow signal-line companion to %K used for crossover signals."
        ),
        "formula": "%D = SMA(stoch_rsi, 3)",
        "example": "stoch_rsi_k crossing above stoch_rsi_d below 0.20 is a classic long signal.",
    },
    "momentum_stoch_rsi_k": {
        "description": (
            "%K component of the Stochastic RSI. Faster of the two stoch_rsi lines."
        ),
        "formula": "%K = SMA(stoch_rsi, 3) (in the `ta` library; raw stoch_rsi smoothed)",
        "example": "%K above 0.80 is overbought territory; below 0.20 is oversold.",
    },
    "momentum_tsi": {
        "description": (
            "True Strength Index. Doubly-smoothed momentum oscillator that filters out "
            "short-term noise. Centred around zero."
        ),
        "formula": "TSI = (EMA(EMA(price_change, 25), 13) / EMA(EMA(|price_change|, 25), 13)) × 100",
        "example": "TSI > 0 → up-trend; TSI crossing zero is a stronger trend-change signal than RSI.",
    },
    "momentum_uo": {
        "description": (
            "Ultimate Oscillator. Combines short / medium / long-term buying pressure into "
            "a single bounded oscillator, designed to reduce false signals."
        ),
        "formula": "UO = 100 × (4×A_7 + 2×A_14 + A_28) / 7\n  A_N = sum(BP) / sum(TR) over N periods.",
        "example": "UO ≥ 70 overbought; ≤ 30 oversold; bullish divergence below 30 is the canonical long.",
    },
    "momentum_wr": {
        "description": (
            "Williams %R. Inverted stochastic, bounded in [-100, 0]. Reads overbought "
            "near 0 and oversold near -100."
        ),
        "formula": "%R = (highest_high_N - close) / (highest_high_N - lowest_low_N) × -100",
        "example": "%R between -20 and 0 is overbought; between -100 and -80 is oversold.",
    },

    # ----- trend -----
    "trend_adx": {
        "description": (
            "Average Directional Index. Measures trend *strength*, not direction. "
            "Bounded in [0, 100]. Ignore for direction; use to gate other signals."
        ),
        "formula": "ADX = SMA(|+DI - -DI| / (+DI + -DI) × 100, 14)",
        "example": "ADX > 25 → market is trending strongly enough to trade; ADX < 20 → range-bound.",
    },
    "trend_adx_neg": {
        "description": (
            "Negative Directional Indicator (-DI). Component of ADX that measures "
            "downward directional movement only."
        ),
        "formula": "-DI = SMA(-DM / TR, 14) × 100\n  -DM = down-move when down-move > up-move else 0.",
        "example": "-DI rising above +DI flags a developing down-trend.",
    },
    "trend_adx_pos": {
        "description": (
            "Positive Directional Indicator (+DI). The up-move counterpart of -DI."
        ),
        "formula": "+DI = SMA(+DM / TR, 14) × 100",
        "example": "+DI > -DI with ADX > 25 is the canonical 'trending up, take longs' state.",
    },
    "trend_aroon_down": {
        "description": (
            "Aroon Down. Measures how recently the lowest low occurred within the lookback. "
            "Bounded in [0, 100]; high values mean a recent new low."
        ),
        "formula": "Aroon Down = ((N - bars_since_lowest_low) / N) × 100",
        "example": "Aroon Down > 70 with Aroon Up < 30 → established down-trend.",
    },
    "trend_aroon_ind": {
        "description": (
            "Aroon Indicator (Aroon Up - Aroon Down). Combined trend strength + direction "
            "in [-100, 100]."
        ),
        "formula": "Aroon Indicator = Aroon Up - Aroon Down",
        "example": "Aroon Indicator > 50 → strong up-trend; < -50 → strong down-trend.",
    },
    "trend_aroon_up": {
        "description": (
            "Aroon Up. How recently the highest high occurred within the lookback."
        ),
        "formula": "Aroon Up = ((N - bars_since_highest_high) / N) × 100",
        "example": "Aroon Up > 70 with Aroon Down < 30 → established up-trend.",
    },
    "trend_cci": {
        "description": (
            "Commodity Channel Index. Measures how far the typical price has deviated from "
            "its moving average, normalised by mean deviation. Unbounded; ±100 are common "
            "reference levels."
        ),
        "formula": "CCI = (typical_price - SMA(typical_price, N)) / (0.015 × mean_dev)",
        "example": "CCI > +100 → strong up-momentum; < -100 → strong down-momentum.",
    },
    "trend_ema_fast": {
        "description": (
            "Exponential Moving Average (fast period). Reacts quicker to recent price than SMA."
        ),
        "formula": "EMA_t = α × price_t + (1 - α) × EMA_{t-1}\n  α = 2 / (N + 1)\n  N for 'fast' is typically 12.",
        "example": "Fast EMA crossing above slow EMA is the basic bullish trigger.",
    },
    "trend_ema_slow": {
        "description": (
            "Exponential Moving Average (slow period). Smoother version used as the trend baseline."
        ),
        "formula": "EMA_t = α × price_t + (1 - α) × EMA_{t-1}\n  N for 'slow' is typically 26.",
        "example": "Price > slow EMA → up-trend bias.",
    },
    "trend_ichimoku_a": {
        "description": (
            "Ichimoku Senkou Span A. Mid-line of the Kumo cloud. Forecasts support / "
            "resistance 26 periods ahead."
        ),
        "formula": "Span A = (Tenkan + Kijun) / 2  (plotted 26 periods forward)",
        "example": "Price above the cloud (Span A > Span B and price > Span A) → bullish bias.",
    },
    "trend_ichimoku_b": {
        "description": (
            "Ichimoku Senkou Span B. The cloud's other boundary; slower to move than Span A."
        ),
        "formula": "Span B = (highest_high_52 + lowest_low_52) / 2  (plotted 26 periods forward)",
        "example": "Span A above Span B is a bullish cloud; below is bearish.",
    },
    "trend_ichimoku_base": {
        "description": (
            "Ichimoku Kijun-sen (base line). Mid-point of the highest high / lowest low over 26 periods. "
            "Used as the standard mid-term trend reference."
        ),
        "formula": "Kijun = (highest_high_26 + lowest_low_26) / 2",
        "example": "Price above Kijun → up-trend; below → down-trend.",
    },
    "trend_ichimoku_conv": {
        "description": (
            "Ichimoku Tenkan-sen (conversion line). The shorter mid-point line; faster than Kijun."
        ),
        "formula": "Tenkan = (highest_high_9 + lowest_low_9) / 2",
        "example": "Tenkan crossing above Kijun is the Ichimoku 'TK cross' bullish signal.",
    },
    "trend_kst": {
        "description": (
            "Know Sure Thing. Smoothed sum of four ROC series of different lookbacks. "
            "Slow trend-strength oscillator."
        ),
        "formula": "KST = sum_{i=1..4} (SMA(ROC_i) × weight_i)",
        "example": "KST crossing above zero or its signal line → long signal.",
    },
    "trend_kst_diff": {
        "description": "KST minus its signal line — the 'histogram' difference between fast and slow KST.",
        "formula": "KST diff = KST - KST signal",
        "example": "KST diff > 0 with rising magnitude is a strengthening up-trend.",
    },
    "trend_kst_sig": {
        "description": "The KST signal line — typically a 9-period SMA of KST itself.",
        "formula": "KST signal = SMA(KST, 9)",
        "example": "KST crossing above KST signal is a buy.",
    },
    "trend_macd": {
        "description": (
            "Moving Average Convergence Divergence. Difference between fast and slow EMAs. "
            "The most-watched momentum-trend hybrid in technical analysis."
        ),
        "formula": "MACD = EMA(close, 12) - EMA(close, 26)",
        "example": "MACD > 0 → fast EMA above slow EMA (up-trend); zero crosses are entry/exit triggers.",
    },
    "trend_macd_diff": {
        "description": "MACD histogram — MACD minus its signal line. Visualises crossover strength.",
        "formula": "MACD diff = MACD - MACD signal",
        "example": "Positive and growing → strengthening up-trend; turning negative → weakening / reversal.",
    },
    "trend_macd_signal": {
        "description": "MACD signal line — a 9-period EMA of MACD itself.",
        "formula": "MACD signal = EMA(MACD, 9)",
        "example": "MACD crossing above MACD signal is the textbook bullish trigger.",
    },
    "trend_mass_index": {
        "description": (
            "Mass Index. Detects trend reversals by spotting bulges in the high-low range. "
            "Doesn't care about direction — reads the volatility envelope."
        ),
        "formula": "MI = sum_{i=1..25} (EMA(H-L, 9) / EMA(EMA(H-L, 9), 9))",
        "example": "MI rising above 27 then falling back below 26.5 is the classic 'reversal bulge'.",
    },
    "trend_psar_down": {
        "description": (
            "Parabolic SAR — the descending dot series active during a down-trend. "
            "Acts as a trailing stop above price while shorting."
        ),
        "formula": "PSAR_t = PSAR_{t-1} - AF × (PSAR_{t-1} - extreme_high)\n  AF accelerates as the trend continues.",
        "example": "Close crossing above PSAR-down flips the trend; the indicator switches to PSAR-up.",
    },
    "trend_psar_up": {
        "description": (
            "Parabolic SAR — the ascending dot series active during an up-trend. "
            "Acts as a trailing stop below price."
        ),
        "formula": "PSAR_t = PSAR_{t-1} + AF × (extreme_low - PSAR_{t-1})",
        "example": "Close crossing below PSAR-up flips the trend.",
    },
    "trend_sma_fast": {
        "description": "Simple Moving Average (fast period). Plain mean of the last N closes.",
        "formula": "SMA = mean(close, N)\n  Fast N is typically 12.",
        "example": "Fast SMA crossing above slow SMA = 'golden cross' (in shorter periods, faster signal).",
    },
    "trend_sma_slow": {
        "description": "Simple Moving Average (slow period). The trend baseline most strategies reference.",
        "formula": "SMA = mean(close, N)\n  Slow N is typically 26.",
        "example": "Price above slow SMA → up-trend; below → down-trend.",
    },
    "trend_stc": {
        "description": (
            "Schaff Trend Cycle. Combines MACD with stochastic smoothing for a faster, less-noisy "
            "trend signal in [0, 100]."
        ),
        "formula": "STC = stochastic transform applied iteratively to MACD line.",
        "example": "STC crossing above 25 → up-trend confirmed; below 75 → down-trend confirmed.",
    },
    "trend_trix": {
        "description": (
            "TRIX. Triple-smoothed EMA's percentage rate of change. Filters out noise; useful "
            "for spotting trend exhaustion."
        ),
        "formula": "TRIX = ROC(triple_EMA(close, N))",
        "example": "TRIX crossing zero is a strong directional signal; divergences flag exhaustion.",
    },
    "trend_vortex_ind_diff": {
        "description": "Vortex Indicator — VI+ minus VI-. Direction + magnitude in one number.",
        "formula": "VI diff = VI_pos - VI_neg",
        "example": "VI diff > 0 → up-trend; sign change is a trend reversal.",
    },
    "trend_vortex_ind_neg": {
        "description": "Vortex Indicator (VI-). Captures downward true-range movement.",
        "formula": "VI- = sum(|low_t - high_{t-1}|, N) / sum(TR, N)",
        "example": "VI- > VI+ → down-trend.",
    },
    "trend_vortex_ind_pos": {
        "description": "Vortex Indicator (VI+). Captures upward true-range movement.",
        "formula": "VI+ = sum(|high_t - low_{t-1}|, N) / sum(TR, N)",
        "example": "VI+ > VI- → up-trend.",
    },

    # ----- volatility -----
    "volatility_atr": {
        "description": (
            "Average True Range. Measures market volatility as the EMA of the true range. "
            "Used for position sizing and volatility-adjusted stops, not direction."
        ),
        "formula": "TR = max(H-L, |H - prev_close|, |L - prev_close|)\nATR = EMA(TR, 14)",
        "example": "Higher ATR → wider stops needed; ATR doubling is often a regime change.",
    },
    "volatility_bbh": {
        "description": "Upper Bollinger Band. Two standard deviations above the moving average.",
        "formula": "BBH = SMA(close, 20) + 2 × σ(close, 20)",
        "example": "Close > BBH → 'overbought' in mean-reversion frame; trend continuation in momentum frame.",
    },
    "volatility_bbhi": {
        "description": "Bollinger Bands High Indicator. Boolean 1/0 signal that the close is above the upper band.",
        "formula": "BBHI = 1 if close > BBH else 0",
        "example": "BBHI = 1 is a binary 'bands breached upward' signal.",
    },
    "volatility_bbl": {
        "description": "Lower Bollinger Band.",
        "formula": "BBL = SMA(close, 20) - 2 × σ(close, 20)",
        "example": "Close < BBL → oversold (mean-reversion) or breakdown (trend).",
    },
    "volatility_bbli": {
        "description": "Bollinger Bands Low Indicator. Boolean 1/0 signal that the close is below the lower band.",
        "formula": "BBLI = 1 if close < BBL else 0",
        "example": "BBLI = 1 is a binary 'bands breached downward' signal.",
    },
    "volatility_bbm": {
        "description": "Bollinger Bands Middle. The 20-period SMA at the centre.",
        "formula": "BBM = SMA(close, 20)",
        "example": "Used as the mean-reversion target after a band touch.",
    },
    "volatility_bbp": {
        "description": (
            "Bollinger Bands %B. Where the close sits inside the bands, normalised. "
            "0 = at the lower band, 1 = at the upper band."
        ),
        "formula": "%B = (close - BBL) / (BBH - BBL)",
        "example": "%B > 1 → above upper band; %B < 0 → below lower band.",
    },
    "volatility_bbw": {
        "description": (
            "Bollinger Bands Width. Spread between upper and lower bands as a fraction of the middle. "
            "A volatility regime indicator."
        ),
        "formula": "BBW = (BBH - BBL) / BBM",
        "example": "BBW shrinking → 'volatility squeeze' often preceding a breakout.",
    },
    "volatility_dch": {
        "description": "Donchian Channel High. Highest high over the lookback window.",
        "formula": "DCH = max(high, N)\n  Default N = 20.",
        "example": "Close breaking above DCH is the classic Turtle Traders entry signal.",
    },
    "volatility_dcl": {
        "description": "Donchian Channel Low. Lowest low over the lookback window.",
        "formula": "DCL = min(low, N)",
        "example": "Close breaking below DCL is a short / exit signal.",
    },
    "volatility_dcm": {
        "description": "Donchian Channel Mid. Midpoint of high and low.",
        "formula": "DCM = (DCH + DCL) / 2",
        "example": "Used as the equilibrium reference; price > DCM → upper half of the range.",
    },
    "volatility_dcp": {
        "description": "Donchian Channel %P. Position within the Donchian channel; 0 = at low, 1 = at high.",
        "formula": "DCP = (close - DCL) / (DCH - DCL)",
        "example": "DCP > 0.8 → close near recent high; < 0.2 → near recent low.",
    },
    "volatility_dcw": {
        "description": "Donchian Channel Width. Range size as fraction of the midpoint.",
        "formula": "DCW = (DCH - DCL) / DCM",
        "example": "Widening DCW → volatility expansion; narrowing → quiet period.",
    },
    "volatility_kcc": {
        "description": "Keltner Channel Centre. The 20-period EMA at the channel's middle.",
        "formula": "KCC = EMA(close, 20)",
        "example": "Mean-reversion target after a band touch.",
    },
    "volatility_kch": {
        "description": "Keltner Channel High. ATR-based upper envelope around the EMA centre.",
        "formula": "KCH = EMA(close, 20) + 2 × ATR",
        "example": "Close above KCH is a bullish breakout under the Keltner framework.",
    },
    "volatility_kchi": {
        "description": "Keltner Channel High Indicator. Binary signal that the close is above the upper Keltner band.",
        "formula": "KCHI = 1 if close > KCH else 0",
        "example": "KCHI = 1 → upper-band breach.",
    },
    "volatility_kcl": {
        "description": "Keltner Channel Low. ATR-based lower envelope.",
        "formula": "KCL = EMA(close, 20) - 2 × ATR",
        "example": "Close below KCL → bearish breakout.",
    },
    "volatility_kcli": {
        "description": "Keltner Channel Low Indicator. Binary signal that the close is below the lower Keltner band.",
        "formula": "KCLI = 1 if close < KCL else 0",
        "example": "KCLI = 1 → lower-band breach.",
    },
    "volatility_kcp": {
        "description": "Keltner Channel %P. Where the close sits inside the Keltner channel.",
        "formula": "KCP = (close - KCL) / (KCH - KCL)",
        "example": "KCP > 1 → above upper band; < 0 → below lower band.",
    },
    "volatility_kcw": {
        "description": "Keltner Channel Width. ATR-derived band width as fraction of centre.",
        "formula": "KCW = (KCH - KCL) / KCC",
        "example": "Compressing KCW often precedes a directional break.",
    },
    "volatility_ui": {
        "description": (
            "Ulcer Index. Measures downside volatility — depth and duration of drawdowns "
            "from recent highs. Higher = more painful equity curve."
        ),
        "formula": "UI = sqrt(mean(((close - max_close_N) / max_close_N × 100)^2, N))",
        "example": "UI is most useful comparing two strategies at equal returns — lower UI is preferred.",
    },

    # ----- volume -----
    "volume": {
        "description": "Raw bar volume. Compared against its own moving average / previous value to flag activity spikes.",
        "formula": "volume_t = volume traded during bar t",
        "example": "volume > volume_previous → unusually active bar; pair with a directional signal.",
    },
    "volume_adi": {
        "description": (
            "Accumulation / Distribution Index. Volume-weighted measure of money flow direction. "
            "Cumulative — track its slope, not its level."
        ),
        "formula": "ADI_t = ADI_{t-1} + ((close - low) - (high - close)) / (high - low) × volume",
        "example": "Rising ADI → accumulation (bullish); falling → distribution (bearish).",
    },
    "volume_cmf": {
        "description": (
            "Chaikin Money Flow. ADI normalised by volume over a rolling window. Bounded near [-1, 1]."
        ),
        "formula": "CMF = sum(MFV, 20) / sum(volume, 20)\n  MFV = ((C-L) - (H-C)) / (H-L) × V",
        "example": "CMF > 0.25 → strong buying pressure; < -0.25 → strong selling pressure.",
    },
    "volume_em": {
        "description": (
            "Ease of Movement. Combines price change and volume into a single number — high values "
            "mean price moved a lot on relatively low volume."
        ),
        "formula": "EM = ((H + L)/2 - prev_(H+L)/2) × (H - L) / volume",
        "example": "EM > 0 → price rising on light volume; < 0 → falling on light volume.",
    },
    "volume_fi": {
        "description": "Force Index. Combines volume and price change to gauge buying / selling pressure.",
        "formula": "FI = (close - prev_close) × volume",
        "example": "Sustained positive FI confirms an up-trend; spikes flag climactic activity.",
    },
    "volume_mfi": {
        "description": (
            "Money Flow Index. Volume-weighted RSI — bounded in [0, 100], same overbought / oversold "
            "interpretation but informed by volume."
        ),
        "formula": "MFI = 100 - 100 / (1 + money_ratio)\n  money_ratio = pos_money_flow / neg_money_flow",
        "example": "MFI ≥ 80 → overbought with volume confirmation; ≤ 20 → oversold.",
    },
    "volume_nvi": {
        "description": (
            "Negative Volume Index. Tracks price moves on bars where volume *decreased*. "
            "The thesis: smart money moves on low-volume days."
        ),
        "formula": "NVI_t = NVI_{t-1} × (1 + pct_change(close)) if volume_t < volume_{t-1} else NVI_{t-1}",
        "example": "NVI > its 255-period EMA → smart-money accumulation.",
    },
    "volume_obv": {
        "description": (
            "On-Balance Volume. Cumulative volume — adds when close > prev close, subtracts when "
            "close < prev close. Slope matters more than level."
        ),
        "formula": "OBV_t = OBV_{t-1} + sign(close - prev_close) × volume",
        "example": "OBV trending up while price chops sideways → accumulation; divergence with price → reversal.",
    },
    "volume_sma_em": {
        "description": "14-period SMA of Ease of Movement. Smoothed volume-flow signal.",
        "formula": "SMA_EM = SMA(EM, 14)",
        "example": "SMA_EM crossing zero is a less-noisy version of the raw EM cross.",
    },
    "volume_vpt": {
        "description": "Volume Price Trend. Cumulative volume weighted by percentage price change.",
        "formula": "VPT_t = VPT_{t-1} + volume × (close - prev_close) / prev_close",
        "example": "VPT trending up → accumulation; divergence with price → potential reversal.",
    },
    "volume_vwap": {
        "description": (
            "Volume Weighted Average Price. The fair-value price level adjusted for volume; "
            "intraday traders use it as a benchmark."
        ),
        "formula": "VWAP = sum(typical_price × volume) / sum(volume)\n  typical_price = (H + L + C) / 3",
        "example": "Price > VWAP → buyers in control; < VWAP → sellers in control.",
    },
}


def get_doc(indicator: str) -> Doc:
    """Return the doc entry for ``indicator`` or a generic fallback."""
    return INDICATOR_DOCS.get(
        indicator,
        {
            "description": (
                f"`{indicator}` is in the catalogue but isn't documented here yet. "
                "See the `ta` library docs (https://technical-analysis-library-in-python."
                "readthedocs.io/) for the precise formula and parameters."
            ),
            "formula": "(no formula on file)",
            "example": "(no example on file)",
        },
    )
