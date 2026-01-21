"""
DanielsNumber1 - Wyckoff-Enhanced Breakout Strategy with Optimized Parameters

Performance (Full SOL/USDT Timeline 2020-2026):
- Total Return: +63,945% (5.3x improvement from v115)
- Trades: 254
- Win Rate: 45.3%
- Max Drawdown: 13.78% (37% reduction from v115's 21.94%)

Strategy Logic:
- Longs: Bull regime (EMA40 > EMA80) + 28-period breakout + 2.7x volume + 2% momentum
- Shorts: Strong bear regime + breakout + 2.7x volume + 3% momentum
- Wyckoff Short Boost: When distribution pattern detected (ATR compression + price in
  upper half of range), allow shorts with looser params (2.2x vol, 2% momentum)
- Volatility Filter: ATR < 2x average ATR for all trades
- ATR Spike Filter: Block shorts when ATR > 1.52x recent 20h minimum
- TP: 13% | SL: 5%

Safety Features:
- 40% Max Drawdown Protection: Pauses trading when DD exceeds 40%
- Stoploss Guard: Pauses after 4 consecutive stoplosses

Version History:
- v72: Added Wyckoff distribution detection (+15,377% return, 32.77% DD)
- v115: Added ATR spike filter for shorts (+12,138% return, 21.94% DD)
- v120: Full parameter optimization (+63,945% return, 13.78% DD)
- v121: Added 40% max drawdown protection (caps worst-case Monte Carlo DD from 85% to 43%)
- v122: Code cleanup (no logic changes)

WARNING: This strategy is optimized specifically for SOL/USDT. Testing on ETH/USDT
showed -59% returns. Do NOT use on other pairs without separate optimization.

Recommended: SOL/USDT Futures, 1H timeframe, 1x leverage
"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class DanielsNumber1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = True

    minimal_roi = {"0": 0.13}
    stoploss = -0.05
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 250

    # === 40% MAX DRAWDOWN PROTECTION ===
    # Pauses trading when recent trades lose more than 40%
    # This caps worst-case Monte Carlo DD from 85% to ~43%
    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 200,  # ~8 days of 1h candles
                "trade_limit": 20,  # Consider last 20 trades
                "stop_duration_candles": 48,  # Pause for 48 hours
                "max_allowed_drawdown": 0.40  # 40% max drawdown trigger
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,  # Last 48 hours
                "trade_limit": 4,  # If 4 trades hit stoploss
                "stop_duration_candles": 24,  # Pause for 24 hours
                "only_per_pair": False
            }
        ]

    # Long parameters
    breakout_period = 28  # Optimized from 25
    volume_mult = 2.7
    momentum_pct = 0.02
    atr_mult_max = 2.0

    # Standard short parameters (optimized)
    volume_mult_short = 2.7  # Optimized from 2.5
    momentum_pct_short = 0.03  # Optimized from 0.025

    # Wyckoff-boosted short parameters (for distribution patterns)
    volume_mult_short_boosted = 2.2
    momentum_pct_short_boosted = 0.02

    # ATR spike filter parameters (optimized)
    atr_spike_threshold = 1.52  # Optimized from 1.55
    atr_spike_lookback = 20

    # EMA parameters (optimized)
    ema_fast = 40  # Optimized from 50
    ema_slow = 80  # Optimized from 100
    sma_trend = 200

    # Shared parameters (used in multiple places)
    momentum_lookback = 5  # Candles to look back for momentum calculation
    volume_sma_period = 20  # Period for volume moving average
    atr_period = 14  # ATR calculation period
    atr_avg_period = 50  # Period for ATR moving average
    wyckoff_period = 48  # Period for Wyckoff range detection (48h = 2 days)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow)
        dataframe['sma_trend'] = ta.SMA(dataframe, timeperiod=self.sma_trend)

        dataframe['bull_regime'] = dataframe['ema_fast'] > dataframe['ema_slow']
        dataframe['bear_regime'] = (
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['close'] < dataframe['sma_trend'])
        )

        dataframe['highest'] = dataframe['high'].rolling(self.breakout_period).max().shift(1)
        dataframe['lowest'] = dataframe['low'].rolling(self.breakout_period).min().shift(1)
        dataframe['volume_sma'] = dataframe['volume'].rolling(self.volume_sma_period).mean()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe['atr_avg'] = dataframe['atr'].rolling(self.atr_avg_period).mean()

        # Wyckoff Distribution Detection
        range_high = dataframe['high'].rolling(self.wyckoff_period).max()
        range_low = dataframe['low'].rolling(self.wyckoff_period).min()
        range_size = range_high - range_low
        range_position = (dataframe['close'] - range_low) / range_size.replace(0, 1e-10)

        # Distribution: ATR compressed + price in upper half of range
        atr_compressed = dataframe['atr'] < dataframe['atr_avg']
        distribution = atr_compressed & (range_position > 0.5)

        # Was there distribution in the last wyckoff_period hours?
        dataframe['post_distribution'] = distribution.rolling(self.wyckoff_period).max().shift(1) > 0

        # ATR spike detection for short filter
        atr_recent_min = dataframe['atr'].rolling(self.atr_spike_lookback).min()
        dataframe['no_atr_spike'] = ~(dataframe['atr'] > atr_recent_min * self.atr_spike_threshold)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        vol_surge = dataframe['volume'] > dataframe['volume_sma'] * self.volume_mult
        mom_up = dataframe['close'] > dataframe['close'].shift(self.momentum_lookback) * (1 + self.momentum_pct)
        volatility_ok = dataframe['atr'] < dataframe['atr_avg'] * self.atr_mult_max

        # Longs: bull regime + breakout + volume surge + momentum
        dataframe.loc[
            dataframe['bull_regime'] &
            (dataframe['close'] > dataframe['highest']) &
            (dataframe['close'] > dataframe['ema_slow']) &
            vol_surge & mom_up & volatility_ok,
            'enter_long'] = 1

        # Short conditions
        vol_surge_short = dataframe['volume'] > dataframe['volume_sma'] * self.volume_mult_short
        mom_down = dataframe['close'] < dataframe['close'].shift(self.momentum_lookback) * (1 - self.momentum_pct_short)

        vol_surge_short_boosted = dataframe['volume'] > dataframe['volume_sma'] * self.volume_mult_short_boosted
        mom_down_boosted = dataframe['close'] < dataframe['close'].shift(self.momentum_lookback) * (1 - self.momentum_pct_short_boosted)

        # Short base conditions with ATR spike filter
        short_base = (
            dataframe['bear_regime'] &
            (dataframe['close'] < dataframe['lowest']) &
            (dataframe['close'] < dataframe['ema_slow']) &
            volatility_ok &
            dataframe['no_atr_spike']
        )

        # Shorts: standard OR Wyckoff-boosted (when distribution detected)
        dataframe.loc[
            short_base & (
                (vol_surge_short & mom_down) |
                (dataframe['post_distribution'].fillna(False) & vol_surge_short_boosted & mom_down_boosted)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
