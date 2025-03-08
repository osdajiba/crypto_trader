# src/strategy/three_factors.py

import logging
from lib import factors_lib,auto_order,BN_market


class SPBBT:
    def __init__(self, coin_data, stop_loss_percent=0.08, order_timeout=1):
        """
        Initialize SPBBT strategy parameters.

        :param stop_loss_percent: Percentage for stop-loss.
        :param order_timeout: Timeout for order execution, time to cancel the order.
        """
        self.all_data = coin_data
        self.order_signal = None
        self.open_signal = None
        self.close_signal = None
        self.order_price = None
        self.prev_high = None
        self.prev_low = None
        self.stop_loss_percent = stop_loss_percent
        self.order_duration = None
        self.order_timer = None
        self.order_timeout = order_timeout

    def open_position(self, signal, price, exe_price):
        # Open position logic
        auto_order.place_limit_order()
        pass

    def close_position(self):
        # Close position logic

        pass

    def cancel_order(self, current_vwap, current_spt, bb_lower, bb_upper):
        # Cancel order logic
        if (
                (self.open_signal == "Buy" and current_vwap > bb_lower and current_spt > bb_lower)
                or (self.open_signal == "Sell" and current_vwap > bb_upper and current_spt > bb_lower)
        ):
            # Cancel order
            auto_order.place_limit_order()
            # Re-enter order
            pass
        else:
            self.open_signal = None

    def update_counters(self):
        # Update order timers
        if self.open_signal is not None:
            self.order_timer += 1
        self.order_duration += 1

    def trade_logic(self, current_close, current_high, current_low, current_vwap, current_spt, bb_upper, bb_lower):
        """
        :param current_close: Close _price of the current candle.
        :param current_high: High _price of the current candle.
        :param current_low: Low _price of the current candle.

        :param current_vwap: Volume Weighted Average Price (VWAP) of the current candle.
        :param current_spt: Signal Price Threshold (SPT) of the current candle.
        :param bb_upper: Upper Bollinger Band value.
        :param bb_lower: Lower Bollinger Band value.
        """

        # Extract strategy parameters from self
        order_signal = self.order_signal
        order_price = self.order_price
        prev_high = self.prev_high
        prev_low = self.prev_low
        stop_loss_percent = self.stop_loss_percent
        order_duration = self.order_duration
        order_timer = self.order_timer
        order_timeout = self.order_timeout

        # 主交易逻辑

        # 出现: 收盘价格首次跌出布林带下轨
        if current_close < bb_lower and order_signal is None:
            # 开多信号
            if current_vwap > bb_lower and current_spt > bb_lower:
                # 开多条件: P2 最低价保持在 BB- 下方
                if current_low < bb_lower:
                    # 下单: 0.03 BTC 前K最低价
                    order_signal = "Buy"
                    order_price = current_low
                    prev_low = current_low
                    self.open_position(order_signal, order_price, prev_low)
                    order_duration = 0
                    order_timer = 0

            # 出现: 收盘价格首次突破布林带上轨
        elif current_close > bb_upper and order_signal is None:
            # 开空信号
            if current_vwap > bb_upper and current_spt > bb_lower:
                # 开空条件: P 最高价保持在 BB+ 上方
                if current_high > bb_upper:
                    # 下单: 0.03 BTC 前K最高价
                    order_signal = "Sell"
                    order_price = current_high
                    prev_high = current_high
                    self.open_position(order_signal, order_price, prev_high)
                    order_duration = 0
                    order_timer = 0

            # 止损信号
        elif order_signal is not None and current_close < order_price * (1 - stop_loss_percent):
            # 止损条件: 单笔亏损8%
            # 下单: 0.03 BTC 上一K线最高价(最低价) - 手续费
            self.close_position()
            order_signal = None

            # 止盈信号
        elif order_signal is not None and current_spt > bb_upper:
            # 止盈条件: SPT上穿越布林带
            # 下单: 0.03 BTC 上一K线最高价(最低价) - 手续费
            self.close_position()
            order_signal = None

            # 撤单信号
        elif order_signal is not None and order_timer >= order_timeout:
            # 撤单条件: 1/2 K线时长内，订单未成交
            if order_duration <= order_timeout / 2:
                self.cancel_order(current_vwap, current_spt, bb_lower, bb_upper)

        # Update counters
        self.update_counters()

        # Update class attributes
        self.order_signal = order_signal
        self.order_price = order_price
        self.prev_high = prev_high
        self.prev_low = prev_low
        self.order_duration = order_duration
        self.order_timer = order_timer


# 交易逻辑的实现，与之前的示例相同

def process_trade(self, current_close, current_high, current_low, current_vwap, current_spt, bb_upper, bb_lower):
    current_vwap = factors_lib.vwap(self.all_data)
    current_spt = factors_lib.SuperTrend(self.all_data)
    bb = factors_lib.BB(self.all_data)

    self.trade_logic(current_close, current_high, current_low, current_vwap, current_spt, bb["upper_band"],
                     bb["upper_band"])

    # 更新订单计时器
    # if self.order_signal is not None:
    #     self.order_timer += 1
    # self.order_duration += 1

# 使用示例
# strategy = lib.SPBBT()
# 在每个新的K线上调用交易逻辑
# strategy.process_trade(current_close, current_high, current_low, current_vwap, current_spt, bb_upper, bb_lower)
