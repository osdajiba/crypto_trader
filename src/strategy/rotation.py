# src/strategy/rotation.py

import pandas as pd
import ccxt

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# =====创建ccxt交易所
exchange = ccxt.okex()  # 其他交易所为huobipro, binance, okex

# =====设置参数
time_interval = '1d'
N = 20  # 计算最近N天的涨跌幅

# =====获取最新数据，计算涨跌幅
change_dict = {}
for symbol in ['BTC/USDT', 'ETH/USDT']:
    # 获取数据    
    df = exchange.fetch_ohlcv(symbol=symbol, timeframe=time_interval, limit=N + 5)

    # 整理数据
    df = pd.DataFrame(df, dtype=float)  # 将数据转换为dataframe
    df.rename(columns={0: 'MTS', 1: 'open', 2: 'high',
                       3: 'low', 4: 'close', 5: 'volume'}, inplace=True)  # 重命名
    df['candle_begin_time'] = pd.to_datetime(df['MTS'], unit='ms')  # 整理时间
    df = df[['candle_begin_time', 'open', 'high', 'low', 'close']]  # 整理列的顺序
    df['最近N天涨跌幅'] = df['close'].pct_change(N)

    change_dict[symbol] = df.iloc[-1]['最近N天涨跌幅']

    # print(df)

print(change_dict)

# =====判断操作
# 两者都<0
if change_dict['BTC/USDT'] < 0 and change_dict['ETH/USDT'] < 0:
    print('比特币和以太坊涨幅都<0，空仓')

# 并非两者都<0时，且比特币涨得多
elif change_dict['BTC/USDT'] > change_dict['ETH/USDT']:
    print('比特币涨幅大于0且大于以太坊涨幅，买入比特币')

# 并非两者都<0时，且以太坊涨得多
elif change_dict['BTC/USDT'] < change_dict['ETH/USDT']:
    print('以太坊涨幅大于0且大于比特币涨幅，买入以太坊')

# # =====手工操作
# # 简单实用：每天8点之前手动运行本程序，按照说明手工操作。提前一点没事。
# # 高级一点：每天7点59分自动运行程序，发送操作说明到邮箱、或者钉钉、或者企业微信。


# # =====程序操作
# # 每天产生信号后，对比当前持仓，进行卖出、买入等操作
