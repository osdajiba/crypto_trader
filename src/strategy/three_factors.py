# src/strategy/three_factors.py

import pandas as pd
import numpy as np


def data_generate(IndexDateLine, StockDateLine, StockPriceMatrix, IndexPriceMatrix):
    dateline = pd.to_datetime(list(set(IndexDateLine) & set(StockDateLine)))
    dateline = dateline.sort_values()

    stock_mask = StockDateLine.isin(dateline)
    StockPriceMatrix = StockPriceMatrix.loc[:, stock_mask]
    StockReturn = StockPriceMatrix.pct_change().iloc[1:]

    index_mask = IndexDateLine.isin(dateline)
    IndexPriceMatrix = IndexPriceMatrix.loc[:, index_mask]
    IndexReturn = IndexPriceMatrix.pct_change().iloc[1:]

    # 30-day return, 180-day return
    oneplus_r = StockReturn + 1
    oneplus_r[oneplus_r.isna()] = 1
    cr = oneplus_r.cumprod()
    T30, T5M = 30, 150
    mom30dd = cr.iloc[T30:] / cr.iloc[:-T30 + 1] - 1
    r5m = cr.iloc[T5M:] / cr.iloc[:-T5M + 1] - 1

    dateline = dateline[T30 + T5M - 1:]
    StockReturn = StockReturn.iloc[T30 + T5M - 1:]
    IndexReturn = IndexReturn.iloc[T30 + T5M - 1:]

    mom30dd = mom30dd.iloc[T5M - 1:]
    r5m = r5m.iloc[:-T30 + 1]  # in effect, r5m is from 31-210 days in the past

    return dateline, StockReturn, IndexReturn, mom30dd, r5m


def calculate_momentum(returns, ranking_window, mom_window):
    mom = pd.DataFrame(index=returns.index, columns=returns.columns)

    for t in range(1, len(returns)):
        # Extract the previous day's return (t-1) and handle NaN values
        ttt = ranking_window.iloc[t - 1].copy()
        nan_mask = returns.iloc[t].isna()
        ttt_with_flag = ttt.copy()
        ttt_with_flag[nan_mask] = np.nan

        # Select the top and bottom 10% based on rank
        selected = ttt_with_flag[~nan_mask].nlargest(int(len(ttt_with_flag) * 0.1)).index.union(
            ttt_with_flag[~nan_mask].nsmallest(int(len(ttt_with_flag) * 0.1)).index)

        # Calculate momentum as the difference between the mean of top and bottom returns
        mom.iloc[t] = returns.iloc[t, selected].mean() - returns.iloc[t, ~selected].mean()

    return mom


def calculate_index_diff(StockReturn, indexlist, index_1, index_2):
    r_long = get_index_return(StockReturn, indexlist, index_1)
    r_short = get_index_return(StockReturn, indexlist, index_2)
    return r_long - r_short


def get_index_return(StockReturn, indexlist, target_index):
    tf, ix = np.isin(target_index, indexlist)
    return StockReturn.loc[:, ix]


def construct(IndexDateLine, StockDateLine, StockPriceMatrix, IndexPriceMatrix, IndexList):
    dateline, StockReturn, IndexReturn, mom30dd, r5m = data_generate(IndexDateLine, StockDateLine,
                                                                     StockPriceMatrix, IndexPriceMatrix)

    # Calculate 5m and 30d momentum
    mom_5m = calculate_momentum(StockReturn, ranking_window=r5m, mom_window=5)
    mom_30d = calculate_momentum(StockReturn, ranking_window=mom30dd, mom_window=30)

    # Calculate r_lms (SMB)
    # 000132.XSHG--上证100
    # 000044.XSHG--上证中盘;
    r_lms = calculate_index_diff(StockReturn, IndexList, '000132.XSHG', '000044.XSHG')

    # Calculate r_hml (HML)
    # 000029.XSHG--180价值
    # 000028.XSHG--180成长
    r_hml = calculate_index_diff(StockReturn, IndexList, '000029.XSHG', '000028.XSHG')

    return mom_5m, mom_30d, r_hml, r_lms, StockReturn, IndexReturn
