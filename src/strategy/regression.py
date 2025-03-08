# src/strategy/Regression.py

import numpy as np
import pandas as pd
import statsmodels.api as sm


class Regression:
    def __init__(self, stocklist, sectormap, selectindexname, StockReturn, selectIndexReturn, mom30d, mom5m, r_lms,
                 r_hml):
        self.stocklist = stocklist
        self.sectormap = sectormap
        self.selectindexname = selectindexname
        self.StockReturn = StockReturn
        self.selectIndexReturn = selectIndexReturn
        self.mom30d = mom30d
        self.mom5m = mom5m
        self.r_lms = r_lms
        self.r_hml = r_hml

        # Initialize results matrices
        self.results1 = pd.DataFrame(columns=["Instrument ID", "Beta - Industry Return", "T-stat - Industry Return",
                                              "Ordinary R-squared", "Industry Index"])
        self.results2 = pd.DataFrame(columns=["Instrument ID", "Beta - Industry Return", "T-stat - Industry Return",
                                              "Beta - LMS", "T-stat - LMS", "Beta - HML", "T-stat - HML",
                                              "Ordinary R-squared", "Industry Index"])
        self.results3 = pd.DataFrame(columns=["Instrument ID", "Beta - Industry Return", "T-stat - Industry Return",
                                              "Beta - LMS", "T-stat - LMS", "Beta - HML", "T-stat - HML",
                                              "Beta - Mom30d", "T-stat - Mom30d", "Beta - Mom5m", "T-stat - Mom5m",
                                              "Ordinary R-squared", "Industry Index"])

    def fit_linear_model(self, X, y):
        X = sm.add_constant(X)  # Add a constant term to the predictor
        model = sm.OLS(y, X).fit()
        return model

    def analyze_factors(self):
        for i, ticker in enumerate(self.stocklist):
            print(f"Analyzing {ticker} ({i + 1}/{len(self.stocklist)})")

            # Get sector index
            ix = np.where(self.sectormap[:, 0] == ticker)[0][0]
            sindex = self.sectormap[ix, 4]

            # Check if the sector index is in the selected index names
            if sindex not in self.selectindexname:
                continue

            # Get stock and index returns
            sr = self.StockReturn[:, i]
            ix = np.where(self.selectindexname == sindex)[0][0]
            ir = self.selectIndexReturn[:, ix]

            # Define conditions for filtering data
            xf = np.abs(sr) < 0.099

            # Conduct linear regression analysis based on the number of factors
            if np.sum(xf) > 1:
                if hasattr(self, 'r_lms') and hasattr(self, 'r_hml'):
                    X = np.column_stack([ir[xf], self.r_lms[xf], self.r_hml[xf]])
                    model = self.fit_linear_model(X, sr[xf])
                    self.results2 = self.results2.append(
                        [i, model.params[1], model.tvalues[1], model.params[2], model.tvalues[2],
                         model.params[3], model.tvalues[3], model.rsquared, sindex],
                        ignore_index=True
                    )
                else:
                    X = ir[xf]
                    model = self.fit_linear_model(X, sr[xf])
                    self.results1 = self.results1.append(
                        [i, model.params[1], model.tvalues[1], model.rsquared, sindex],
                        ignore_index=True
                    )
            else:
                if hasattr(self, 'r_lms') and hasattr(self, 'r_hml'):
                    self.results2 = self.results2.append(
                        [i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, sindex],
                        ignore_index=True
                    )
                else:
                    self.results1 = self.results1.append(
                        [i, np.nan, np.nan, np.nan, sindex],
                        ignore_index=True
                    )

            if hasattr(self, 'mom30d') and hasattr(self, 'mom5m'):
                if np.sum(xf) > 1:
                    X = np.column_stack([ir[xf], self.r_lms[xf], self.r_hml[xf], self.mom30d[xf], self.mom5m[xf]])
                    model = self.fit_linear_model(X, sr[xf])
                    self.results3 = self.results3.append(
                        [i, model.params[1], model.tvalues[1], model.params[2], model.tvalues[2],
                         model.params[3], model.tvalues[3], model.params[4], model.tvalues[4],
                         model.params[5], model.tvalues[5], model.rsquared, sindex],
                        ignore_index=True
                    )
                else:
                    self.results3 = self.results3.append(
                        [i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                         sindex],
                        ignore_index=True
                    )

    def summary_statistics(self, results, ind_rsq, ind_industry):
        # Filter results based on R-squared values
        filter_condition = (results.iloc[:, ind_rsq] > 0.2) & (results.iloc[:, ind_rsq] < 1)
        filtered_results = results[filter_condition]

        # Calculate mean and t-statistics for beta values grouped by industry index
        xbar_columns = [f"Beta - {col}" for col in results.columns[1::2]]
        tbar_columns = [f"T-stat - {col}" for col in results.columns[1::2]]
        xbar, _, _ = filtered_results.groupby(results.iloc[:, ind_industry])[results.columns[1::2]].agg(
            ['mean', 'size'])
        tbar, _, _ = filtered_results.groupby(results.iloc[:, ind_industry])[results.columns[2::2]].agg(
            ['mean', 'size'])

        return xbar, tbar
