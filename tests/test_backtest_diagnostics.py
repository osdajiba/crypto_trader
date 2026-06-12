import unittest


class BacktestDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_estimates_gross_return_cost_drag_and_risk_ratios(self):
        from src.reporting.backtest_diagnostics import diagnose_backtest_report

        report = {
            "initial_capital": 100000,
            "final_equity": 99950,
            "total_return": -50,
            "total_return_pct": -0.05,
            "max_drawdown_pct": 0.1,
            "total_trades": 10,
            "current_positions": {},
            "costs": {
                "total_transaction_cost": 45,
                "total_notional": 20000,
            },
        }

        diagnostics = diagnose_backtest_report(report)

        self.assertEqual(diagnostics["net_return"], -50)
        self.assertEqual(diagnostics["net_return_pct"], -0.05)
        self.assertEqual(diagnostics["gross_return_estimate"], -5)
        self.assertEqual(diagnostics["gross_return_pct_estimate"], -0.005)
        self.assertEqual(diagnostics["total_transaction_cost"], 45)
        self.assertEqual(diagnostics["cost_drag_pct_of_initial"], 0.045)
        self.assertEqual(diagnostics["cost_to_abs_net_return"], 0.9)
        self.assertEqual(diagnostics["average_trade_notional"], 2000)
        self.assertEqual(diagnostics["turnover_pct_of_initial"], 20)
        self.assertEqual(diagnostics["return_to_drawdown"], -0.5)
        self.assertFalse(diagnostics["has_open_positions"])

    def test_diagnostics_handles_zero_drawdown_and_zero_net_return(self):
        from src.reporting.backtest_diagnostics import diagnose_backtest_report

        diagnostics = diagnose_backtest_report({
            "initial_capital": 100000,
            "total_return": 0,
            "total_return_pct": 0,
            "max_drawdown_pct": 0,
            "total_trades": 0,
            "current_positions": {"BTC/USDT": 0},
            "costs": {},
        })

        self.assertIsNone(diagnostics["cost_to_abs_net_return"])
        self.assertIsNone(diagnostics["return_to_drawdown"])
        self.assertFalse(diagnostics["has_open_positions"])

    def test_diagnostics_describes_round_trip_trade_behavior(self):
        from src.reporting.backtest_diagnostics import diagnose_backtest_report

        diagnostics = diagnose_backtest_report({
            "initial_capital": 100000,
            "total_return": 8,
            "total_return_pct": 0.008,
            "max_drawdown_pct": 0.01,
            "total_trades": 2,
            "current_positions": {},
            "costs": {"total_transaction_cost": 2, "total_notional": 210},
            "trades": [
                {
                    "timestamp": "2025-01-01 00:00:00+00:00",
                    "symbol": "BTC/USDT",
                    "action": "buy",
                    "quantity": 1,
                    "price": 100,
                    "commission": 1,
                },
                {
                    "timestamp": "2025-01-01 00:30:00+00:00",
                    "symbol": "BTC/USDT",
                    "action": "sell",
                    "quantity": 1,
                    "price": 110,
                    "commission": 1,
                },
            ],
        })

        self.assertEqual(diagnostics["round_trip_count"], 1)
        self.assertEqual(diagnostics["round_trip_win_rate"], 100.0)
        self.assertEqual(diagnostics["average_holding_minutes"], 30.0)
        self.assertEqual(diagnostics["average_round_trip_return"], 8.0)
        self.assertEqual(diagnostics["average_round_trip_return_pct"], 8.0)


if __name__ == "__main__":
    unittest.main()
