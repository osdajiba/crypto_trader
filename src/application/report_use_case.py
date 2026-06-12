from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.reporting.performance_analyzer import PerformanceAnalyzer
from src.reporting.backtest_diagnostics import diagnose_backtest_report
from src.reporting.strategy_chart import StrategyChartWriter
from src.reporting.writers import ReportWriter


class TradingReportUseCase:
    """生成、保存并记录交易模式报告。

    报告用例只读取 mode/context 中的组合状态和成交记录，不参与交易决策，
    也不修改组合账本。保留旧 JSON 字段和文件命名规则，保护现有报告使用方。
    """

    def __init__(self, context: Any):
        self.context = context
        self.performance_analyzer = PerformanceAnalyzer()

    def generate(self) -> Dict[str, Any]:
        """根据当前组合状态生成报告字典。"""
        initial_capital = self.context.config.get(
            self.context.mode_name,
            "initial_capital",
            default=self.context.config.get("default_config", "user_config", "initial_cash", default=100000),
        )

        domain_reporter = getattr(self.context, "domain_reporter", None)
        if domain_reporter is not None and hasattr(domain_reporter, "fills") and hasattr(domain_reporter, "snapshots"):
            report = self.performance_analyzer.analyze(
                fills=list(domain_reporter.fills),
                snapshots=list(domain_reporter.snapshots),
                initial_capital=initial_capital,
            )
            self.context._add_mode_specific_metrics(report)
            self._attach_diagnostics(report)
            return report

        final_equity = self.context._calculate_equity()
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

        trades = self.context.state["trades"]
        buy_trades = len([trade for trade in trades if self._trade_action_value(trade) == "buy"])
        sell_trades = len([trade for trade in trades if self._trade_action_value(trade) == "sell"])

        report = {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": self.context.state["max_drawdown"] * 100,
            "total_trades": len(trades),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "current_positions": self.context.state["positions"],
            "remaining_cash": self.context.state["cash"],
            "trades": trades,
        }

        # 有权益曲线时补充夏普等统计指标；字段名保持与旧报告兼容。
        equity_curve = self.context.state["equity_curve"]
        if len(equity_curve) > 1:
            equity_df = pd.DataFrame(equity_curve)

            if "equity" in equity_df.columns:
                equity_df["return"] = equity_df["equity"].pct_change()

                if not equity_df["return"].empty:
                    equity_df["cumulative_return"] = (1 + equity_df["return"]).cumprod() - 1

                    if equity_df["return"].std() > 0:
                        report["sharpe_ratio"] = (equity_df["return"].mean() / equity_df["return"].std()) * (252 ** 0.5)

        # mode 仍负责补充 backtest/paper/live 各自特有的运行参数。
        self.context._add_mode_specific_metrics(report)
        self._attach_diagnostics(report)

        return report

    def save(self, report: Dict[str, Any]) -> None:
        """按配置保存报告，并输出摘要日志。"""
        report_dir = self.context.config.get(
            "reporting",
            f"{self.context.mode_name}_reports_dir",
            default=f"reports/{self.context.mode_name}/",
        )
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        output_formats = self.context.config.get("reporting", "output_formats", default=["json"])
        written_paths = ReportWriter(report_dir, mode_name=self.context.mode_name).write(report, output_formats)
        report.setdefault("artifacts", {})["report_paths"] = [str(path) for path in written_paths]
        for path in written_paths:
            if path.name.endswith(".json"):
                self.context.logger.info(f"Performance report saved to {path}")
            elif "_trades_" in path.name:
                self.context.logger.info(f"Trade records saved to {path}")
            elif "_equity_curve_" in path.name:
                self.context.logger.info(f"Equity curve saved to {path}")

        self._save_strategy_chart(report)
        self.log_summary(report)

    def prepare_for_serialization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """清理 JSON 不易序列化的字段，保持旧报告写入行为。"""
        return ReportWriter(".", mode_name=self.context.mode_name).prepare_for_serialization(report)

    def log_summary(self, report: Dict[str, Any]) -> None:
        """打印和旧 BaseTradingMode 一致的报告摘要。"""
        logger = self.context.logger
        mode_name = self.context.mode_name

        logger.info(f"==== {mode_name.capitalize()} Performance Summary ====")
        logger.info(f"Initial capital: ${report.get('initial_capital', 0):,.2f}")
        logger.info(f"Final equity: ${report.get('final_equity', 0):,.2f}")
        logger.info(f"Total return: ${report.get('total_return', 0):,.2f} ({report.get('total_return_pct', 0):.2f}%)")
        logger.info(f"Max drawdown: {report.get('max_drawdown_pct', 0):.2f}%")

        if "sharpe_ratio" in report:
            logger.info(f"Sharpe ratio: {report.get('sharpe_ratio', 0):.2f}")

        logger.info(f"Total trades: {report.get('total_trades', 0)}")
        logger.info("==========================================")

    def _trade_action_value(self, trade: Dict[str, Any]) -> str:
        """兼容 action 是字符串或 Direction 枚举两种情况。"""
        action = trade.get("action", "")
        if hasattr(action, "value"):
            return str(action.value).lower()
        return str(action).lower()

    def _save_strategy_chart(self, report: Dict[str, Any]) -> None:
        """Write a visual strategy chart for backtests when market data is available."""
        if self.context.mode_name != "backtest":
            return

        historical_data = getattr(self.context, "historical_data", None)
        if not historical_data:
            return

        try:
            active_strategy = self.context.config.get("strategy", "active", default="dual_ma")
            chart_dir = self.context.config.get("reporting", "strategy_charts_dir", default="reports/final")
            chart_path = StrategyChartWriter(chart_dir).write(
                historical_data=historical_data,
                report=report,
                short_window=self._strategy_parameter(active_strategy, "short_window", 20),
                long_window=self._strategy_parameter(active_strategy, "long_window", 50),
                start_date=getattr(self.context, "start_date", None),
                end_date=getattr(self.context, "end_date", None),
            )
            report.setdefault("artifacts", {})["strategy_chart_path"] = str(chart_path)
            self.context.logger.info(f"Strategy chart saved to {chart_path}")
        except Exception as exc:
            self.context.logger.warning(f"Strategy chart generation skipped: {exc}")

    def _strategy_parameter(self, active_strategy: str, name: str, default):
        return self.context.config.get(
            "strategy",
            "parameters",
            name,
            default=self.context.config.get("strategy", active_strategy, name, default=default),
        )

    def _attach_diagnostics(self, report: Dict[str, Any]) -> None:
        report["diagnostics"] = diagnose_backtest_report(report)
