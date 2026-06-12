"""Report file writers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class ReportWriter:
    """Write report dictionaries to configured output formats."""

    def __init__(self, report_dir, mode_name: str) -> None:
        self.report_dir = Path(report_dir)
        self.mode_name = mode_name

    def write(self, report: dict, output_formats: list[str] | None = None) -> list[Path]:
        output_formats = output_formats or ["json"]
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        written: list[Path] = []

        if "csv" in output_formats and report.get("trades"):
            trades_csv = self.report_dir / f"{self.mode_name}_trades_{timestamp}.csv"
            pd.DataFrame(report["trades"]).to_csv(trades_csv, index=False)
            written.append(trades_csv)

        if "csv" in output_formats and report.get("equity_curve"):
            equity_csv = self.report_dir / f"{self.mode_name}_equity_curve_{timestamp}.csv"
            pd.DataFrame(report["equity_curve"]).to_csv(equity_csv, index=False)
            written.append(equity_csv)

        if "json" in output_formats:
            report_json = self.report_dir / f"{self.mode_name}_report_{timestamp}.json"
            report_json.write_text(
                json.dumps(self.prepare_for_serialization(report), indent=4, default=str),
                encoding="utf-8",
            )
            written.append(report_json)

        return written

    def prepare_for_serialization(self, report: dict) -> dict:
        clean_report = report.copy()

        for key in ["equity_curve", "strategy"]:
            if key in clean_report:
                if key == "equity_curve":
                    clean_report[key] = f"{len(clean_report[key])} records"
                elif key == "strategy":
                    clean_report[key] = clean_report[key].__class__.__name__

        return clean_report
