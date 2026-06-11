import json
import tempfile
import unittest
from pathlib import Path


class ReportWriterTests(unittest.TestCase):
    def test_json_writer_cleans_report_and_returns_written_path(self):
        from src.reporting.writers import ReportWriter

        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "final_equity": 100000,
                "total_trades": 1,
                "equity_curve": [{"timestamp": "2025-01-01", "equity": 100000}],
                "strategy": object(),
            }

            written = ReportWriter(Path(tmp), mode_name="backtest").write(report, output_formats=["json"])

            self.assertEqual(len(written), 1)
            saved = json.loads(written[0].read_text())

        self.assertEqual(saved["final_equity"], 100000)
        self.assertEqual(saved["total_trades"], 1)
        self.assertEqual(saved["equity_curve"], "1 records")
        self.assertEqual(saved["strategy"], "object")
