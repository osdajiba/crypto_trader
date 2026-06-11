import importlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch


class ParquetFileManagerTest(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("src.common.helpers", None)
        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            self.helpers_module = importlib.import_module("src.common.helpers")

    def test_finds_binance_nested_historical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)
            target_dir = base_path / "binance" / "BTC_USDT" / "1m" / "2025" / "01"
            target_dir.mkdir(parents=True)
            expected = target_dir / "2025-01-01T00_00_00+00_00to2025-01-02T00_00_00+00_00.parquet"
            expected.touch()

            files = self.helpers_module.ParquetFileManager.find_files_in_date_range(
                str(base_path),
                "1m",
                "BTC/USDT",
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(files, [str(expected)])

    def test_finds_flat_binance_symbol_timeframe_file_in_sibling_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            historical_path = data_dir / "historical"
            target_dir = data_dir / "binance" / "BTC_USDT" / "1m"
            target_dir.mkdir(parents=True)
            historical_path.mkdir(parents=True)
            expected = target_dir / "BTC_USDT_1m.parquet"
            expected.touch()

            files = self.helpers_module.ParquetFileManager.find_files_in_date_range(
                str(historical_path),
                "1m",
                "BTC/USDT",
                datetime(2025, 4, 21, tzinfo=timezone.utc),
                datetime(2025, 4, 28, tzinfo=timezone.utc),
            )

        self.assertEqual(files, [str(expected)])


if __name__ == "__main__":
    unittest.main()
