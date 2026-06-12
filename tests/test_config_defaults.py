import unittest
from pathlib import Path

import yaml


class ConfigDefaultsTest(unittest.TestCase):
    def test_main_config_defaults_strategy_interface_to_domain(self):
        repo_root = Path(__file__).resolve().parents[1]
        data = yaml.safe_load((repo_root / "conf" / "config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(data["strategy"]["interface"], "domain")
        self.assertEqual(data["strategy"]["active"], "dual_ma")


if __name__ == "__main__":
    unittest.main()
