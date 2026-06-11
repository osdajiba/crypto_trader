import unittest
from datetime import datetime, timedelta, timezone


class BacktestPlanningTests(unittest.TestCase):
    def test_parameter_grid_expands_in_stable_order(self):
        from src.application.backtest_planning import expand_parameter_grid

        combinations = expand_parameter_grid({
            "short_window": [5, 10],
            "long_window": [20, 30],
            "risk_profile": ["base"],
        })

        self.assertEqual(
            combinations,
            [
                {"short_window": 5, "long_window": 20, "risk_profile": "base"},
                {"short_window": 5, "long_window": 30, "risk_profile": "base"},
                {"short_window": 10, "long_window": 20, "risk_profile": "base"},
                {"short_window": 10, "long_window": 30, "risk_profile": "base"},
            ],
        )

    def test_parameter_grid_rejects_empty_parameter_values(self):
        from src.application.backtest_planning import expand_parameter_grid

        with self.assertRaisesRegex(ValueError, "short_window"):
            expand_parameter_grid({"short_window": [], "long_window": [20]})

    def test_walk_forward_windows_are_generated_from_train_test_and_step(self):
        from src.application.backtest_planning import WalkForwardWindow, generate_walk_forward_windows

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 10, tzinfo=timezone.utc)

        windows = generate_walk_forward_windows(
            start=start,
            end=end,
            train_size=timedelta(days=3),
            test_size=timedelta(days=2),
        )

        self.assertEqual(
            windows,
            [
                WalkForwardWindow(start, start + timedelta(days=3), start + timedelta(days=3), start + timedelta(days=5)),
                WalkForwardWindow(start + timedelta(days=2), start + timedelta(days=5), start + timedelta(days=5), start + timedelta(days=7)),
                WalkForwardWindow(start + timedelta(days=4), start + timedelta(days=7), start + timedelta(days=7), start + timedelta(days=9)),
            ],
        )

    def test_walk_forward_windows_reject_non_positive_durations(self):
        from src.application.backtest_planning import generate_walk_forward_windows

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 10, tzinfo=timezone.utc)

        with self.assertRaisesRegex(ValueError, "train_size"):
            generate_walk_forward_windows(
                start=start,
                end=end,
                train_size=timedelta(0),
                test_size=timedelta(days=1),
            )


if __name__ == "__main__":
    unittest.main()
