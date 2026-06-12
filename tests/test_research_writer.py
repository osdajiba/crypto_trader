import json
import tempfile
import unittest
from pathlib import Path


class ResearchSummaryWriterTests(unittest.TestCase):
    def test_writer_saves_summary_json_and_returns_path(self):
        from src.reporting.research_writer import ResearchSummaryWriter

        summary = {
            "metric": "diagnostics.cost_to_abs_net_return",
            "run_count": 1,
            "success_count": 1,
            "failed_count": 0,
            "best_run_id": "run-001",
            "best_metric": 0.25,
            "results": [
                {
                    "run_id": "run-001",
                    "status": "success",
                    "parameters": {"short_window": 5},
                    "metric": 0.25,
                    "diagnostics": {"cost_to_abs_net_return": 0.25},
                    "error": None,
                    "rank": 1,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = ResearchSummaryWriter(Path(tmp)).write(summary)
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(path.parent, Path(tmp))
            self.assertTrue(path.name.startswith("research_summary_"))
            self.assertEqual(path.suffix, ".json")

        expected = dict(summary)
        expected["summary_path"] = str(path)
        self.assertEqual(saved, expected)

    def test_writer_updates_research_index(self):
        from src.reporting.research_writer import ResearchSummaryWriter

        summary = {
            "metric": "final_equity",
            "run_count": 2,
            "success_count": 1,
            "failed_count": 1,
            "best_run_id": "run-001",
            "best_metric": 100010,
            "results": [],
        }

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            writer = ResearchSummaryWriter(Path(tmp))
            first_path = writer.write(summary)
            second_path = writer.write(dict(summary, best_run_id="run-002"))

            index_path = Path(tmp) / "research_index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))

        self.assertEqual(index["summary_count"], 2)
        self.assertEqual(index["summaries"][0]["summary_path"], str(first_path))
        self.assertEqual(index["summaries"][0]["best_run_id"], "run-001")
        self.assertEqual(index["summaries"][1]["summary_path"], str(second_path))
        self.assertEqual(index["summaries"][1]["best_run_id"], "run-002")


if __name__ == "__main__":
    unittest.main()
