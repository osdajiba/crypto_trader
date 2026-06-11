"""Writers for local backtest research summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResearchSummaryWriter:
    """Persist scan summaries produced by local research runs."""

    def __init__(self, report_dir: str | Path = "reports/research") -> None:
        self.report_dir = Path(report_dir)

    def write(self, summary: dict[str, Any]) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.report_dir / f"research_summary_{timestamp}.json"
        payload = dict(summary)
        payload.setdefault("summary_path", str(path))
        path.write_text(
            json.dumps(payload, indent=4, default=str),
            encoding="utf-8",
        )
        self._update_index(payload)
        return path

    def _update_index(self, summary: dict[str, Any]) -> None:
        index_path = self.report_dir / "research_index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            summaries = list(index.get("summaries", []))
        else:
            summaries = []

        summaries.append({
            "summary_path": summary.get("summary_path"),
            "metric": summary.get("metric"),
            "run_count": summary.get("run_count"),
            "success_count": summary.get("success_count"),
            "failed_count": summary.get("failed_count"),
            "best_run_id": summary.get("best_run_id"),
            "best_metric": summary.get("best_metric"),
            "written_at": datetime.now().isoformat(),
        })
        index_path.write_text(
            json.dumps(
                {
                    "summary_count": len(summaries),
                    "summaries": summaries,
                },
                indent=4,
                default=str,
            ),
            encoding="utf-8",
        )
