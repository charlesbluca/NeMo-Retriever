# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from nemo_retriever.harness import history
import nemo_retriever.harness.portal.app as portal_app


def _record_run(
    db_path: str,
    artifact_dir,
    *,
    timestamp: str,
    success: bool,
    dataset: str,
    pages_per_sec: float | None = 12.5,
) -> int:
    return history.record_run(
        {
            "timestamp": timestamp,
            "latest_commit": "abcdef123456",
            "success": success,
            "return_code": 0 if success else 1,
            "test_config": {"dataset_label": dataset, "preset": "baseline"},
            "summary_metrics": {
                "pages": 10,
                "files": 2,
                "ingest_secs": 0.8,
                "pages_per_sec_ingest": pages_per_sec,
                "recall_1": 0.7,
                "recall_5": 0.9,
                "recall_10": 0.95,
            },
            "run_metadata": {"host": "runner-1", "gpu_type": "H100"},
            "tags": ["nightly"],
        },
        artifact_dir,
        db_path=db_path,
        num_gpus=2,
    )


def test_get_successful_runs_since_filters_inclusively_and_orders_newest(tmp_path) -> None:
    db_path = str(tmp_path / "history.db")
    cutoff = "20260628_120000_UTC"

    newest_id = _record_run(
        db_path,
        tmp_path,
        timestamp="20260629_115959_UTC",
        success=True,
        dataset="newest",
        pages_per_sec=None,
    )
    boundary_id = _record_run(
        db_path,
        tmp_path,
        timestamp=cutoff,
        success=True,
        dataset="boundary",
    )
    _record_run(
        db_path,
        tmp_path,
        timestamp="20260628_115959_UTC",
        success=True,
        dataset="expired",
    )
    _record_run(
        db_path,
        tmp_path,
        timestamp="20260629_110000_UTC",
        success=False,
        dataset="failed",
    )

    runs = history.get_successful_runs_since(cutoff, db_path=db_path)

    assert [run["id"] for run in runs] == [newest_id, boundary_id]
    assert [run["dataset"] for run in runs] == ["newest", "boundary"]
    assert runs[0]["pages_per_sec"] is None
    assert runs[0]["tags"] == ["nightly"]
    assert runs[0]["num_gpus"] == 2


def test_get_successful_runs_since_is_not_limited_to_general_feed_size(tmp_path) -> None:
    db_path = str(tmp_path / "history.db")
    conn = history._connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO runs (timestamp, dataset, success) VALUES (?, ?, 1)",
            [("20260629_120000_UTC", f"dataset-{index}") for index in range(205)],
        )
        conn.commit()
    finally:
        conn.close()

    runs = history.get_successful_runs_since("20260628_120000_UTC", db_path=db_path)

    assert len(runs) == 205


def test_recent_success_api_reports_fixed_utc_window_and_precedes_dynamic_route(monkeypatch) -> None:
    captured: list[str] = []
    expected_runs = [{"id": 7, "success": 1}]

    def fake_query(since_timestamp: str):
        captured.append(since_timestamp)
        return expected_runs

    monkeypatch.setattr(portal_app.history, "get_successful_runs_since", fake_query)

    response = TestClient(portal_app.app).get("/api/runs/recent-successes")
    assert response.status_code == 200
    payload = response.json()

    window_start = datetime.fromisoformat(payload["window_start"])
    window_end = datetime.fromisoformat(payload["window_end"])
    assert window_start.utcoffset() == timedelta(0)
    assert window_end - window_start == timedelta(hours=24)
    assert payload["window_hours"] == 24
    assert payload["runs"] == expected_runs
    assert captured == [window_start.strftime("%Y%m%d_%H%M%S_UTC")]

    route_paths = [route.path for route in portal_app.app.routes]
    assert route_paths.index("/api/runs/recent-successes") < route_paths.index("/api/runs/{run_id}")


def test_recent_successes_view_is_loaded_and_wired() -> None:
    static_dir = portal_app.STATIC_DIR
    index_source = (static_dir / "index.html").read_text(encoding="utf-8")
    layout_source = (static_dir / "views" / "layout.jsx").read_text(encoding="utf-8")
    app_source = (static_dir / "views" / "app.jsx").read_text(encoding="utf-8")
    view_source = (static_dir / "views" / "recent-successes.jsx").read_text(encoding="utf-8")

    assert '"/static/views/recent-successes.jsx"' in index_source
    assert "@babel/standalone@7.26.10/babel.min.js" in index_source
    assert "react-dom@18.3.1/umd/react-dom.production.min.js" in index_source
    assert "unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js" in index_source
    assert '[["react", { runtime: "classic" }]]' in index_source
    assert "showPortalStartupError" in index_source
    assert "Recent Successes" in layout_source
    assert "onNavigate('recent-successes')" in layout_source
    assert 'fetch("/api/runs/recent-successes")' in app_source
    assert 'activeView === "recent-successes"' in app_source
    assert 'activeView==="recent-successes"' in app_source
    assert "<RecentSuccessesView" in app_source

    expected_metrics = {
        "pages",
        "files",
        "pages_per_sec",
        "ingest_secs",
        "recall_1",
        "recall_5",
        "recall_10",
        "gpu_type",
        "num_gpus",
    }
    for metric in expected_metrics:
        assert f'key: "{metric}"' in view_source

    assert "compareRecentRuns" in view_source
    assert "usePagination(sortedRuns, 25)" in view_source
    assert "onSelectRun(run.id)" in view_source
    assert "No successful runs in the last 24 hours" in view_source
