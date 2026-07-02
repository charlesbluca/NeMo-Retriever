# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import typer
import yaml

from nemo_retriever.harness.artifacts import get_artifacts_root, last_commit, now_timestr
from nemo_retriever.harness.contracts import EXIT_SUCCESS
from nemo_retriever.harness.execution import run_benchmark
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.harness.runfile import load_runfile
from nemo_retriever.harness.slack import load_replay_report, load_session_report, post_report_to_slack


def _load_nightly_config(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read nightly config {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse nightly config {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Nightly config must contain a mapping: {source}")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Nightly config 'runs' must be a non-empty list")
    for item in runs:
        if not isinstance(item, Mapping) or not isinstance(item.get("runfile"), str):
            raise ValueError("Each nightly run must contain a runfile path")
    slack = payload.get("slack", {})
    if not isinstance(slack, Mapping):
        raise ValueError("Nightly config 'slack' must be a mapping")
    return dict(payload)


def _resolve_runfile(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _post_summary(summary_path: Path, slack_config: dict[str, Any], *, skip_slack: bool) -> bool:
    if skip_slack or not bool(slack_config.get("enabled", True)):
        return False
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("Slack reporting is enabled but SLACK_WEBHOOK_URL is not set")
    post_report_to_slack(load_session_report(summary_path), slack_config, webhook_url=webhook)
    return True


def nightly_command(
    runs_config: Path = typer.Option(..., "--runs-config", help="Artifact-first nightly runs YAML."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Directory for nightly session artifacts."),
    skip_slack: bool = typer.Option(False, "--skip-slack", help="Intentionally skip Slack delivery."),
    replay: list[Path] = typer.Option([], "--replay", help="Replay a prior session or run result to Slack."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve runs without Helm or benchmark execution."),
) -> None:
    """Run an artifact-first nightly session and publish its Slack summary."""

    config_path = runs_config.expanduser().resolve()
    try:
        config = _load_nightly_config(config_path)
        slack_config = dict(config.get("slack") or {})
        if replay:
            if skip_slack:
                typer.echo("Slack replay skipped (--skip-slack).")
                raise typer.Exit(code=0)
            webhook = os.environ.get("SLACK_WEBHOOK_URL")
            if not webhook:
                raise RuntimeError("Slack replay requires SLACK_WEBHOOK_URL")
            report = load_replay_report(replay)
            post_report_to_slack(report, slack_config, webhook_url=webhook)
            typer.echo(f"Posted Slack summary for session {report.session_name}.")
            raise typer.Exit(code=0)

        session_name = str(config.get("session_name") or "nightly")
        session_dir = (
            output_dir.expanduser().resolve()
            if output_dir is not None
            else (get_artifacts_root() / f"{session_name}_{now_timestr()}").resolve()
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        run_summaries: list[dict[str, Any]] = []
        session_exit = EXIT_SUCCESS
        for index, item in enumerate(config["runs"], start=1):
            runfile_path = _resolve_runfile(config_path, str(item["runfile"]))
            request = load_runfile(runfile_path)
            run_name = str(item.get("name") or request.name or request.benchmark)
            artifact_dir = session_dir / f"{index:03d}_{run_name}"
            outcome = run_benchmark(
                request.benchmark,
                output_dir=str(artifact_dir),
                run_id=request.run_id or f"{session_name}_{index:03d}_{run_name}",
                mode=request.mode or "local",
                overrides=request.overrides,
                requirements=request.requirements,
                dry_run=dry_run or bool(request.dry_run),
                runfile_payload=dict(request.payload),
                runfile_path=str(request.source_path),
                target=request.target or "library",
                helm_config=request.helm_config,
            )
            run_summaries.append(
                {
                    "run_name": run_name,
                    "benchmark": request.benchmark,
                    "target": request.target or "library",
                    "artifact_dir": str(outcome.artifact_dir),
                    "success": outcome.exit_code == EXIT_SUCCESS,
                    "exit_code": outcome.exit_code,
                    "summary_metrics": outcome.results.get("summary_metrics", {}),
                    "results_path": str(outcome.artifact_dir / "results.json"),
                }
            )
            if session_exit == EXIT_SUCCESS and outcome.exit_code != EXIT_SUCCESS:
                session_exit = outcome.exit_code

        summary = {
            "session_name": session_name,
            "session_type": "nightly",
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "success": session_exit == EXIT_SUCCESS,
            "exit_code": session_exit,
            "dry_run": bool(dry_run),
            "runs": run_summaries,
        }
        summary_path = session_dir / "session_summary.json"
        write_json(summary_path, summary)
        typer.echo(f"Nightly session: {session_dir}")
        typer.echo(f"Session summary: {summary_path}")

        slack_failed = False
        if not dry_run:
            try:
                if _post_summary(summary_path, slack_config, skip_slack=skip_slack):
                    typer.echo(f"Posted Slack summary for session {session_name}.")
            except RuntimeError as exc:
                typer.echo(f"Slack post failed: {exc}", err=True)
                slack_failed = True
        else:
            typer.echo("Slack posting skipped for --dry-run.")

        raise typer.Exit(code=session_exit if session_exit != EXIT_SUCCESS else (1 if slack_failed else 0))
    except typer.Exit:
        raise
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
