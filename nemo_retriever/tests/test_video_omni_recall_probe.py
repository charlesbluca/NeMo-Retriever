# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "video_omni_recall_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("video_omni_recall_probe", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_segment():
    return {
        "segment_id": "scene-alpha",
        "start_seconds": 0.0,
        "end_seconds": 2.5,
        "summary": "Alpha scene with a warning marker.",
        "visual_text": ["ALPHA-17"],
        "objects": ["red triangle", "blue crate"],
        "actions": ["static display", "warning shown"],
        "audio_or_speech": "A quiet audio check tone is present.",
        "retrieval_keywords": ["alpha", "warning", "crate"],
        "uncertainties": [],
        "confidence": 0.93,
    }


def test_parse_args_defaults_with_tmp_artifact_dir_and_dry_run(tmp_path):
    probe = load_probe_module()

    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--dry-run"])

    assert config.artifact_dir == tmp_path
    assert config.dry_run is True
    assert config.use_cache is True
    assert "ALPHA-17" in config.queries[0]


def test_parse_args_no_cache_repeated_queries_and_top_k_clamp(tmp_path):
    probe = load_probe_module()

    config = probe.parse_args(
        [
            "--artifact-dir",
            str(tmp_path),
            "--no-cache",
            "--top-k",
            "0",
            "--query",
            "first query",
            "--query",
            "second query",
        ]
    )

    assert config.artifact_dir == tmp_path
    assert config.use_cache is False
    assert config.top_k == 1
    assert config.queries == ("first query", "second query")


def test_parse_args_dataset_eval_uses_text_embedder_and_batch_controls(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    dataset_dir.mkdir()

    config = probe.parse_args(
        [
            "--dataset-eval",
            "--dataset-dir",
            str(dataset_dir),
            "--chunk-seconds",
            "45",
            "--chunk-overlap-seconds",
            "7.5",
            "--max-videos",
            "2",
            "--video-bin",
            "Very Short",
            "--query-limit",
            "3",
            "--text-embed-model",
            "text-model",
            "--embed-model",
            "vl-model",
        ]
    )

    assert config.dataset_eval is True
    assert config.video_path is None
    assert config.text_embed_model == "text-model"
    assert config.embed_model == "vl-model"
    assert config.chunk_seconds == 45.0
    assert config.chunk_overlap_seconds == 7.5
    assert config.max_videos == 2
    assert config.video_bin == "Very Short"
    assert config.query_limit == 3


def test_parse_args_dataset_dir_resolves_first_video_and_queries(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"video")
    (dataset_dir / "query.csv").write_text(
        "name,question,answer,answer_modality,start_time,end_time\n"
        "clip_one,Question one?,Answer one,Audio + Visual,1.5,3.25\n"
        "clip_two,Question two?,Answer two,Visual,4.0,5.0\n",
        encoding="utf-8",
    )

    config = probe.parse_args(["--dataset-dir", str(dataset_dir), "--query-limit", "1"])

    assert config.video_path == corpus_dir / "clip_one.mp4"
    assert config.video_name == "clip_one"
    assert config.queries == ("Question one?",)
    assert config.query_expectations == (
        {
            "source_video_name": "clip_one",
            "expected_answer": "Answer one",
            "answer_modality": "Audio + Visual",
            "expected_start_seconds": 1.5,
            "expected_end_seconds": 3.25,
        },
    )


def test_parse_args_video_name_filters_queries_before_limit(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"one")
    (corpus_dir / "clip_two.mp4").write_bytes(b"two")
    (dataset_dir / "query.csv").write_text(
        "name,question,answer,start_time,end_time\n"
        "clip_one,First clip question?,First answer,1,2\n"
        "clip_two,Second clip first question?,Second answer one,3,4\n"
        "clip_two,Second clip second question?,Second answer two,5,6\n",
        encoding="utf-8",
    )

    config = probe.parse_args(
        ["--dataset-dir", str(dataset_dir), "--video-name", "clip_two", "--query-limit", "1"]
    )

    assert config.video_path == corpus_dir / "clip_two.mp4"
    assert config.queries == ("Second clip first question?",)
    assert config.query_expectations[0]["source_video_name"] == "clip_two"
    assert config.query_expectations[0]["expected_answer"] == "Second answer one"


def test_parse_args_missing_requested_dataset_video_raises(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"one")
    (dataset_dir / "query.csv").write_text(
        "name,question,answer,start_time,end_time\nclip_two,Missing clip question?,Answer,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="clip_two"):
        probe.parse_args(["--dataset-dir", str(dataset_dir), "--video-name", "clip_two"])


def test_parse_args_query_csv_missing_question_column_raises(tmp_path):
    probe = load_probe_module()
    queries_path = tmp_path / "query.csv"
    queries_path.write_text("name,answer\nclip,Answer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        probe.parse_args(["--queries-file", str(queries_path)])


def test_parse_args_filtered_query_csv_without_questions_raises(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_two.mp4").write_bytes(b"two")
    (dataset_dir / "query.csv").write_text(
        "name,question,answer,start_time,end_time\nclip_one,First clip question?,Answer,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No usable queries"):
        probe.parse_args(["--dataset-dir", str(dataset_dir), "--video-name", "clip_two"])


def test_parse_args_repeated_queries_override_queries_file(tmp_path):
    probe = load_probe_module()
    queries_path = tmp_path / "query.csv"
    queries_path.write_text(
        "name,question,answer,start_time,end_time\nclip,CSV question?,CSV answer,1,2\n",
        encoding="utf-8",
    )

    config = probe.parse_args(["--queries-file", str(queries_path), "--query", "manual question"])

    assert config.queries == ("manual question",)
    assert config.query_expectations == ({},)


def test_parse_args_repeated_queries_skip_malformed_queries_file(tmp_path):
    probe = load_probe_module()
    queries_path = tmp_path / "query.csv"
    queries_path.write_text("name,answer\nclip,CSV answer\n", encoding="utf-8")

    config = probe.parse_args(["--queries-file", str(queries_path), "--query", "manual question"])

    assert config.queries == ("manual question",)
    assert config.query_expectations == ({},)


def test_parse_time_seconds_accepts_colon_and_float_values():
    probe = load_probe_module()

    assert probe.parse_time_seconds("0:02:33") == pytest.approx(153.0)
    assert probe.parse_time_seconds("01:02:03.5") == pytest.approx(3723.5)
    assert probe.parse_time_seconds("12.25") == pytest.approx(12.25)
    assert probe.parse_time_seconds("") is None


def test_plan_video_chunks_clamps_final_window_and_uses_overlap():
    probe = load_probe_module()

    chunks = probe.plan_video_chunks(65.0, chunk_seconds=30.0, overlap_seconds=5.0)

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-0000", "chunk-0001", "chunk-0002"]
    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [
        (0.0, 30.0),
        (25.0, 55.0),
        (50.0, 65.0),
    ]


def test_plan_video_chunks_rejects_non_advancing_overlap():
    probe = load_probe_module()

    with pytest.raises(ValueError, match="overlap"):
        probe.plan_video_chunks(60.0, chunk_seconds=30.0, overlap_seconds=30.0)


def test_resolve_dataset_video_path_matches_stem(tmp_path):
    probe = load_probe_module()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    video_path = corpus_dir / "Example_Clip.mp4"
    video_path.write_bytes(b"video")

    assert probe.resolve_dataset_video_path(tmp_path, "Example_Clip") == video_path
    assert probe.resolve_dataset_video_path(tmp_path, "Example_Clip.mp4") == video_path


def test_load_eval_video_jobs_filters_video_bin_and_limits_per_video_queries(tmp_path):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"one")
    (corpus_dir / "clip_two.mp4").write_bytes(b"two")
    (dataset_dir / "query.csv").write_text(
        "name,video_duration,video_bin,question,answer,answer_modality,start_time,end_time\n"
        "clip_one,0:01:05,Very Short,One q1?,One a1,Audio + Visual,1,2\n"
        "clip_one,0:01:05,Very Short,One q2?,One a2,Visual,3,4\n"
        "clip_two,0:02:00,Short,Two q1?,Two a1,Audio only,5,6\n",
        encoding="utf-8",
    )

    jobs = probe.load_eval_video_jobs(dataset_dir, video_bin="Very Short", max_videos=2, query_limit=1)

    assert len(jobs) == 1
    assert jobs[0].video_name == "clip_one"
    assert jobs[0].video_path == corpus_dir / "clip_one.mp4"
    assert jobs[0].duration_seconds == pytest.approx(65.0)
    assert jobs[0].queries == ("One q1?",)
    assert jobs[0].query_expectations == (
        {
            "source_video_name": "clip_one",
            "expected_answer": "One a1",
            "answer_modality": "Audio + Visual",
            "expected_start_seconds": 1.0,
            "expected_end_seconds": 2.0,
        },
    )


def test_safe_excerpt_keeps_short_text():
    probe = load_probe_module()

    assert probe.safe_excerpt("short text", limit=60) == "short text"


def test_safe_excerpt_truncates_long_text_and_preserves_tail():
    probe = load_probe_module()

    excerpt = probe.safe_excerpt("a" * 80 + "b" * 80, limit=60)

    assert "\n...[truncated]...\n" in excerpt
    assert excerpt.endswith("b" * 10)


def test_missing_api_key_raises_with_env_name():
    probe = load_probe_module()

    with pytest.raises(RuntimeError, match="NGC_NV_DEVELOPER_NVCF"):
        probe.require_api_key({})


def test_fenced_json_parses_to_object():
    probe = load_probe_module()

    assert probe.extract_json_object('```json\n{"segments": []}\n```') == {"segments": []}


def test_surrounded_json_parses_segment_id():
    probe = load_probe_module()

    result = probe.extract_json_object('model output: {"segment_id": "scene-alpha"} thanks')

    assert result["segment_id"] == "scene-alpha"


def test_json_array_segments_parses_to_segments_object():
    probe = load_probe_module()

    result = probe.extract_json_object('[{"segment_id": "scene-alpha"}]')

    assert result == {"segments": [{"segment_id": "scene-alpha"}]}


def test_truncated_segments_object_recovers_segments_array():
    probe = load_probe_module()

    result = probe.extract_json_object('{"segments": [{"segment_id": "scene-alpha"}]')

    assert result == {"segments": [{"segment_id": "scene-alpha"}]}


def test_cosine_similarity_handles_identical_zero_and_shape_mismatch():
    probe = load_probe_module()

    assert probe.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert probe.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert probe.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_run_command_quotes_arguments_with_spaces(monkeypatch):
    probe = load_probe_module()

    def fake_run(args, stdout, stderr, text):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"probe 'two words'"):
        probe.run_command(["probe", "two words"])


def test_scene_specs_are_deterministic():
    probe = load_probe_module()

    scenes = probe.scene_specs()

    assert [scene.segment_id for scene in scenes] == [
        "scene-alpha",
        "scene-beta",
        "scene-calibration",
        "scene-audio",
    ]
    assert scenes[0].expected_terms[0] == "alpha-17"
    assert probe.total_duration_seconds(scenes) == 10.0


def test_uniform_scene_duration_seconds_returns_default_scene_duration():
    probe = load_probe_module()

    assert probe.uniform_scene_duration_seconds(probe.scene_specs()) == 2.5


def test_uniform_scene_duration_seconds_raises_for_non_uniform_durations():
    probe = load_probe_module()
    scenes = probe.scene_specs()
    non_uniform_scenes = (scenes[0], replace(scenes[1], end_seconds=6.0))

    with pytest.raises(ValueError, match="uniform scene durations"):
        probe.uniform_scene_duration_seconds(non_uniform_scenes)


def test_uniform_scene_duration_seconds_raises_for_empty_scenes():
    probe = load_probe_module()

    with pytest.raises(ValueError, match="At least one scene"):
        probe.uniform_scene_duration_seconds(())


def test_draw_scene_frame_writes_png(tmp_path):
    probe = load_probe_module()
    output_path = tmp_path / "scene.png"

    probe.draw_scene_frame(probe.scene_specs()[0], output_path, size=(320, 180))

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(probe.image_file_to_b64(output_path)) > 100


def test_write_silent_wav_writes_expected_header(tmp_path):
    probe = load_probe_module()
    output_path = tmp_path / "silent.wav"

    probe.write_silent_wav(output_path, 0.25, sample_rate=8000)

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"RIFF")


def test_artifact_path_returns_named_file_under_artifact_dir(tmp_path):
    probe = load_probe_module()
    config = SimpleNamespace(artifact_dir=tmp_path)

    assert probe.artifact_path(config, "x") == tmp_path / "x"


def test_build_video_from_frames_writes_concat_list_and_runs_ffmpeg(tmp_path, monkeypatch):
    probe = load_probe_module()
    frame_paths = [tmp_path / "frame-1.png", tmp_path / "frame-2.png"]
    for frame_path in frame_paths:
        frame_path.write_bytes(b"fake frame")
    output_path = tmp_path / "fixture_video_only.mp4"
    commands = []

    monkeypatch.setattr(probe, "run_command", lambda args: commands.append(args))

    probe.build_video_from_frames(frame_paths, output_path, seconds_per_scene=2.5)

    concat_list = output_path.with_suffix(".frames.txt")
    assert concat_list.read_text(encoding="utf-8").splitlines() == [
        f"file '{frame_paths[0].resolve()}'",
        "duration 2.500",
        f"file '{frame_paths[1].resolve()}'",
        "duration 2.500",
        f"file '{frame_paths[1].resolve()}'",
    ]
    assert commands == [
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-vf",
            "fps=24,format=yuv420p",
            str(output_path),
        ]
    ]


def test_prepare_audio_track_writes_silent_fixture_when_source_absent(tmp_path, monkeypatch):
    probe = load_probe_module()
    artifact_dir = tmp_path / "artifacts"
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe, "write_silent_wav", lambda path, duration_seconds: calls.append((path, duration_seconds)))

    audio_path = probe.prepare_audio_track(artifact_dir, duration_seconds=10.0)

    assert audio_path == artifact_dir / "fixture_audio.wav"
    assert calls == [(artifact_dir / "fixture_audio.wav", 10.0)]


def test_generate_synthetic_fixture_reuses_existing_fixture_and_manifest(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    manifest_path = tmp_path / "scene_frames.json"
    metadata_path = tmp_path / "synthetic_fixture_metadata.json"
    fixture_path.write_bytes(b"cached video")
    manifest = {scene.segment_id: f"{scene.segment_id}-frame" for scene in probe.scene_specs()}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    probe.write_json(metadata_path, probe.expected_fixture_metadata())
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])

    def fail_if_called():
        raise AssertionError("ffmpeg should not be required when cached artifacts are reused")

    monkeypatch.setattr(probe, "check_ffmpeg", fail_if_called)
    generated_fixture_path, frame_manifest = probe.generate_synthetic_fixture(config)

    assert generated_fixture_path == fixture_path
    assert frame_manifest == manifest


def test_generate_synthetic_fixture_stale_metadata_regenerates(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    manifest_path = tmp_path / "scene_frames.json"
    metadata_path = tmp_path / "synthetic_fixture_metadata.json"
    fixture_path.write_bytes(b"cached video")
    manifest_path.write_text('{"scene-alpha": "cached-frame"}\n', encoding="utf-8")
    probe.write_json(metadata_path, {"fixture_version": "stale"})
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "draw_scene_frame", lambda scene, path: path.write_bytes(scene.segment_id.encode("utf-8")))
    monkeypatch.setattr(probe, "image_file_to_b64", lambda path: f"regenerated-{path.stem}")
    monkeypatch.setattr(
        probe,
        "build_video_from_frames",
        lambda frame_paths, output_path, seconds_per_scene: output_path.write_bytes(b"new video"),
    )
    monkeypatch.setattr(probe, "prepare_audio_track", lambda artifact_dir, duration_seconds: artifact_dir / "fixture_audio.wav")
    monkeypatch.setattr(probe, "run_command", lambda args: fixture_path.write_bytes(b"regenerated video"))

    generated_fixture_path, frame_manifest = probe.generate_synthetic_fixture(config)

    assert generated_fixture_path == fixture_path
    assert frame_manifest["scene-alpha"] == "regenerated-scene-alpha"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == probe.expected_fixture_metadata()


def test_generate_synthetic_fixture_missing_metadata_regenerates(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    manifest_path = tmp_path / "scene_frames.json"
    fixture_path.write_bytes(b"cached video")
    manifest_path.write_text('{"scene-alpha": "cached-frame"}\n', encoding="utf-8")
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])

    called = {"check_ffmpeg": False}
    monkeypatch.setattr(probe, "check_ffmpeg", lambda: called.update({"check_ffmpeg": True}))
    monkeypatch.setattr(probe, "draw_scene_frame", lambda scene, path: path.write_bytes(scene.segment_id.encode("utf-8")))
    monkeypatch.setattr(probe, "image_file_to_b64", lambda path: f"regenerated-{path.stem}")
    monkeypatch.setattr(
        probe,
        "build_video_from_frames",
        lambda frame_paths, output_path, seconds_per_scene: output_path.write_bytes(b"new video"),
    )
    monkeypatch.setattr(probe, "prepare_audio_track", lambda artifact_dir, duration_seconds: artifact_dir / "fixture_audio.wav")
    monkeypatch.setattr(probe, "run_command", lambda args: fixture_path.write_bytes(b"regenerated video"))

    probe.generate_synthetic_fixture(config)

    assert called["check_ffmpeg"] is True


def test_generate_synthetic_fixture_no_cache_regenerates_existing_fixture_and_manifest(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    manifest_path = tmp_path / "scene_frames.json"
    fixture_path.write_bytes(b"cached video")
    manifest_path.write_text('{"scene-alpha": "cached-frame"}\n', encoding="utf-8")
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--no-cache"])

    replacement_frame_b64 = {"scene-alpha": "regenerated-frame"}
    captured_video_build = {}

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "draw_scene_frame", lambda scene, path: path.write_bytes(scene.segment_id.encode("utf-8")))
    monkeypatch.setattr(probe, "image_file_to_b64", lambda path: replacement_frame_b64[path.stem])
    monkeypatch.setattr(
        probe,
        "build_video_from_frames",
        lambda frame_paths, output_path, seconds_per_scene: (
            captured_video_build.update({"duration": seconds_per_scene}),
            output_path.write_bytes(b"new video"),
        ),
    )
    monkeypatch.setattr(probe, "prepare_audio_track", lambda artifact_dir, duration_seconds: artifact_dir / "fixture_audio.wav")
    monkeypatch.setattr(probe, "run_command", lambda args: fixture_path.write_bytes(b"regenerated video"))
    monkeypatch.setattr(probe, "scene_specs", lambda: (probe.DEFAULT_SCENES[0],))
    monkeypatch.setattr(probe, "total_duration_seconds", lambda scenes=probe.DEFAULT_SCENES: 2.5)

    generated_fixture_path, frame_manifest = probe.generate_synthetic_fixture(config)

    assert generated_fixture_path == fixture_path
    assert frame_manifest == replacement_frame_b64
    assert captured_video_build["duration"] == 2.5
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == replacement_frame_b64


def test_generate_synthetic_fixture_force_regenerates_existing_fixture_and_manifest(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    manifest_path = tmp_path / "scene_frames.json"
    fixture_path.write_bytes(b"cached video")
    manifest_path.write_text('{"scene-alpha": "cached-frame"}\n', encoding="utf-8")
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--force-fixture"])

    captured_video_build = {}

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "draw_scene_frame", lambda scene, path: path.write_bytes(scene.segment_id.encode("utf-8")))
    monkeypatch.setattr(probe, "image_file_to_b64", lambda path: f"regenerated-{path.stem}")
    monkeypatch.setattr(
        probe,
        "build_video_from_frames",
        lambda frame_paths, output_path, seconds_per_scene: (
            captured_video_build.update({"frame_paths": frame_paths, "duration": seconds_per_scene}),
            output_path.write_bytes(b"new video"),
        ),
    )
    monkeypatch.setattr(probe, "prepare_audio_track", lambda artifact_dir, duration_seconds: artifact_dir / "fixture_audio.wav")
    monkeypatch.setattr(probe, "run_command", lambda args: fixture_path.write_bytes(b"regenerated video"))

    generated_fixture_path, frame_manifest = probe.generate_synthetic_fixture(config)

    assert generated_fixture_path == fixture_path
    assert fixture_path.read_bytes() == b"regenerated video"
    assert frame_manifest == {
        "scene-alpha": "regenerated-scene-alpha",
        "scene-beta": "regenerated-scene-beta",
        "scene-calibration": "regenerated-scene-calibration",
        "scene-audio": "regenerated-scene-audio",
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == frame_manifest
    assert len(captured_video_build["frame_paths"]) == 4
    assert captured_video_build["duration"] == 2.5


def test_main_dry_run_returns_before_requiring_api_key(tmp_path, monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (config.artifact_dir / "synthetic_fixture.mp4", {}))

    def fail_if_called():
        raise AssertionError("dry run should not require an API key")

    monkeypatch.setattr(probe, "require_api_key", fail_if_called)

    assert probe.main(["--artifact-dir", str(tmp_path), "--dry-run"]) == 0


def test_main_dry_run_with_video_path_skips_synthetic_fixture(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        probe,
        "generate_synthetic_fixture",
        lambda config: (_ for _ in ()).throw(AssertionError("external video should not generate synthetic fixture")),
    )

    assert probe.main(["--artifact-dir", str(tmp_path / "artifacts"), "--video-path", str(video_path), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert f"Source video: {video_path}" in output
    assert "Synthetic fixture:" not in output


def test_main_dataset_eval_dry_run_loads_jobs_without_hosted_calls(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"one")
    (dataset_dir / "query.csv").write_text(
        "name,video_duration,video_bin,question,answer,start_time,end_time\n"
        "clip_one,0:00:20,Very Short,Question?,Answer,1,2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(probe, "require_api_key", lambda: (_ for _ in ()).throw(AssertionError("dry run should not need API key")))

    assert (
        probe.main(
            [
                "--dataset-eval",
                "--dataset-dir",
                str(dataset_dir),
                "--artifact-dir",
                str(tmp_path / "artifacts"),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Dataset eval jobs: 1 videos, 1 queries" in output


def test_run_dataset_eval_writes_results_summary_and_uses_text_embedder(tmp_path, monkeypatch):
    probe = load_probe_module()
    dataset_dir = tmp_path / "video_retrieval"
    corpus_dir = dataset_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "clip_one.mp4").write_bytes(b"one")
    (dataset_dir / "query.csv").write_text(
        "name,video_duration,video_bin,question,answer,answer_modality,start_time,end_time\n"
        "clip_one,0:00:20,Very Short,Question?,Answer,Visual,1,2\n",
        encoding="utf-8",
    )
    config = probe.parse_args(
        [
            "--dataset-eval",
            "--dataset-dir",
            str(dataset_dir),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--text-embed-model",
            "text-model",
            "--embed-model",
            "vl-model",
        ]
    )
    calls = {}

    def fake_extract_rows(job, config, token):
        calls["job"] = job
        return probe.pd.DataFrame(
            [
                {
                    "segment_id": "clip_one:chunk-0000:segment-1",
                    "text": "Answer",
                    "_image_b64": "",
                    "_embed_modality": "text",
                    "metadata": {
                        "source_video_name": "clip_one",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "raw_omni_segment": {},
                    },
                }
            ]
        )

    def fake_compare_query_results(queries, rows_df, **kwargs):
        calls["compare_kwargs"] = kwargs
        return [
            {
                "query": queries[0],
                "answer_modality": "Visual",
                "expected_answer": "Answer",
                "source_video_name": "clip_one",
                "expected_time_range": {"start_seconds": 1.0, "end_seconds": 2.0},
                "text_only": [
                    {"source_video_name": "clip_one", "start_seconds": 1.0, "end_seconds": 2.0, "text": "Answer"}
                ],
                "vl_text_image": [
                    {"source_video_name": "clip_one", "start_seconds": 3.0, "end_seconds": 4.0, "text": "miss"}
                ],
                "text_only_top_contains_expected": True,
                "vl_top_contains_expected": False,
                "text_only_top_overlaps_expected_time": True,
                "vl_top_overlaps_expected_time": False,
                "text_only_any_overlaps_expected_time": True,
                "vl_any_overlaps_expected_time": False,
            }
        ]

    monkeypatch.setattr(probe, "extract_rows_for_eval_job", fake_extract_rows)
    monkeypatch.setattr(probe, "compare_query_results", fake_compare_query_results)

    result = probe.run_dataset_eval(config, token="token")

    assert result["summary"]["metrics"]["text_only_top1_time_overlap_rate"] == 1.0
    assert result["summary"]["metrics"]["vl_top1_time_overlap_rate"] == 0.0
    assert calls["compare_kwargs"]["text_model_name"] == "text-model"
    assert calls["compare_kwargs"]["model_name"] == "vl-model"
    assert calls["compare_kwargs"]["top_k"] == config.top_k
    assert json.loads((config.artifact_dir / "eval_results.json").read_text(encoding="utf-8"))[0]["query"] == "Question?"
    assert json.loads((config.artifact_dir / "eval_summary.json").read_text(encoding="utf-8"))["metrics"]["query_count"] == 1


def test_extract_rows_for_eval_job_skips_bad_omni_chunk_and_records_error(tmp_path, monkeypatch):
    probe = load_probe_module()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    config = probe.parse_args(
        [
            "--dataset-eval",
            "--dataset-dir",
            str(tmp_path),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--chunk-seconds",
            "10",
            "--chunk-overlap-seconds",
            "0",
        ]
    )
    job = probe.EvalVideoJob(
        video_name="clip",
        video_path=video_path,
        duration_seconds=20.0,
        queries=("Question?",),
        query_expectations=({"source_video_name": "clip"},),
    )
    calls = []

    def fake_extract_video_chunk(source_video, chunk, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"chunk")

    monkeypatch.setattr(probe, "extract_video_chunk", fake_extract_video_chunk)
    monkeypatch.setattr(probe, "extract_representative_frames_from_segments", lambda video_path, segments, artifact_dir: {})

    def fake_call_nano_omni(chunk_path, chunk_config, token):
        calls.append(chunk_path.name)
        if chunk_path.name == "chunk-0000.mp4":
            raise ValueError("No JSON object found.")
        return {
            "elapsed_seconds": 0.25,
            "request": {},
            "raw_response": {},
            "parsed": {"segments": [dict(sample_segment(), segment_id="ok", start_seconds=1.0, end_seconds=2.0)]},
        }

    monkeypatch.setattr(probe, "call_nano_omni", fake_call_nano_omni)

    rows_df = probe.extract_rows_for_eval_job(job, config, token="token")

    rows = rows_df.to_dict(orient="records")
    assert calls == ["chunk-0000.mp4", "chunk-0001.mp4"]
    assert len(rows) == 1
    assert rows[0]["metadata"]["source_video_name"] == "clip"
    assert rows[0]["metadata"]["start_seconds"] == 11.0
    error_path = config.artifact_dir / "videos" / "clip" / "chunks" / "chunk-0000.error.json"
    error_payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_payload["chunk_id"] == "chunk-0000"
    assert "No JSON object found" in error_payload["error"]


def test_extract_rows_for_eval_job_removes_stale_error_after_chunk_success(tmp_path, monkeypatch):
    probe = load_probe_module()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    config = probe.parse_args(
        [
            "--dataset-eval",
            "--dataset-dir",
            str(tmp_path),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--chunk-seconds",
            "10",
            "--chunk-overlap-seconds",
            "0",
        ]
    )
    job = probe.EvalVideoJob(
        video_name="clip",
        video_path=video_path,
        duration_seconds=10.0,
        queries=("Question?",),
        query_expectations=({"source_video_name": "clip"},),
    )
    error_path = config.artifact_dir / "videos" / "clip" / "chunks" / "chunk-0000.error.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text('{"error": "stale"}\n', encoding="utf-8")

    def fake_extract_video_chunk(source_video, chunk, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"chunk")

    monkeypatch.setattr(probe, "extract_video_chunk", fake_extract_video_chunk)
    monkeypatch.setattr(probe, "extract_representative_frames_from_segments", lambda video_path, segments, artifact_dir: {})
    monkeypatch.setattr(
        probe,
        "call_nano_omni",
        lambda chunk_path, chunk_config, token: {
            "elapsed_seconds": 0.25,
            "request": {},
            "raw_response": {},
            "parsed": {"segments": [dict(sample_segment(), segment_id="ok", start_seconds=1.0, end_seconds=2.0)]},
        },
    )

    rows_df = probe.extract_rows_for_eval_job(job, config, token="token")

    assert len(rows_df) == 1
    assert not error_path.exists()


def test_check_ffmpeg_missing_raises_clear_error(monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(probe.shutil, "which", lambda command: None)

    with pytest.raises(RuntimeError, match="Missing system ffmpeg"):
        probe.check_ffmpeg()


def test_video_data_url_encodes_mp4(tmp_path):
    probe = load_probe_module()
    video_path = tmp_path / "fixture.mp4"
    video_bytes = b"\x00\x00\x00\x18ftypmp42"
    video_path.write_bytes(video_bytes)

    data_url = probe.video_data_url(video_path)

    assert data_url.startswith("data:video/mp4;base64,")
    encoded = data_url.removeprefix("data:video/mp4;base64,")
    assert probe.base64.b64decode(encoded) == video_bytes


def test_build_omni_payload_enables_audio_and_sampling():
    probe = load_probe_module()

    payload = probe.build_omni_payload("data:video/mp4;base64,AAAA", "test-model")

    assert payload["model"] == "test-model"
    assert payload["mm_processor_kwargs"] == {"use_audio_in_video": True}
    assert payload["media_io_kwargs"]["video"]["num_frames"] == 16
    assert payload["messages"][0]["content"][1]["type"] == "video_url"


def test_parse_args_allow_custom_endpoint_flag(tmp_path):
    probe = load_probe_module()

    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--allow-custom-endpoint"])

    assert config.allow_custom_endpoint is True


def test_require_allowed_endpoint_rejects_custom_https_without_override():
    probe = load_probe_module()

    with pytest.raises(ValueError, match="allow-custom-endpoint"):
        probe.require_allowed_endpoint("https://example.test/v1")


def test_require_allowed_endpoint_accepts_default_nvidia_endpoint_and_explicit_override():
    probe = load_probe_module()

    probe.require_allowed_endpoint(probe.CHAT_ENDPOINT)
    probe.require_allowed_endpoint("https://example.test/v1", allow_custom_endpoint=True)


def test_build_omni_prompt_contains_schema_and_fixture_cues():
    probe = load_probe_module()

    prompt = probe.build_omni_prompt()

    assert "segments" in prompt
    assert "ALPHA-17" in prompt
    assert "known_fixture_cues" in prompt


def test_build_omni_prompt_for_external_video_omits_fixture_cues():
    probe = load_probe_module()

    prompt = probe.build_omni_prompt(include_fixture_cues=False, video_name="clip_one")

    assert "segments" in prompt
    assert "clip_one" in prompt
    assert "ALPHA-17" not in prompt
    assert "known_fixture_cues" not in prompt


def test_extract_message_content_and_validate_segments():
    probe = load_probe_module()
    response = {"choices": [{"message": {"content": '{"segments": [{"segment_id": "scene-alpha", "start_seconds": 0, "end_seconds": 2.5, "summary": "Alpha scene", "visual_text": ["ALPHA-17"], "objects": ["red triangle"], "actions": ["static display"], "audio_or_speech": "none", "retrieval_keywords": ["alpha"], "uncertainties": [], "confidence": 0.9}]}'}}]}

    content = probe.extract_message_content(response)
    segments = probe.validate_omni_segments(probe.extract_json_object(content))

    assert segments == [
        {
            "segment_id": "scene-alpha",
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "summary": "Alpha scene",
            "visual_text": ["ALPHA-17"],
            "objects": ["red triangle"],
            "actions": ["static display"],
            "audio_or_speech": "none",
            "retrieval_keywords": ["alpha"],
            "uncertainties": [],
            "confidence": 0.9,
        }
    ]


def test_validate_omni_segments_rejects_empty_or_bad_segments():
    probe = load_probe_module()

    with pytest.raises(ValueError, match="non-empty 'segments' list"):
        probe.validate_omni_segments({"segments": []})
    with pytest.raises(ValueError, match="Segment 0 must be an object"):
        probe.validate_omni_segments({"segments": ["bad"]})


def test_validate_omni_segments_accepts_single_segment_object():
    probe = load_probe_module()

    segments = probe.validate_omni_segments(dict(sample_segment(), segment_id="single"))

    assert len(segments) == 1
    assert segments[0]["segment_id"] == "single"


def test_validate_omni_segments_coerces_string_lists_without_char_splitting():
    probe = load_probe_module()

    segments = probe.validate_omni_segments(
        {
            "segments": [
                {
                    "summary": "String fields",
                    "visual_text": "ALPHA-17",
                    "objects": ("red triangle", 17),
                    "actions": None,
                    "retrieval_keywords": "alpha, warning",
                }
            ]
        }
    )

    assert segments[0]["visual_text"] == ["ALPHA-17"]
    assert segments[0]["objects"] == ["red triangle", "17"]
    assert segments[0]["actions"] == []
    assert segments[0]["retrieval_keywords"] == ["alpha, warning"]


def test_validate_omni_segments_defaults_non_finite_values_and_clamps_confidence():
    probe = load_probe_module()

    segments = probe.validate_omni_segments(
        {
            "segments": [
                {"start_seconds": "nan", "end_seconds": "inf", "confidence": float("-inf")},
                {"start_seconds": 2.0, "end_seconds": 1.0, "confidence": 2.5},
                {"start_seconds": 1.0, "end_seconds": 2.0, "confidence": -0.5},
            ]
        }
    )

    assert segments[0]["start_seconds"] == 0.0
    assert segments[0]["end_seconds"] == 0.0
    assert segments[0]["confidence"] == 0.0
    assert segments[1]["end_seconds"] == 2.0
    assert segments[1]["confidence"] == 1.0
    assert segments[2]["confidence"] == 0.0


def test_flatten_str_list_handles_empty_values():
    probe = load_probe_module()

    assert probe.flatten_str_list([" ALPHA-17 ", "", None, 17, "  "]) == ["ALPHA-17", "17"]


def test_build_retrieval_text_contains_visual_audio_and_keywords():
    probe = load_probe_module()

    text = probe.build_retrieval_text(sample_segment())

    assert "Segment: scene-alpha" in text
    assert "Visual text: ALPHA-17" in text
    assert "Objects: red triangle, blue crate" in text
    assert "Actions: static display, warning shown" in text
    assert "Audio or speech: A quiet audio check tone is present." in text
    assert "Keywords: alpha, warning, crate" in text


def test_nearest_scene_frame_prefers_exact_and_unknown_without_timing_returns_empty():
    probe = load_probe_module()

    frame_b64 = {"scene-alpha": "alpha-frame", "scene-beta": "beta-frame"}

    assert probe.nearest_scene_frame("scene-alpha", frame_b64) == "alpha-frame"
    assert probe.nearest_scene_frame("missing", frame_b64) == ""
    assert probe.nearest_scene_frame("missing", {}) == ""


def test_nearest_scene_frame_uses_unknown_segment_time_midpoint():
    probe = load_probe_module()

    frame_b64 = {"scene-alpha": "alpha-frame", "scene-beta": "beta-frame"}
    segment = {"start_seconds": 2.7, "end_seconds": 3.1}

    assert probe.nearest_scene_frame("model-scene-1", frame_b64, segment) == "beta-frame"


def test_rows_from_segments_unknown_id_without_timing_uses_text_modality(tmp_path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    segment = dict(sample_segment(), segment_id="unknown", start_seconds=99.0, end_seconds=100.0)

    rows_df = probe.rows_from_segments([segment], {"scene-alpha": "alpha-frame"}, fixture_path, config)
    row = rows_df.to_dict(orient="records")[0]

    assert row["_image_b64"] == ""
    assert row["_embed_modality"] == "text"


def test_rows_from_segments_unknown_id_with_default_zero_time_uses_text_modality(tmp_path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    segment = dict(sample_segment(), segment_id="unknown", start_seconds=0.0, end_seconds=0.0)

    rows_df = probe.rows_from_segments([segment], {"scene-alpha": "alpha-frame"}, fixture_path, config)
    row = rows_df.to_dict(orient="records")[0]

    assert row["_image_b64"] == ""
    assert row["_embed_modality"] == "text"


def test_rows_from_segments_unknown_id_with_known_time_uses_matching_frame(tmp_path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    segment = dict(sample_segment(), segment_id="unknown", start_seconds=5.2, end_seconds=5.8)
    frame_b64 = {"scene-alpha": "alpha-frame", "scene-calibration": "calibration-frame"}

    rows_df = probe.rows_from_segments([segment], frame_b64, fixture_path, config)
    row = rows_df.to_dict(orient="records")[0]

    assert row["_image_b64"] == "calibration-frame"
    assert row["_embed_modality"] == "text_image"


def test_rows_from_segments_preserves_metadata(tmp_path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--chat-endpoint", "https://chat.test/v1", "--embed-endpoint", "https://embed.test/v1"])
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    frame_b64 = {"scene-alpha": "alpha-frame"}

    rows_df = probe.rows_from_segments([sample_segment()], frame_b64, fixture_path, config)
    rows = rows_df.to_dict(orient="records")

    assert len(rows) == 1
    row = rows[0]
    assert {"segment_id", "text", "_image_b64", "_embed_modality", "metadata"}.issubset(row)
    assert row["segment_id"] == "scene-alpha"
    assert row["text"].startswith("Segment: scene-alpha")
    assert row["_image_b64"] == "alpha-frame"
    assert row["_embed_modality"] == "text_image"
    assert row["metadata"]["source_path"] == str(fixture_path)
    assert row["metadata"]["segment_id"] == "scene-alpha"
    assert row["metadata"]["start_seconds"] == 0.0
    assert row["metadata"]["end_seconds"] == 2.5
    assert row["metadata"]["raw_omni_segment"] == sample_segment()
    assert row["metadata"]["extraction_model"] == config.omni_model
    assert row["metadata"]["embedding_model"] == config.embed_model
    assert row["metadata"]["prompt_version"] == probe.PROMPT_VERSION
    assert row["metadata"]["scaffold"] == {
        "chat_endpoint": "https://chat.test/v1",
        "embed_endpoint": "https://embed.test/v1",
    }


def test_extract_representative_frames_from_segments_uses_segment_midpoint(tmp_path, monkeypatch):
    probe = load_probe_module()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    commands = []

    def fake_run_command(args):
        commands.append(args)
        Path(args[-1]).write_bytes(b"frame-bytes")

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "run_command", fake_run_command)

    frames = probe.extract_representative_frames_from_segments(
        video_path,
        [{"segment_id": "Segment 1/Alpha", "start_seconds": 10.0, "end_seconds": 14.0}],
        tmp_path,
    )

    assert probe.base64.b64decode(frames["Segment 1/Alpha"]) == b"frame-bytes"
    assert commands == [
        [
            "ffmpeg",
            "-y",
            "-ss",
            "12.000",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:-1",
            str(tmp_path / "representative_frames" / "Segment_1_Alpha.png"),
        ]
    ]


def test_extract_representative_frames_skips_segments_without_positive_time(tmp_path, monkeypatch):
    probe = load_probe_module()
    calls = []

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: calls.append("ffmpeg"))
    monkeypatch.setattr(probe, "run_command", lambda args: (_ for _ in ()).throw(AssertionError("no frame should be extracted")))

    frames = probe.extract_representative_frames_from_segments(
        tmp_path / "clip.mp4",
        [{"segment_id": "bad", "start_seconds": 0.0, "end_seconds": 0.0}],
        tmp_path,
    )

    assert frames == {}
    assert calls == ["ffmpeg"]


def test_extract_representative_frames_degrades_to_text_when_ffmpeg_missing(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: (_ for _ in ()).throw(RuntimeError("missing ffmpeg")))
    monkeypatch.setattr(probe, "run_command", lambda args: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")))

    frames = probe.extract_representative_frames_from_segments(
        tmp_path / "clip.mp4",
        [{"segment_id": "segment-1", "start_seconds": 1.0, "end_seconds": 2.0}],
        tmp_path,
    )

    assert frames == {}
    assert "Skipping representative frame extraction" in capsys.readouterr().out


def test_extract_representative_frames_skips_negative_segments(tmp_path, monkeypatch):
    probe = load_probe_module()

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "run_command", lambda args: (_ for _ in ()).throw(AssertionError("no frame should be extracted")))

    frames = probe.extract_representative_frames_from_segments(
        tmp_path / "clip.mp4",
        [{"segment_id": "negative", "start_seconds": -10.0, "end_seconds": -2.0}],
        tmp_path,
    )

    assert frames == {}


def test_extract_representative_frames_degrades_to_text_when_ffmpeg_fails(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()

    monkeypatch.setattr(probe, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(probe, "run_command", lambda args: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")))

    frames = probe.extract_representative_frames_from_segments(
        tmp_path / "clip.mp4",
        [{"segment_id": "bad-window", "start_seconds": 10.0, "end_seconds": 12.0}],
        tmp_path,
    )

    assert frames == {}
    assert "Skipping representative frame for segment bad-window" in capsys.readouterr().out


def test_rows_from_segments_marks_text_modality_without_image(tmp_path):
    probe = load_probe_module()
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    fixture_path = tmp_path / "synthetic_fixture.mp4"

    rows_df = probe.rows_from_segments([sample_segment()], {}, fixture_path, config)
    row = rows_df.to_dict(orient="records")[0]

    assert row["_image_b64"] == ""
    assert row["_embed_modality"] == "text"


def test_write_json_creates_parent_and_pretty_json(tmp_path):
    probe = load_probe_module()
    path = tmp_path / "nested" / "payload.json"

    probe.write_json(path, {"b": 2, "a": {"c": 3}})

    assert path.read_text(encoding="utf-8") == '{\n  "a": {\n    "c": 3\n  },\n  "b": 2\n}\n'


def test_write_json_rejects_non_standard_nan(tmp_path):
    probe = load_probe_module()
    path = tmp_path / "bad.json"

    with pytest.raises(ValueError):
        probe.write_json(path, {"x": float("nan")})


def test_embedding_from_payload_extracts_embedding_and_rejects_bad_payloads():
    probe = load_probe_module()

    assert probe.embedding_from_payload({"embedding": [1, "2.5", 3]}) == [1.0, 2.5, 3.0]
    assert probe.embedding_from_payload({"embedding": "bad"}) == []
    assert probe.embedding_from_payload(None) == []


def test_rank_rows_orders_by_cosine_similarity():
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "far",
                "text": "far row",
                "text_embedding": {"embedding": [0.0, 1.0]},
                "metadata": {"start_seconds": 2.0, "end_seconds": 3.0, "raw_omni_segment": {"segment_id": "far"}},
            },
            {
                "segment_id": "near",
                "text": "near row",
                "text_embedding": {"embedding": [1.0, 0.0]},
                "metadata": {"start_seconds": 0.0, "end_seconds": 1.0, "raw_omni_segment": {"segment_id": "near"}},
            },
        ]
    )

    hits = probe.rank_rows(rows_df, [1.0, 0.0], "text_embedding", top_k=2)

    assert [hit["segment_id"] for hit in hits] == ["near", "far"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["start_seconds"] == 0.0
    assert hits[0]["raw_omni_segment"] == {"segment_id": "near"}


def test_rank_rows_includes_source_video_name_for_golden_eval():
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "clip-one-segment",
                "text": "answer row",
                "text_embedding": {"embedding": [1.0, 0.0]},
                "metadata": {
                    "source_video_name": "clip_one",
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                    "raw_omni_segment": {},
                },
            }
        ]
    )

    hits = probe.rank_rows(rows_df, [1.0, 0.0], "text_embedding", top_k=1)

    assert hits[0]["source_video_name"] == "clip_one"


def test_rank_rows_raises_for_empty_query_embedding():
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "scene-alpha",
                "text": "Alpha warning",
                "text_embedding": {"embedding": [1.0, 0.0]},
                "metadata": {},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Query embedding is empty"):
        probe.rank_rows(rows_df, [], "text_embedding", top_k=1)


def test_rank_rows_raises_for_empty_row_embedding():
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "scene-alpha",
                "text": "Alpha warning",
                "text_embedding": {"embedding": []},
                "metadata": {},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Missing or empty embedding.*text_embedding.*scene-alpha"):
        probe.rank_rows(rows_df, [1.0, 0.0], "text_embedding", top_k=1)


def test_rank_rows_raises_for_query_document_dimension_mismatch():
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "scene-alpha",
                "text": "Alpha warning",
                "text_embedding": {"embedding": [1.0, 0.0, 0.0]},
                "metadata": {},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="dimension mismatch.*text_embedding.*query dimension 2.*row dimension 3.*scene-alpha"):
        probe.rank_rows(rows_df, [1.0, 0.0], "text_embedding", top_k=1)


def test_expected_query_terms_and_hit_contains_terms():
    probe = load_probe_module()

    assert probe.expected_query_terms("Which segment shows ALPHA-17?") == ("alpha-17", "warning", "scene-alpha")
    assert probe.expected_query_terms("Any calibration panel?") == ("calibration",)
    assert probe.expected_query_terms("spoken audio") == ("audio", "speech", "spoken", "waveform")
    assert probe.expected_query_terms("colored object") == ("red triangle", "warning", "alpha-17")
    assert probe.expected_query_terms("unmatched") == ()
    assert probe.hit_contains_terms({"text": "The red triangle is visible."}, ("green square", "red triangle")) is True
    assert probe.hit_contains_terms({"text": "No match here."}, ("calibration",)) is False
    assert probe.hit_contains_terms({"text": "ALPHA-17"}, ()) is False


def test_hit_contains_terms_uses_evidence_not_raw_keys():
    probe = load_probe_module()
    hit = {
        "segment_id": "scene-beta",
        "text": "unrelated visual evidence",
        "raw_omni_segment": {"audio_or_speech": ""},
    }

    assert probe.hit_contains_terms(hit, probe.expected_query_terms("spoken audio")) is False


def test_audio_hit_requires_audio_content_not_segment_id_only():
    probe = load_probe_module()
    terms = probe.expected_query_terms("Which segment includes spoken or audio content?")
    segment_id_only_hit = {
        "segment_id": "scene-audio",
        "text": "Segment: scene-audio\nSummary: Fixture segment marker with static visual details.",
        "raw_omni_segment": {
            "segment_id": "scene-audio",
            "summary": "",
            "visual_text": [],
            "objects": [],
            "actions": [],
            "audio_or_speech": "",
            "retrieval_keywords": [],
            "uncertainties": [],
        },
    }

    assert probe.hit_contains_terms(segment_id_only_hit, terms) is False
    assert probe.hit_contains_terms({"text": "This segment contains spoken content."}, terms) is True
    assert probe.hit_contains_terms({"text": "A waveform appears in the extracted evidence."}, terms) is True


def test_warning_colored_object_requires_red_triangle_evidence():
    probe = load_probe_module()
    terms = probe.expected_query_terms("What colored object appears with the warning text?")

    assert probe.hit_contains_terms({"text": "Green square status panel with warning-like styling."}, terms) is False
    assert probe.hit_contains_terms({"text": "Purple circle confirms camera alignment."}, terms) is False
    assert probe.hit_contains_terms({"text": "ALPHA-17 warning beside a red triangle."}, terms) is True


def test_embed_query_uses_query_input_type(monkeypatch):
    probe = load_probe_module()
    calls = []

    def fake_embed_dataframe(rows_df, **kwargs):
        calls.append((rows_df, kwargs))
        return probe.pd.DataFrame([{"query_embedding": {"embedding": [1.0, 0.0]}}])

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)

    assert probe.embed_query("hello", api_key="token", endpoint="https://embed.test/v1", model_name="model") == [1.0, 0.0]
    rows_df, kwargs = calls[0]
    assert rows_df.loc[0, "text"] == "hello"
    assert kwargs["input_type"] == "query"
    assert kwargs["modality"] == "text"
    assert kwargs["output_column"] == "query_embedding"
    assert kwargs["batch_size"] == 1


def test_compare_query_results_reports_top_result_change(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {"segment_id": "scene-alpha", "text": "Alpha warning", "metadata": {"start_seconds": 0.0, "end_seconds": 1.0}},
            {"segment_id": "scene-calibration", "text": "Calibration panel", "metadata": {"start_seconds": 1.0, "end_seconds": 2.0}},
        ]
    )

    def fake_embed_dataframe(input_df, **kwargs):
        out_df = input_df.copy()
        if kwargs["output_column"] == "text_embedding":
            out_df["text_embedding"] = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        elif kwargs["output_column"] == "vl_text_image_embedding":
            out_df["vl_text_image_embedding"] = [{"embedding": [0.0, 1.0]}, {"embedding": [1.0, 0.0]}]
        else:
            out_df[kwargs["output_column"]] = [{"embedding": [1.0, 0.0]}]
        return out_df

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)

    results = probe.compare_query_results(
        ("Which segment shows ALPHA-17?",),
        rows_df,
        api_key="token",
        endpoint="https://embed.test/v1",
        model_name="model",
        top_k=1,
    )

    assert results[0]["text_only"][0]["segment_id"] == "scene-alpha"
    assert results[0]["vl_text_image"][0]["segment_id"] == "scene-calibration"
    assert results[0]["text_only_top_contains_expected"] is True
    assert results[0]["vl_top_contains_expected"] is False
    assert results[0]["top_result_changed"] is True


def test_compare_query_results_uses_dedicated_text_embed_model_and_vl_embed_model(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {"segment_id": "text-hit", "text": "Text model answer", "metadata": {"start_seconds": 0.0, "end_seconds": 1.0}},
            {"segment_id": "vl-hit", "text": "VL model answer", "metadata": {"start_seconds": 1.0, "end_seconds": 2.0}},
        ]
    )
    embed_calls = []
    query_calls = []

    def fake_embed_dataframe(input_df, **kwargs):
        embed_calls.append(kwargs)
        out_df = input_df.copy()
        if kwargs["output_column"] == "text_embedding":
            assert kwargs["model_name"] == "text-model"
            out_df["text_embedding"] = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        elif kwargs["output_column"] == "vl_text_image_embedding":
            assert kwargs["model_name"] == "vl-model"
            out_df["vl_text_image_embedding"] = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        else:
            raise AssertionError(f"unexpected output column {kwargs['output_column']}")
        return out_df

    def fake_embed_query(query, **kwargs):
        query_calls.append(kwargs)
        if kwargs["model_name"] == "text-model":
            return [1.0, 0.0]
        if kwargs["model_name"] == "vl-model":
            return [0.0, 1.0]
        raise AssertionError(f"unexpected model {kwargs['model_name']}")

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)
    monkeypatch.setattr(probe, "embed_query", fake_embed_query)

    results = probe.compare_query_results(
        ("Which answer?",),
        rows_df,
        api_key="token",
        endpoint=probe.EMBED_ENDPOINT,
        model_name="vl-model",
        text_model_name="text-model",
        top_k=1,
    )

    assert [call["model_name"] for call in embed_calls] == ["text-model", "vl-model"]
    assert [call["model_name"] for call in query_calls] == ["text-model", "vl-model"]
    assert results[0]["text_embedding_model"] == "text-model"
    assert results[0]["vl_embedding_model"] == "vl-model"
    assert results[0]["text_only"][0]["segment_id"] == "text-hit"
    assert results[0]["vl_text_image"][0]["segment_id"] == "vl-hit"


def test_compare_query_results_reports_expected_time_overlap(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {"segment_id": "answer-window", "text": "Answer row", "metadata": {"start_seconds": 30.0, "end_seconds": 40.0}},
            {"segment_id": "wrong-window", "text": "Wrong row", "metadata": {"start_seconds": 80.0, "end_seconds": 90.0}},
        ]
    )

    def fake_embed_dataframe(input_df, **kwargs):
        out_df = input_df.copy()
        if kwargs["output_column"] == "text_embedding":
            out_df["text_embedding"] = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        elif kwargs["output_column"] == "vl_text_image_embedding":
            out_df["vl_text_image_embedding"] = [{"embedding": [0.0, 1.0]}, {"embedding": [1.0, 0.0]}]
        else:
            out_df[kwargs["output_column"]] = [{"embedding": [1.0, 0.0]}]
        return out_df

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)

    results = probe.compare_query_results(
        ("Where is the answer shown?",),
        rows_df,
        api_key="token",
        endpoint="https://embed.test/v1",
        model_name="model",
        top_k=1,
        query_expectations=(
            {
                "expected_answer": "Answer text",
                "expected_start_seconds": 32.0,
                "expected_end_seconds": 34.0,
            },
        ),
    )

    assert results[0]["expected_answer"] == "Answer text"
    assert results[0]["expected_time_range"] == {"start_seconds": 32.0, "end_seconds": 34.0}
    assert results[0]["text_only_top_overlaps_expected_time"] is True
    assert results[0]["vl_top_overlaps_expected_time"] is False
    assert results[0]["text_only_any_overlaps_expected_time"] is True
    assert results[0]["vl_any_overlaps_expected_time"] is False


def test_expected_time_overlap_requires_matching_source_video():
    probe = load_probe_module()
    expectation = {
        "source_video_name": "clip_one",
        "expected_start_seconds": 10.0,
        "expected_end_seconds": 20.0,
    }

    assert (
        probe.hit_overlaps_expected_time(
            {"source_video_name": "clip_two", "start_seconds": 12.0, "end_seconds": 14.0},
            expectation,
        )
        is False
    )
    assert (
        probe.hit_overlaps_expected_time(
            {"source_video_name": "clip_one", "start_seconds": 12.0, "end_seconds": 14.0},
            expectation,
        )
        is True
    )


def test_compare_query_results_uses_external_expected_answer_when_terms_are_unknown(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {"segment_id": "answer", "text": "The answer is a personal AI assistant.", "metadata": {}},
            {"segment_id": "other", "text": "Unrelated text", "metadata": {}},
        ]
    )

    def fake_embed_dataframe(input_df, **kwargs):
        out_df = input_df.copy()
        out_df[kwargs["output_column"]] = [
            {"embedding": [1.0, 0.0]} if index == 0 else {"embedding": [0.0, 1.0]}
            for index in range(len(out_df))
        ]
        return out_df

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)

    results = probe.compare_query_results(
        ("What is Microsoft Copilot?",),
        rows_df,
        api_key="token",
        endpoint=probe.EMBED_ENDPOINT,
        model_name="model",
        top_k=1,
        query_expectations=({"expected_answer": "A personal AI assistant"},),
    )

    assert results[0]["expected_terms"] == []
    assert results[0]["text_only_top_contains_expected"] is True
    assert results[0]["vl_top_contains_expected"] is True


def test_compare_query_results_returns_empty_without_embedding_zero_queries(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"segment_id": "scene-alpha", "text": "Alpha warning", "metadata": {}}])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("zero diagnostic queries should not embed rows")

    monkeypatch.setattr(probe, "embed_dataframe", fail_if_called)

    assert (
        probe.compare_query_results(
            (),
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            top_k=1,
        )
        == []
    )


def test_summarize_observations_counts_baseline_and_vl_hits(tmp_path):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"fixture bytes")

    observations = probe.summarize_observations(
        omni_elapsed_seconds=1.25,
        fixture_path=fixture_path,
        query_results=[
            {
                "text_only_top_contains_expected": True,
                "vl_top_contains_expected": False,
                "top_result_changed": True,
            },
            {
                "text_only_top_contains_expected": False,
                "vl_top_contains_expected": True,
                "top_result_changed": False,
            },
        ],
    )

    assert observations["metrics"] == {
        "omni_elapsed_seconds": 1.25,
        "fixture_bytes": len(b"fixture bytes"),
        "text_only_expected_hits": 1,
        "vl_expected_hits": 1,
        "top_result_changes": 1,
        "query_count": 2,
    }
    assert "Text-only baseline hit expected terms for 1/2 diagnostic queries." in observations["pros"]
    assert "VL text+image hit expected terms for 1/2 diagnostic queries." in observations["pros"]
    assert "Top result changed between baseline and VL retrieval for 1/2 diagnostic queries." in observations["cons"]


def test_summarize_observations_accepts_positional_arguments(tmp_path):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"fixture bytes")
    query_results = [
        {
            "text_only_top_contains_expected": True,
            "vl_top_contains_expected": True,
            "top_result_changed": False,
        },
        {
            "text_only_top_contains_expected": False,
            "vl_top_contains_expected": True,
            "top_result_changed": True,
        },
    ]

    observations = probe.summarize_observations(1.25, fixture_path, query_results)

    assert observations["metrics"] == {
        "omni_elapsed_seconds": 1.25,
        "fixture_bytes": len(b"fixture bytes"),
        "text_only_expected_hits": 1,
        "vl_expected_hits": 2,
        "top_result_changes": 1,
        "query_count": 2,
    }


def test_summarize_observations_handles_missing_fixture_and_zero_queries(tmp_path):
    probe = load_probe_module()

    observations = probe.summarize_observations(
        omni_elapsed_seconds=0.0,
        fixture_path=tmp_path / "missing.mp4",
        query_results=[],
    )

    assert observations["metrics"]["fixture_bytes"] == 0
    assert observations["metrics"]["query_count"] == 0
    prose = "\n".join(observations["pros"] + observations["cons"])
    assert "0/0" not in prose
    assert "No diagnostic queries were run; retrieval hit metrics are unevaluated." in prose


def test_summarize_observations_warns_for_silent_fallback_audio(tmp_path):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"fixture bytes")
    probe.write_json(tmp_path / "synthetic_fixture_metadata.json", {"audio_source": "silent_fallback"})

    observations = probe.summarize_observations(
        omni_elapsed_seconds=0.5,
        fixture_path=fixture_path,
        query_results=[
            {
                "text_only_top_contains_expected": True,
                "vl_top_contains_expected": True,
                "top_result_changed": False,
            }
        ],
    )

    assert observations["metrics"]["audio_source"] == "silent_fallback"
    assert any("visual/text diagnostics only" in item for item in observations["cons"])


def test_summarize_observations_counts_expected_time_overlaps(tmp_path):
    probe = load_probe_module()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video bytes")

    observations = probe.summarize_observations(
        omni_elapsed_seconds=2.0,
        fixture_path=video_path,
        query_results=[
            {
                "text_only_top_overlaps_expected_time": True,
                "vl_top_overlaps_expected_time": False,
                "text_only_any_overlaps_expected_time": True,
                "vl_any_overlaps_expected_time": True,
            },
            {
                "text_only_top_overlaps_expected_time": False,
                "vl_top_overlaps_expected_time": True,
                "text_only_any_overlaps_expected_time": False,
                "vl_any_overlaps_expected_time": True,
            },
        ],
        fixture_metadata={"source_kind": "external_video"},
    )

    assert observations["metrics"]["text_only_top_time_overlaps"] == 1
    assert observations["metrics"]["vl_top_time_overlaps"] == 1
    assert observations["metrics"]["text_only_any_time_overlaps"] == 1
    assert observations["metrics"]["vl_any_time_overlaps"] == 2
    assert "external video" in observations["pros"][0]
    prose = "\n".join(observations["pros"] + observations["cons"])
    assert "Text-only top result overlapped expected timestamp windows for 1/2 diagnostic queries." in prose
    assert "VL top result overlapped expected timestamp windows for 1/2 diagnostic queries." in prose


def test_summarize_dataset_eval_results_reports_text_and_vl_metrics():
    probe = load_probe_module()
    query_results = [
        {
            "answer_modality": "Audio + Visual",
            "expected_answer": "answer one",
            "source_video_name": "clip_one",
            "expected_time_range": {"start_seconds": 10.0, "end_seconds": 20.0},
            "text_only": [
                {"source_video_name": "clip_one", "start_seconds": 12.0, "end_seconds": 14.0, "text": "answer one"}
            ],
            "vl_text_image": [
                {"source_video_name": "clip_two", "start_seconds": 12.0, "end_seconds": 14.0, "text": "answer one"}
            ],
            "text_only_top_contains_expected": True,
            "vl_top_contains_expected": True,
            "text_only_top_overlaps_expected_time": True,
            "vl_top_overlaps_expected_time": False,
            "text_only_any_overlaps_expected_time": True,
            "vl_any_overlaps_expected_time": False,
        },
        {
            "answer_modality": "Visual",
            "expected_answer": "answer two",
            "source_video_name": "clip_one",
            "expected_time_range": {"start_seconds": 30.0, "end_seconds": 40.0},
            "text_only": [
                {"source_video_name": "clip_one", "start_seconds": 1.0, "end_seconds": 2.0, "text": "miss"},
                {"source_video_name": "clip_one", "start_seconds": 32.0, "end_seconds": 34.0, "text": "answer two"},
            ],
            "vl_text_image": [
                {"source_video_name": "clip_one", "start_seconds": 32.0, "end_seconds": 34.0, "text": "answer two"}
            ],
            "text_only_top_contains_expected": False,
            "vl_top_contains_expected": True,
            "text_only_top_overlaps_expected_time": False,
            "vl_top_overlaps_expected_time": True,
            "text_only_any_overlaps_expected_time": True,
            "vl_any_overlaps_expected_time": True,
        },
    ]

    summary = probe.summarize_dataset_eval_results(query_results, top_k=2)

    assert summary["metrics"]["query_count"] == 2
    assert summary["metrics"]["text_only_top1_time_overlap_rate"] == 0.5
    assert summary["metrics"]["vl_top1_time_overlap_rate"] == 0.5
    assert summary["metrics"]["text_only_top2_time_overlap_rate"] == 1.0
    assert summary["metrics"]["vl_top2_time_overlap_rate"] == 0.5
    assert summary["metrics"]["text_only_time_mrr"] == pytest.approx(0.75)
    assert summary["metrics"]["vl_time_mrr"] == pytest.approx(0.5)
    assert summary["metrics"]["text_only_answer_hit_rate"] == 0.5
    assert summary["metrics"]["vl_answer_hit_rate"] == 1.0
    assert summary["by_answer_modality"]["Visual"]["vl_top1_time_overlap_rate"] == 1.0


def test_print_query_results_includes_expected_terms_and_snippets(capsys):
    probe = load_probe_module()
    long_text = "  ".join(["Alpha"] * 60)

    probe.print_query_results(
        [
            {
                "query": "Which segment shows ALPHA-17?",
                "expected_terms": ["alpha-17", "warning"],
                "text_only": [
                    {
                        "score": 0.98765,
                        "segment_id": "scene-alpha",
                        "start_seconds": 0.0,
                        "end_seconds": 2.5,
                        "text": f"  {long_text}\nwarning marker  ",
                    }
                ],
                "vl_text_image": [
                    {
                        "score": 0.5,
                        "segment_id": "scene-beta",
                        "start_seconds": 2.5,
                        "end_seconds": 5.0,
                        "text": "Beta panel text",
                    }
                ],
                "text_only_top_contains_expected": True,
                "vl_top_contains_expected": False,
                "top_result_changed": True,
            }
        ]
    )

    output = capsys.readouterr().out
    assert "Query: Which segment shows ALPHA-17?" in output
    assert "expected_terms: alpha-17, warning" in output
    assert "text_only:" in output
    assert "vl_text_image:" in output
    assert "rank=1 score=0.9877 segment=scene-alpha time=0.0-2.5" in output
    assert "snippet=Alpha Alpha Alpha" in output
    assert "text_only_top_contains_expected: True" in output
    assert "vl_top_contains_expected: False" in output
    assert "top_result_changed: True" in output


def test_print_query_results_renders_bad_score_as_na(capsys):
    probe = load_probe_module()

    probe.print_query_results(
        [
            {
                "query": "bad score",
                "expected_terms": [],
                "text_only": [{"score": "not-a-number", "segment_id": "scene-alpha", "text": "Alpha"}],
                "vl_text_image": [],
            }
        ]
    )

    output = capsys.readouterr().out
    assert "rank=1 score=n/a segment=scene-alpha" in output


def test_print_observations_outputs_pros_and_cons(capsys):
    probe = load_probe_module()

    probe.print_observations({"pros": ["good thing"], "cons": ["rough edge"], "metrics": {}})

    output = capsys.readouterr().out
    assert "Observations" in output
    assert "  Pros:" in output
    assert "    - good thing" in output
    assert "  Cons:" in output
    assert "    - rough edge" in output


def test_compare_query_results_propagates_query_document_dimension_mismatch(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "segment_id": "scene-alpha",
                "text": "Alpha warning",
                "metadata": {"start_seconds": 0.0, "end_seconds": 1.0},
            }
        ]
    )

    def fake_embed_dataframe(input_df, **kwargs):
        out_df = input_df.copy()
        out_df[kwargs["output_column"]] = [{"embedding": [1.0, 0.0]}]
        return out_df

    monkeypatch.setattr(probe, "embed_dataframe", fake_embed_dataframe)
    monkeypatch.setattr(probe, "embed_query", lambda query, **kwargs: [1.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="dimension mismatch.*text_embedding.*query dimension 3.*row dimension 2.*scene-alpha"):
        probe.compare_query_results(
            ("Which segment shows ALPHA-17?",),
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            top_k=1,
        )


def test_embed_dataframe_raises_when_no_embeddings(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "hello", "metadata": {}}])

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        out_df = rows_df.copy()
        out_df[transform_config.output_payload_column] = [{"not_embedding": []}]
        return out_df, {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    with pytest.raises(RuntimeError, match="No embeddings produced for text_embedding"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_rejects_http_endpoint_before_embedding_call(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "hello", "metadata": {}}])

    def fail_if_called(rows_df, task_config, transform_config):
        raise AssertionError("embedding helper should not receive API key for non-HTTPS endpoints")

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fail_if_called)

    with pytest.raises(ValueError, match="HTTPS"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="http://example.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_rejects_custom_https_without_override(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "hello", "metadata": {}}])

    def fail_if_called(rows_df, task_config, transform_config):
        raise AssertionError("embedding helper should not receive API key for custom endpoint without override")

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fail_if_called)

    with pytest.raises(ValueError, match="allow-custom-endpoint"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
        )


def test_embed_dataframe_raises_when_output_column_missing(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "hello", "metadata": {}}])

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        return rows_df.copy(), {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    with pytest.raises(RuntimeError, match="missing output column text_embedding"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_raises_when_row_count_changes(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "first", "metadata": {}}, {"text": "second", "metadata": {}}])

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        out_df = rows_df.head(1).copy()
        out_df[transform_config.output_payload_column] = [{"embedding": [1.0, 0.0]}]
        return out_df, {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    with pytest.raises(RuntimeError, match="returned 1 rows for 2 input rows"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_raises_when_any_embedding_missing(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "first", "metadata": {}}, {"text": "second", "metadata": {}}])

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        out_df = rows_df.copy()
        out_df[transform_config.output_payload_column] = [{"embedding": [1.0, 0.0]}, {"not_embedding": []}]
        return out_df, {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    with pytest.raises(RuntimeError, match="Missing or empty embedding at row 1"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_raises_when_embedding_dimensions_differ(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame([{"text": "first", "metadata": {}}, {"text": "second", "metadata": {}}])

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        out_df = rows_df.copy()
        out_df[transform_config.output_payload_column] = [{"embedding": [1.0, 0.0]}, {"embedding": [1.0]}]
        return out_df, {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    with pytest.raises(RuntimeError, match="Inconsistent embedding dimensions"):
        probe.embed_dataframe(
            rows_df,
            api_key="token",
            endpoint="https://embed.test/v1",
            model_name="model",
            modality="text",
            output_column="text_embedding",
            allow_custom_endpoint=True,
        )


def test_embed_dataframe_text_modality_uses_text_only_config(monkeypatch):
    probe = load_probe_module()
    rows_df = probe.pd.DataFrame(
        [
            {
                "text": "Alpha text",
                "_image_b64": "image-payload",
                "_embed_modality": "text_image",
                "metadata": {},
            }
        ]
    )
    captured = {}

    def fake_create_text_embeddings_for_df(rows_df, task_config, transform_config):
        captured["transform_config"] = transform_config
        out_df = rows_df.copy()
        out_df[transform_config.output_payload_column] = [{"embedding": [1.0, 0.0]}]
        return out_df, {}

    monkeypatch.setattr(probe, "create_text_embeddings_for_df", fake_create_text_embeddings_for_df)

    probe.embed_dataframe(
        rows_df,
        api_key="token",
        endpoint="https://embed.test/v1",
        model_name="model",
        modality="text",
        output_column="text_embedding",
        allow_custom_endpoint=True,
    )

    assert captured["transform_config"].embed_modality == "text"
    assert captured["transform_config"].text_column == "text"


def test_post_json_reports_http_errors(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        return SimpleNamespace(status_code=401, text="bad token")

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP 401.*bad token"):
        probe.post_json("https://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True)


def test_post_json_rejects_http_before_request(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        raise AssertionError("requests.post should not be called for non-HTTPS endpoints")

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(ValueError, match="HTTPS"):
        probe.post_json("http://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True)


def test_post_json_rejects_custom_https_endpoint_before_request(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        raise AssertionError("requests.post should not receive API key for custom endpoint without override")

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(ValueError, match="allow-custom-endpoint"):
        probe.post_json("https://example.test/chat", "token", {"model": "x"})


def test_post_json_wraps_request_exceptions(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        raise probe.requests.RequestException("connection failed")

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Request to https://example.test/chat failed.*connection failed"):
        probe.post_json("https://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True)


def test_post_json_rejects_non_json_success(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        return SimpleNamespace(status_code=200, text="not json", json=lambda: (_ for _ in ()).throw(ValueError("bad json")))

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Expected JSON object from https://example.test/chat"):
        probe.post_json("https://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True)


def test_post_json_rejects_non_object_success(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        return SimpleNamespace(status_code=200, text="[1, 2]", json=lambda: [1, 2])

    monkeypatch.setattr(probe.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="Expected JSON object from https://example.test/chat"):
        probe.post_json("https://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True)


def test_post_json_returns_json_object(monkeypatch):
    probe = load_probe_module()

    def fake_post(url, headers, json, timeout):
        return SimpleNamespace(status_code=200, text='{"ok": true}', json=lambda: {"ok": True})

    monkeypatch.setattr(probe.requests, "post", fake_post)

    assert probe.post_json("https://example.test/chat", "token", {"model": "x"}, allow_custom_endpoint=True) == {"ok": True}


def test_call_nano_omni_records_request_metadata(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "fixture.mp4"
    fixture_path.write_bytes(b"video")
    config = probe.parse_args(["--artifact-dir", str(tmp_path), "--chat-endpoint", "https://example.test/chat", "--omni-model", "test-model"])

    monkeypatch.setattr(probe.time, "perf_counter", iter([10.0, 12.5]).__next__)
    monkeypatch.setattr(
        probe,
        "post_json",
        lambda url, token, payload, **kwargs: {
            "choices": [{"message": {"content": '{"segments": [{"segment_id": "scene-alpha"}]}'}}],
        },
    )

    result = probe.call_nano_omni(fixture_path, config, "token")

    assert result["elapsed_seconds"] == 2.5
    assert result["request"]["model"] == "test-model"
    assert result["request"]["endpoint"] == "https://example.test/chat"
    assert result["request"]["prompt_version"] == probe.PROMPT_VERSION
    assert result["request"]["fixture_sha256"] == probe.file_sha256(fixture_path)
    assert result["request"]["media_io_kwargs"] == {"video": {"num_frames": 16, "fps": -1}}
    assert result["request"]["mm_processor_kwargs"] == {"use_audio_in_video": True}
    assert result["parsed"] == {"segments": [{"segment_id": "scene-alpha"}]}


def test_main_uses_valid_cached_omni_response_without_omni_api_call(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"cached fixture")
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    cached_response = {
        "elapsed_seconds": 1.0,
        "request": probe.omni_request_metadata(fixture_path, config),
        "raw_response": {},
        "parsed": {"segments": [{"segment_id": "scene-alpha", "visual_text": "ALPHA-17"}]},
    }
    (tmp_path / "omni_response.json").write_text(json.dumps(cached_response), encoding="utf-8")

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (fixture_path, {}))
    monkeypatch.setenv(probe.API_KEY_ENV, "embed-token")
    monkeypatch.setattr(probe, "compare_query_results", lambda queries, rows_df, **kwargs: [])
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: (_ for _ in ()).throw(AssertionError("cache should be used")))

    assert probe.main(["--artifact-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert f"Loaded cached Omni response: {tmp_path / 'omni_response.json'}" in output
    assert "Validated Omni segments: 1" in output


def test_main_writes_extracted_rows_from_cached_omni(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"cached fixture")
    frame_b64 = {"scene-alpha": "alpha-frame"}
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    cached_response = {
        "elapsed_seconds": 1.0,
        "request": probe.omni_request_metadata(fixture_path, config),
        "raw_response": {},
        "parsed": {"segments": [sample_segment()]},
    }
    (tmp_path / "omni_response.json").write_text(json.dumps(cached_response), encoding="utf-8")

    def fake_generate_synthetic_fixture(config):
        fixture_path.write_bytes(b"cached fixture")
        return fixture_path, frame_b64

    monkeypatch.setattr(probe, "generate_synthetic_fixture", fake_generate_synthetic_fixture)
    monkeypatch.setenv(probe.API_KEY_ENV, "embed-token")
    monkeypatch.setattr(probe, "compare_query_results", lambda queries, rows_df, **kwargs: [])
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: (_ for _ in ()).throw(AssertionError("cache should be used")))

    assert probe.main(["--artifact-dir", str(tmp_path)]) == 0

    rows_path = tmp_path / "extracted_rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["segment_id"] == "scene-alpha"
    assert "Visual text: ALPHA-17" in rows[0]["text"]
    assert rows[0]["_image_b64"] == "alpha-frame"
    assert rows[0]["metadata"]["source_path"] == str(fixture_path)
    assert rows[0]["metadata"]["raw_omni_segment"]["visual_text"] == ["ALPHA-17"]


def test_main_writes_query_results_from_cached_omni(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"cached fixture")
    frame_b64 = {"scene-alpha": "alpha-frame"}
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    cached_response = {
        "elapsed_seconds": 1.0,
        "request": probe.omni_request_metadata(fixture_path, config),
        "raw_response": {},
        "parsed": {"segments": [sample_segment()]},
    }
    (tmp_path / "omni_response.json").write_text(json.dumps(cached_response), encoding="utf-8")
    calls = []

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (fixture_path, frame_b64))
    monkeypatch.setenv(probe.API_KEY_ENV, "embed-token")
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: (_ for _ in ()).throw(AssertionError("cache should be used")))

    query_payload = [
        {
            "query": probe.DEFAULT_QUERIES[0],
            "expected_terms": ["alpha-17", "warning"],
            "text_only": [],
            "vl_text_image": [],
            "text_only_top_contains_expected": True,
            "vl_top_contains_expected": False,
            "top_result_changed": True,
        }
    ]

    def fake_compare_query_results(queries, rows_df, **kwargs):
        calls.append((queries, rows_df.to_dict(orient="records"), kwargs))
        return query_payload

    monkeypatch.setattr(probe, "compare_query_results", fake_compare_query_results)

    assert probe.main(["--artifact-dir", str(tmp_path)]) == 0

    query_results_path = tmp_path / "query_results.json"
    query_results = json.loads(query_results_path.read_text(encoding="utf-8"))
    assert query_results == query_payload
    observations_path = tmp_path / "observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    assert observations["metrics"] == {
        "omni_elapsed_seconds": 1.0,
        "fixture_bytes": len(b"cached fixture"),
        "text_only_expected_hits": 1,
        "vl_expected_hits": 0,
        "top_result_changes": 1,
        "query_count": 1,
    }
    assert calls[0][2] == {
        "api_key": "embed-token",
        "endpoint": config.embed_endpoint,
        "model_name": config.embed_model,
        "text_model_name": config.text_embed_model,
        "top_k": config.top_k,
        "allow_custom_endpoint": False,
    }
    assert calls[0][1][0]["segment_id"] == "scene-alpha"


def test_main_refreshes_stale_cached_omni_response(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"new fixture")
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    stale_request = probe.omni_request_metadata(fixture_path, config) | {"model": "old-model"}
    cached_response = {
        "elapsed_seconds": 1.0,
        "request": stale_request,
        "raw_response": {},
        "parsed": {"segments": [{"segment_id": "cached"}]},
    }
    fresh_response = {
        "elapsed_seconds": 2.0,
        "request": probe.omni_request_metadata(fixture_path, config),
        "raw_response": {},
        "parsed": {"segments": [{"segment_id": "fresh"}]},
    }
    (tmp_path / "omni_response.json").write_text(json.dumps(cached_response), encoding="utf-8")
    calls = []

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (fixture_path, {}))
    monkeypatch.setenv(probe.API_KEY_ENV, "token")
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: calls.append((fixture_path, token)) or fresh_response)
    monkeypatch.setattr(probe, "compare_query_results", lambda queries, rows_df, **kwargs: [])

    assert probe.main(["--artifact-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert calls == [(fixture_path, "token")]
    assert f"Cached Omni response is stale; refreshing: {tmp_path / 'omni_response.json'}" in output
    assert json.loads((tmp_path / "omni_response.json").read_text(encoding="utf-8"))["parsed"]["segments"][0]["segment_id"] == "fresh"


def test_main_refreshes_corrupt_cached_omni_response(tmp_path, monkeypatch, capsys):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"new fixture")
    fresh_response = {
        "elapsed_seconds": 2.0,
        "request": probe.omni_request_metadata(fixture_path, probe.parse_args(["--artifact-dir", str(tmp_path)])),
        "raw_response": {},
        "parsed": {"segments": [{"segment_id": "fresh"}]},
    }
    (tmp_path / "omni_response.json").write_text("{not valid json", encoding="utf-8")
    calls = []

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (fixture_path, {}))
    monkeypatch.setenv(probe.API_KEY_ENV, "token")
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: calls.append((fixture_path, token)) or fresh_response)
    monkeypatch.setattr(probe, "compare_query_results", lambda queries, rows_df, **kwargs: [])

    assert probe.main(["--artifact-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert calls == [(fixture_path, "token")]
    assert f"Cached Omni response is unreadable; refreshing: {tmp_path / 'omni_response.json'}" in output


def test_main_force_fixture_bypasses_existing_omni_cache(tmp_path, monkeypatch):
    probe = load_probe_module()
    fixture_path = tmp_path / "synthetic_fixture.mp4"
    fixture_path.write_bytes(b"cached fixture")
    config = probe.parse_args(["--artifact-dir", str(tmp_path)])
    cached_response = {
        "elapsed_seconds": 1.0,
        "request": probe.omni_request_metadata(fixture_path, config),
        "raw_response": {},
        "parsed": {"segments": [{"segment_id": "cached"}]},
    }
    fresh_response = cached_response | {"parsed": {"segments": [{"segment_id": "fresh"}]}}
    (tmp_path / "omni_response.json").write_text(json.dumps(cached_response), encoding="utf-8")
    calls = []

    monkeypatch.setattr(probe, "generate_synthetic_fixture", lambda config: (fixture_path, {}))
    monkeypatch.setenv(probe.API_KEY_ENV, "token")
    monkeypatch.setattr(probe, "call_nano_omni", lambda fixture_path, config, token: calls.append((fixture_path, token)) or fresh_response)
    monkeypatch.setattr(probe, "compare_query_results", lambda queries, rows_df, **kwargs: [])

    assert probe.main(["--artifact-dir", str(tmp_path), "--force-fixture"]) == 0

    assert calls == [(fixture_path, "token")]
