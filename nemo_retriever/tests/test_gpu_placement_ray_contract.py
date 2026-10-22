# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real Ray CPU software contract; no CUDA, models, or corpus data.

Run only in an isolated owned runtime with NRL_RUN_PLACEMENT_RAY_CPU_CONTRACT=1.
The CPU-only placement group tests callbacks and claims, not physical GPU balance.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

import pytest

from nemo_retriever.graph import gpu_placement as placement

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NRL_RUN_PLACEMENT_RAY_CPU_CONTRACT") != "1",
        reason="requires explicit isolated CPU Ray contract window",
    ),
]


@pytest.fixture(scope="module")
def ray_cpu():
    import psutil
    import ray

    assert not ray.is_initialized(), "Never replace an existing runtime"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    # Keep Ray Unix-domain socket paths below the kernel length limit.
    root = Path(tempfile.mkdtemp(prefix="nrb-prc-", dir="/tmp"))
    root.chmod(0o700)
    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            namespace="placement-cpu-" + uuid.uuid4().hex,
            _node_ip_address="127.0.0.1",
            _temp_dir=str(root),
            object_store_memory=80 * 1024**2,
            log_to_driver=False,
            runtime_env={
                "py_executable": sys.executable,
                "env_vars": {"PYTHONPATH": os.pathsep.join(sys.path), "CUDA_VISIBLE_DEVICES": ""},
            },
        )
        assert ray.cluster_resources().get("GPU", 0) == 0
        yield ray
    finally:
        children = [
            (child.pid, placement._process_start_ticks(child.pid))
            for child in psutil.Process().children(recursive=True)
        ]
        ray.shutdown()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(ticks is None or placement._process_start_ticks(pid) != ticks for pid, ticks in children):
                break
            time.sleep(0.02)
        assert all(ticks is None or placement._process_start_ticks(pid) != ticks for pid, ticks in children)
        assert root.parent == Path("/tmp") and root.name.startswith("nrb-prc-") and root.stat().st_uid == os.getuid()
        shutil.rmtree(root)


def owner_for(ray, directory, monkeypatch):
    from ray.util.placement_group import placement_group

    owner = placement.BalancedPlacement.__new__(placement.BalancedPlacement)
    owner.token = uuid.uuid4().hex
    owner.pid = os.getpid()
    owner.ray = ray
    owner.lock = threading.Lock()
    owner.observed_names = set()
    owner.registered = set()
    owner.issued = []
    owner.startup_error = None
    owner.directory = directory
    owner.namespace = ray.get_runtime_context().namespace
    owner.pg = placement_group(
        [{"CPU": 0.5}, {"CPU": 0.5}], strategy="STRICT_PACK", name="placement-cpu-" + owner.token
    )
    ray.get(owner.pg.ready(), timeout=15)
    monkeypatch.setitem(placement._OWNERS, owner.token, owner)
    return owner


def cleanup_owner(ray, owner):
    from ray.util.placement_group import placement_group_table, remove_placement_group

    for name in owner.actor_names("PageElementDetectionActor"):
        try:
            actor = ray.get_actor(name, namespace=owner.namespace)
        except ValueError:
            continue
        ray.kill(actor, no_restart=True)
        del actor
    remove_placement_group(owner.pg)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if placement_group_table(owner.pg).get("state") == "REMOVED":
            return
        time.sleep(0.02)
    raise AssertionError("Owned CPU placement group did not disappear")


def wait_exited(record):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if placement._process_start_ticks(record["pid"]) != record["process_start_ticks"]:
            return
        time.sleep(0.02)
    raise AssertionError("Owned CPU actor process did not exit")


def wait_name_absent(ray, name, namespace):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            ray.get_actor(name, namespace=namespace)
        except ValueError:
            return
        time.sleep(0.02)
    raise AssertionError("Owned dead actor name did not disappear")


def test_real_named_registration_introspection_and_dead_slot_recreation(ray_cpu, tmp_path, monkeypatch):
    ray = ray_cpu
    owner = owner_for(ray, tmp_path, monkeypatch)
    callback = owner.register("PageElementDetectionActor")

    class ClaimedActor:
        def __init__(self, directory, token):
            import ray
            from pathlib import Path
            from nemo_retriever.graph.gpu_placement import _claim_actor_slot, _process_start_ticks

            context = ray.get_runtime_context()
            name = context.get_actor_name()
            slot = int(name.rsplit("-", 1)[1])
            # Driver releases this gate only after checking the next callback.
            # Registration must therefore be visible before any claim exists.
            deadline = time.monotonic() + 10
            while not (Path(directory) / f"release-{slot}.yaml").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned CPU constructor gate timed out")
                time.sleep(0.01)
            self.record = {
                "owner": token,
                "role": "PageElementDetectionActor",
                "slot": slot,
                "actor_name": name,
                "actor_id": context.get_actor_id(),
                "pid": os.getpid(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
            }
            _claim_actor_slot(Path(directory), self.record)

        def info(self):
            return self.record

    actor_class = ray.remote(ClaimedActor)
    handles = []
    try:
        first = callback()
        assert all(callback()["name"] == first["name"] for _ in range(20))
        for slot in range(2):
            options = callback()
            assert options["name"] == owner.actor_names("PageElementDetectionActor")[slot]
            handles.append(
                actor_class.options(num_cpus=0.25, max_restarts=0, **options).remote(str(tmp_path), owner.token)
            )
            expected = owner.actor_names("PageElementDetectionActor")[min(slot + 1, 1)]
            assert all(callback()["name"] == expected for _ in range(20))
            assert not (tmp_path / "claims" / f"PageElementDetectionActor-{slot}").exists()
            gate = tmp_path / f"release-{slot}.yaml"
            gate.write_text("released: true\n")
            gate.chmod(0o600)
        records = ray.get([actor.info.remote() for actor in handles], timeout=15)
        assert len({record["actor_id"] for record in records}) == 2
        with pytest.raises(ray.exceptions.ActorAlreadyExistsError):
            actor_class.options(num_cpus=0.25, max_restarts=0, **callback()).remote(str(tmp_path), owner.token)
        ray.kill(handles[-1], no_restart=True)
        handles.pop()
        wait_exited(records[-1])
        wait_name_absent(ray, records[-1]["actor_name"], owner.namespace)
        assert callback()["name"] == records[-1]["actor_name"]
        replacement = actor_class.options(num_cpus=0.25, max_restarts=0, **callback()).remote(
            str(tmp_path), owner.token
        )
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(replacement.info.remote(), timeout=15)
        assert len(list((tmp_path / "claims").glob("*/claim.yaml"))) == 2
        del replacement
    finally:
        handles.clear()
        cleanup_owner(ray, owner)


def test_real_ray_data_default_issue_detectors_and_normal_actor_drain(ray_cpu, tmp_path, monkeypatch):
    ray = ray_cpu
    owner = owner_for(ray, tmp_path, monkeypatch)
    callback = owner.register("PageElementDetectionActor")

    class ClaimedBatch:
        def __init__(self, directory, token):
            import ray
            from pathlib import Path
            from nemo_retriever.graph.gpu_placement import _claim_actor_slot, _process_start_ticks

            context = ray.get_runtime_context()
            name = context.get_actor_name()
            self.record = {
                "owner": token,
                "role": "PageElementDetectionActor",
                "slot": int(name.rsplit("-", 1)[1]),
                "actor_name": name,
                "actor_id": context.get_actor_id(),
                "pid": os.getpid(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
            }
            _claim_actor_slot(Path(directory), self.record)
            self.directory = Path(directory)

        def __call__(self, batch):
            # A tiny dataset can finish on the first ready actor before Ray
            # starts the other pending fixed-pool worker. Require both normal
            # constructors to run before allowing this test's first result.
            deadline = time.monotonic() + 15
            while len(list((self.directory / "claims").glob("*/claim.yaml"))) != 2:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Both owned CPU pool actors did not initialize")
                time.sleep(0.01)
            return batch

    try:
        import yaml
        from ray.data import DataContext

        context = DataContext.get_current()
        assert context.issue_detectors_config.high_memory_detector_config.detection_time_interval_s >= 0
        assert any("HighMemory" in cls.__name__ for cls in context.issue_detectors_config.detectors)
        assert len({callback()["name"] for _ in range(20)}) == 1
        dataset = ray.data.from_items([{"value": i} for i in range(8)], override_num_blocks=4).map_batches(
            ClaimedBatch,
            batch_size=1,
            batch_format="pyarrow",
            concurrency=2,
            num_cpus=0.25,
            fn_constructor_kwargs={"directory": str(tmp_path), "token": owner.token},
            ray_remote_args_fn=callback,
            max_restarts=0,
        )
        assert sorted(row["value"] for row in dataset.take_all()) == list(range(8))
        records = [yaml.safe_load(path.read_text()) for path in (tmp_path / "claims").glob("*/claim.yaml")]
        assert len(records) == 2 and len({record["actor_id"] for record in records}) == 2
        assert all(callback()["name"] == owner.actor_names("PageElementDetectionActor")[-1] for _ in range(20))
        for record in records:
            wait_exited(record)
    finally:
        cleanup_owner(ray, owner)


def test_real_concurrent_creation_ambiguity_fails_closed(ray_cpu, tmp_path, monkeypatch):
    ray = ray_cpu
    owner = owner_for(ray, tmp_path, monkeypatch)
    callback = owner.register("PageElementDetectionActor")
    options = callback()
    barrier = threading.Barrier(2)

    class ClaimOnce:
        def __init__(self, directory, token):
            import ray
            from pathlib import Path
            from nemo_retriever.graph.gpu_placement import _claim_actor_slot, _process_start_ticks

            context = ray.get_runtime_context()
            self.record = {
                "owner": token,
                "role": "PageElementDetectionActor",
                "slot": 0,
                "actor_name": context.get_actor_name(),
                "actor_id": context.get_actor_id(),
                "pid": os.getpid(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
            }
            _claim_actor_slot(Path(directory), self.record)

        def info(self):
            return self.record

    actor_class = ray.remote(ClaimOnce)

    def create(_):
        barrier.wait(timeout=5)
        try:
            return actor_class.options(num_cpus=0.25, max_restarts=0, **options).remote(str(tmp_path), owner.token)
        except ray.exceptions.ActorAlreadyExistsError as error:
            return error

    handles = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(pool.map(create, range(2)))
        successes, failures = [], 0
        for attempt in attempts:
            if isinstance(attempt, Exception):
                failures += 1
                continue
            handles.append(attempt)
            try:
                successes.append(ray.get(attempt.info.remote(), timeout=15))
            except ray.exceptions.RayActorError:
                failures += 1
        assert len(successes) == failures == 1
        assert len(list((tmp_path / "claims").glob("*/claim.yaml"))) == 1
    finally:
        handles.clear()
        cleanup_owner(ray, owner)
