"""Bounded CPU contracts. These tests do not establish physical GPU placement."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
import pickle
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from nemo_retriever.graph import gpu_placement as placement


def specs():
    return [
        {"name": name, "concurrency": len(bundles), "cpu": 1.0, "gpu": 0.25, "remote": False}
        for name, bundles in placement.ROLE_BUNDLES.items()
    ]


def records(owner="owner", group="abc"):
    rows = []
    for role, bundles in placement.ROLE_BUNDLES.items():
        for slot, bundle in enumerate(bundles):
            rows.append(
                {
                    "owner": owner,
                    "group_id": group,
                    "role": role,
                    "slot": slot,
                    "bundle": bundle,
                    "state": "ready",
                    "GPU_UUID": f"GPU-{bundle}",
                    "pid": 9000000 + len(rows),
                    "process_start_ticks": 1,
                    "actor_name": f"nrl-{owner}-{role}-{slot}",
                    "actor_id": f"a{len(rows):03x}",
                    "worker_id": f"b{len(rows):03x}",
                    "node_id": "decade",
                    "job_id": "face",
                    "accelerator_ids": [str(bundle)],
                    "CUDA_VISIBLE_DEVICES": str(bundle),
                    "model_devices": ["cuda:0"],
                }
            )
    return rows


def test_fixed_role_map_does_not_instantiate_table_structure():
    result = placement.validate_role_specs(specs())
    assert result["actor_resources"] == {"CPU": 14.0, "GPU": 3.5}
    assert result["reserved_resources"] == {"CPU": 16.0, "GPU": 4.0}
    assert len(result["bundles"]) == 4
    assert "TableStructureActor" not in result["roles"]
    assert result["roles"]["_BatchEmbedActor"]["bundles"] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize(
    "change",
    [
        {"concurrency": 10},
        {"concurrency": (1, 8)},
        {"concurrency": True},
        {"cpu": 2.0},
        {"cpu": True},
        {"cpu": float("nan")},
        {"gpu": 0},
        {"gpu": 0.5},
        {"gpu": float("inf")},
        {"remote": True},
    ],
)
def test_preflight_rejects_unsupported_resources_and_elastic_roles(change):
    values = specs()
    values[-1].update(change)
    with pytest.raises(ValueError):
        placement.validate_role_specs(values)


@pytest.mark.parametrize("values", [specs()[:-1], specs() + [{"name": "TableStructureActor"}], specs() * 2])
def test_preflight_rejects_missing_extra_and_duplicate_gpu_roles(values):
    with pytest.raises(ValueError):
        placement.validate_role_specs(values)


@pytest.mark.parametrize(
    "policy,seconds",
    [
        ("unknown", 60),
        ("legacy", 1),
        ("balanced_four_gpu", 0),
        ("balanced_four_gpu", float("nan")),
        ("balanced_four_gpu", float("inf")),
        ("balanced_four_gpu", True),
    ],
)
def test_typed_policy_rejects_ignored_or_unbounded_settings(policy, seconds):
    with pytest.raises(ValueError):
        placement.validate_policy(policy, seconds)


def test_outside_cpu_budget_excludes_all_sixteen_reserved_bundle_cpus():
    from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=224, gpu_count=4), available_resources=Resources(cpu_count=224, gpu_count=4)
    )
    outside = placement.outside_resources(cluster)
    assert outside.available_cpu_count() == outside.total_cpu_count() == 208
    assert outside.available_gpu_count() == 4  # Class admission retains the original GPU lease.
    assert cluster.available_cpu_count() == 224


@pytest.fixture
def graph_preflight(monkeypatch):
    from nemo_retriever.common import ray_runtime
    from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
    from nemo_retriever.graph import operator_resolution
    from nemo_retriever.operators.gpu_operator import GPUOperator

    monkeypatch.setattr(ray_runtime, "require_owned_local_ray_runtime", lambda ray: None)
    monkeypatch.setattr(placement, "_visible_gpu_uuids", lambda: [f"GPU-{i}" for i in range(4)])
    monkeypatch.setattr(operator_resolution, "resolve_graph", lambda graph, resources: graph)
    context = SimpleNamespace(
        namespace="owned", get_job_id=lambda: "aaa", get_node_id=lambda: "bbb", get_worker_id=lambda: "ccc"
    )
    ray = SimpleNamespace(
        get_runtime_context=lambda: context,
        nodes=lambda: [{"Alive": True, "NodeID": "bbb"}],
        cluster_resources=lambda: {"CPU": 224, "GPU": 4},
        available_resources=lambda: {"CPU": 224, "GPU": 4},
    )
    resources = ClusterResources(
        total_resources=Resources(cpu_count=224, gpu_count=4), available_resources=Resources(cpu_count=224, gpu_count=4)
    )

    def executor(roles, source_cpu):
        nodes, overrides = [], {}
        for role in roles:
            gpu = role in placement.ROLE_BUNDLES
            nodes.append(SimpleNamespace(name=role, operator_class=type(role, (GPUOperator,) if gpu else (), {})))
            overrides[role] = {
                "concurrency": len(placement.ROLE_BUNDLES[role]) if gpu else 64,
                "num_cpus": 1,
                "num_gpus": 0.25 if gpu else 0,
            }
        return SimpleNamespace(
            graph=nodes,
            _linearize=lambda graph: graph,
            _stream_ingest_index=lambda nodes: None,
            _node_overrides=overrides,
            _default_num_cpus=1,
            _source_cpu_reservation=source_cpu,
            _auto_concurrency_nodes=set(),
            _has_remote_endpoint=lambda node: False,
            _scheduled_num_gpus=lambda node, options, available: options.get("num_gpus", 0),
        )

    extraction = executor(["PDFExtractionCPUActor", "PageElementDetectionActor", "OCRActor"], 1)
    post = executor(["_BatchEmbedActor"], 0)
    return ray, resources, extraction, post


def test_shared_extraction_and_post_graphs_admit_once_before_group_creation(graph_preflight):
    ray, resources, extraction, post = graph_preflight
    owner = placement.BalancedPlacement(ray, [extraction, post], resources)
    try:
        assert owner.pg is None and owner.token not in placement._OWNERS
        assert extraction._placement is post._placement is owner
        assert owner.receipt["outside_CPU_budget"] == 208
        assert owner.receipt["source_read_CPU"] == 1
        assert owner.receipt["outside_CPU_actors"][0]["concurrency"] == 64
        assert owner.receipt["creator_job_id"] == "aaa"
        assert owner.graphs[id(extraction)] is extraction.graph
    finally:
        owner.directory.rmdir()


@pytest.mark.parametrize("change", [{"concurrency": 10}, {"concurrency": (1, 8)}, {"num_gpus": 0.5}])
def test_actual_shared_graph_rejects_unsupported_actor_resources_before_group(graph_preflight, change):
    ray, resources, extraction, post = graph_preflight
    post._node_overrides["_BatchEmbedActor"].update(change)
    with pytest.raises(ValueError):
        placement.BalancedPlacement(ray, [extraction, post], resources)


def test_legacy_runtime_does_not_read_or_record_new_owner_metadata(monkeypatch):
    from nemo_retriever.common import ray_runtime

    calls = []
    ray = SimpleNamespace(is_initialized=lambda: False, init=lambda **kw: calls.append(kw))
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(ray_runtime, "build_local_ray_runtime_env", lambda: {})
    monkeypatch.setattr(ray_runtime, "_LOCAL_RUNTIME_OWNER", None)
    assert ray_runtime.ensure_local_ray_runtime() is ray
    assert calls == [{"address": None, "ignore_reinit_error": True, "runtime_env": {}}]
    assert ray_runtime._LOCAL_RUNTIME_OWNER is None


@pytest.fixture
def owner(tmp_path, monkeypatch):
    value = placement.BalancedPlacement.__new__(placement.BalancedPlacement)
    value.token = "owner"
    value.pid = os.getpid()
    value.lock = threading.Lock()
    value.cursors = {}
    value.registered = set()
    value.issued = []
    value.pg = SimpleNamespace(id="abc", ready=lambda: "ready")
    value.namespace = "test-owned"
    value.timeout = 0.01
    value.timer_stop = threading.Event()
    value.timer = None
    value.startup_error = None
    value.directory = tmp_path / "owned"
    value.directory.mkdir(mode=0o700)
    value.allowed_uuids = [f"GPU-{i}" for i in range(4)]
    value.receipt = {"schema_version": 1, "policy": "balanced_four_gpu", "terminal_iterator_closed": True}
    removed, killed = [], []

    def absent(name, **kwargs):
        raise ValueError(name)

    value.ray = SimpleNamespace(
        get_actor=absent, kill=lambda actor, **kw: killed.append(actor), available_resources=lambda: {"GPU": 4}
    )
    group_module = ModuleType("ray.util.placement_group")
    group_module.placement_group_table = lambda group: {"state": "REMOVED"}
    group_module.remove_placement_group = removed.append
    group_module.placement_group = lambda *args, **kwargs: value.pg
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", group_module)
    strategy = ModuleType("ray.util.scheduling_strategies")
    strategy.PlacementGroupSchedulingStrategy = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategy)
    monkeypatch.setattr(placement, "_process_start_ticks", lambda pid: None)
    monkeypatch.setitem(placement._OWNERS, value.token, value)
    value.test_removed, value.test_killed = removed, killed
    return value


def test_slot_callback_serialization_keeps_one_owner_cursor_and_no_lock(owner):
    callback = owner.register("PageElementDetectionActor")
    copied = pickle.loads(pickle.dumps(callback))
    first, second = callback(), copied()
    assert first["name"] != second["name"]
    assert second["scheduling_strategy"].placement_group_bundle_index == 1
    assert set(first) == {"name", "scheduling_strategy"}  # Runtime env and resource kwargs are preserved.
    with pytest.raises(RuntimeError, match="overflow"):
        copied()
    with pytest.raises(RuntimeError, match="reused"):
        owner.register("PageElementDetectionActor")


def test_parallel_callback_has_exactly_eight_noncycling_embedding_slots(owner):
    callback = owner.register("_BatchEmbedActor")
    with ThreadPoolExecutor(max_workers=4) as pool:
        options = list(pool.map(lambda _: callback(), range(8)))
    assert len({option["name"] for option in options}) == 8
    assert sorted(option["scheduling_strategy"].placement_group_bundle_index for option in options) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    with pytest.raises(RuntimeError):
        callback()


def test_callback_rejects_foreign_process_and_closed_generation(owner):
    callback = owner.register("OCRActor")
    with pytest.raises(RuntimeError, match="owner/generation"):
        placement.SlotCallback(owner.token, "OCRActor", -1)()
    placement._OWNERS.pop(owner.token)
    with pytest.raises(RuntimeError, match="owner/generation"):
        callback()


def test_runtime_requires_distinct_physical_uuid_per_logical_bundle():
    valid = records()
    placement.validate_runtime_records(valid, "owner", "abc", [f"GPU-{i}" for i in range(4)])
    invalid = deepcopy(valid)
    for row in invalid:
        row["GPU_UUID"] = "GPU-0"
    with pytest.raises(ValueError, match="distinct physical"):
        placement.validate_runtime_records(invalid, "owner", "abc", [f"GPU-{i}" for i in range(4)])


@pytest.mark.parametrize(
    "change",
    [
        {"owner": "foreign"},
        {"group_id": "foreign"},
        {"state": "failed"},
        {"slot": 99},
        {"bundle": 3},
        {"GPU_UUID": "off-lease"},
        {"model_devices": ["cpu"]},
        {"model_devices": []},
        {"process_start_ticks": None},
    ],
)
def test_runtime_rejects_incomplete_or_foreign_evidence(change):
    values = records()
    values[0].update(change)
    with pytest.raises(ValueError):
        placement.validate_runtime_records(values, "owner", "abc", [f"GPU-{i}" for i in range(4)])


def write_records(owner):
    values = records()
    owner.issued = [row["actor_name"] for row in values]
    for index, row in enumerate(values):
        placement._save(owner.directory / f"{index}.yaml", row)


def test_success_requires_closed_iterator_physical_exit_and_gpu_release(owner):
    write_records(owner)
    owner.__exit__(None, None, None)
    assert owner.receipt["status"] == "passed"
    assert owner.receipt["cleanup"] == {
        "status": "passed",
        "forced_actor_names": [],
        "errors": [],
        "owned_group_removed": True,
        "actor_processes_exited": True,
        "GPU_resources_released": True,
    }
    assert owner.test_removed == [owner.pg]
    assert owner.test_killed == []
    assert not owner.directory.exists() and owner.token not in placement._OWNERS


def test_missing_terminal_iterator_close_is_not_a_qualified_success(owner):
    write_records(owner)
    owner.receipt["terminal_iterator_closed"] = False
    with pytest.raises(RuntimeError, match="iterator"):
        owner.__exit__(None, None, None)
    assert owner.test_removed == [owner.pg]
    assert owner.receipt["status"] == "failed"


def test_later_ingest_failure_remains_failed_even_if_all_actors_were_ready(owner):
    write_records(owner)
    owner.__exit__(RuntimeError, RuntimeError("task failed"), None)
    assert owner.receipt["status"] == "failed"
    assert owner.receipt["cleanup"]["status"] == "passed"


def test_missing_actor_evidence_still_removes_only_owned_group(owner):
    owner.issued = [owner.actor_names("OCRActor")[0]]
    with pytest.raises(RuntimeError):
        owner.__exit__(None, None, None)
    assert owner.test_removed == [owner.pg]
    assert owner.receipt["cleanup"]["actor_processes_exited"] is False


def test_group_removed_alone_does_not_prove_actor_process_exit(owner, monkeypatch):
    write_records(owner)
    monkeypatch.setattr(placement, "_process_start_ticks", lambda pid: 1)
    with pytest.raises(RuntimeError, match="physical process exit"):
        owner.__exit__(None, None, None)
    assert owner.receipt["cleanup"]["owned_group_removed"] is True
    assert owner.receipt["cleanup"]["actor_processes_exited"] is False


def test_group_removed_without_returned_gpu_resources_fails_cleanup(owner):
    write_records(owner)
    owner.ray.available_resources = lambda: {"GPU": 0}
    with pytest.raises(RuntimeError, match="resources"):
        owner.__exit__(None, None, None)
    assert owner.receipt["cleanup"]["GPU_resources_released"] is False


def test_only_generated_owned_names_can_be_forcibly_stopped(owner, monkeypatch):
    write_records(owner)
    monkeypatch.setattr(placement, "_process_start_ticks", lambda pid: 1)
    looked_up = []

    def present(name, **kwargs):
        looked_up.append(name)
        return name

    owner.ray.get_actor = present
    with pytest.raises(RuntimeError, match="forced"):
        owner.__exit__(None, None, None)
    assert looked_up == owner.issued
    assert owner.test_killed == owner.issued
    assert owner.receipt["cleanup"]["status"] == "failed"


def test_driver_job_and_node_identity_are_bound_to_actor_evidence():
    values = records()
    with pytest.raises(ValueError):
        placement.validate_runtime_records(
            values, "owner", "abc", [f"GPU-{i}" for i in range(4)], creator_node_id="different", creator_job_id="face"
        )


def test_failure_receipt_survives_exception_serialization():
    error = placement.GpuPlacementError("failed", {"status": "failed"})
    restored = pickle.loads(pickle.dumps(error))
    assert str(restored) == "failed"
    assert restored.gpu_placement_receipt == {"status": "failed"}


@pytest.fixture
def actor_context(tmp_path, monkeypatch):
    gpu = "GPU-11111111-1111-1111-1111-111111111111"
    context = SimpleNamespace(
        get_actor_name=lambda: "nrl-owner-_BatchEmbedActor-0",
        get_placement_group_id=lambda: "abc",
        get_accelerator_ids=lambda: {"GPU": ["0"]},
        get_actor_id=lambda: "aaa",
        get_node_id=lambda: "bbb",
        get_job_id=lambda: "ccc",
        get_worker_id=lambda: "ddd",
    )
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get_runtime_context=lambda: context))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(uuid=gpu))),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return {
        "owner": "owner",
        "role": "_BatchEmbedActor",
        "names": [context.get_actor_name()],
        "group_id": "abc",
        "allowed_GPU_UUIDs": [gpu],
        "bundles": [0],
        "directory": str(tmp_path),
    }


def test_lazy_hf_preserves_normal_constructor_and_batch_without_extra_load(actor_context):
    calls = []
    batch, output = object(), object()

    class Lazy:
        def __init__(self, token):
            calls.append(("construct", token))
            self._model = None

        def __call__(self, value):
            calls.append(("batch", value))
            self._model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device="cuda:0")])
            return output

    adapter = placement.PlacementOperatorAdapter(Lazy, {"token": "unchanged"}, actor_context)
    assert calls == [("construct", "unchanged")]
    assert adapter._placement_record["state"] == "constructed"
    assert adapter(batch) is output
    assert calls == [("construct", "unchanged"), ("batch", batch)]
    assert adapter._placement_record["state"] == "ready"
    assert adapter._placement_record["model_devices"] == ["cuda:0"]


def test_lazy_hf_cpu_fallback_is_failed_after_existing_batch(actor_context):
    class CPU:
        def __init__(self):
            self._model = None

        def __call__(self, batch):
            self._model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device="cpu")])
            return batch

    adapter = placement.PlacementOperatorAdapter(CPU, {}, actor_context)
    with pytest.raises(RuntimeError, match="CUDA"):
        adapter(object())
    assert adapter._placement_record["state"] == "failed"


def test_materialization_closes_iterator_before_notifying_owner_on_failure(monkeypatch):
    from nemo_retriever.graph import executor

    events = []

    def batches(**kwargs):
        try:
            yield object()
            raise RuntimeError("read failed")
        finally:
            events.append("closed")

    monkeypatch.setattr(executor, "arrow_table_to_pandas", lambda batch: batch)
    with pytest.raises(RuntimeError, match="read failed"):
        executor.ray_dataset_to_pandas(
            SimpleNamespace(iter_batches=batches), on_iteration_closed=lambda: events.append("owner")
        )
    assert events == ["closed", "owner"]


@pytest.mark.parametrize("failure", [ValueError("task failure"), KeyboardInterrupt()])
def test_graph_context_transports_terminal_failure_after_cleanup(monkeypatch, failure):
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    events = []
    receipt = {"status": "running"}

    class Owner:
        def __init__(self, *args, **kwargs):
            self.receipt = receipt

        def __enter__(self):
            return self

        def __exit__(self, *args):
            receipt.update(status="failed", cleanup={"status": "passed"})
            events.append("cleanup")

    monkeypatch.setattr(placement, "BalancedPlacement", Owner)
    ingestor = GraphIngestor.__new__(GraphIngestor)
    ingestor._gpu_placement = "balanced_four_gpu"
    ingestor._gpu_placement_ready_timeout_s = 1
    expected = placement.GpuPlacementError if isinstance(failure, Exception) else type(failure)
    with pytest.raises(expected) as caught:
        with ingestor._placement_context(None, [], None):
            raise failure
    assert events == ["cleanup"]
    if isinstance(failure, Exception):
        assert caught.value.__cause__ is failure
    else:
        assert caught.value is failure  # Preserve cancellation/control-flow semantics.
    assert caught.value.gpu_placement_receipt == receipt
    assert ingestor.gpu_placement_receipt == receipt
    ingestor.gpu_placement_receipt.clear()
    assert ingestor.gpu_placement_receipt == receipt  # Callers cannot mutate owner evidence.


def test_factory_defaults_omit_placement_and_opt_in_forwards_exact_typed_values(monkeypatch):
    from nemo_retriever.ingestor import core
    from nemo_retriever.ingestor import graph_ingestor

    monkeypatch.setattr(graph_ingestor, "GraphIngestor", lambda **kwargs: kwargs)
    legacy = core.create_ingestor(run_mode="batch")
    assert "gpu_placement" not in legacy and "gpu_placement_ready_timeout_s" not in legacy
    enabled = core.create_ingestor(
        run_mode="batch", gpu_placement="balanced_four_gpu", gpu_placement_ready_timeout_s=12.5
    )
    assert enabled["gpu_placement"] == "balanced_four_gpu"
    assert enabled["gpu_placement_ready_timeout_s"] == 12.5
    with pytest.raises(ValueError):
        core.create_ingestor(run_mode="service", gpu_placement="balanced_four_gpu")
    with pytest.raises(ValueError):
        core.create_ingestor(run_mode="batch", gpu_placement="balanced_four_gpu", gpu_placement_ready_timeout_s=True)


def test_workflow_summary_has_enabled_receipt_and_legacy_keys_are_unchanged():
    from nemo_retriever.ingest.execution import IngestExecutionResult

    args = dict(
        plan=SimpleNamespace(documents=[]),
        result=None,
        n_rows=1,
        result_n_rows=1,
        initial_n_rows=0,
        lancedb_uri="private",
        table_name="test",
    )
    legacy = IngestExecutionResult(**args, run_metadata={}).to_summary_dict()
    enabled = IngestExecutionResult(**args, run_metadata={"gpu_placement": {"status": "passed"}}).to_summary_dict()
    assert enabled == {**legacy, "gpu_placement": {"status": "passed"}}


def test_ready_timeout_cleans_only_group_created_by_this_owner(owner):
    owner.plan = placement.validate_role_specs(specs())
    owner.ray.get = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("not ready"))
    with pytest.raises(TimeoutError):
        owner.__enter__()
    assert owner.test_removed == [owner.pg]
    assert owner.receipt["status"] == "failed"
    assert owner.token not in placement._OWNERS
