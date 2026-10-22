# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Owned, opt-in placement for one fixed local four-GPU ingestion graph.

The slot map is a scheduling request. Only runtime UUID observations establish
physical placement. No Ray, CUDA, or model import occurs on the legacy path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Literal
import uuid

GpuPlacement = Literal["legacy", "balanced_four_gpu"]
ROLE_BUNDLES = {
    "PageElementDetectionActor": (0, 1),
    "OCRActor": (0, 1, 2, 3),
    "_BatchEmbedActor": (0, 0, 1, 1, 2, 2, 3, 3),
}
_OWNERS: dict[str, Any] = {}


class GpuPlacementError(RuntimeError):
    """Enabled-policy failure with terminal ownership/cleanup evidence."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.gpu_placement_receipt = receipt

    def __reduce__(self) -> tuple[Any, tuple[str, dict[str, Any]]]:
        return type(self), (str(self), self.gpu_placement_receipt)


def validate_policy(policy: str, ready_timeout_s: float) -> None:
    if not isinstance(policy, str) or policy not in {"legacy", "balanced_four_gpu"}:
        raise ValueError("gpu_placement must be 'legacy' or 'balanced_four_gpu'")
    if isinstance(ready_timeout_s, bool) or not isinstance(ready_timeout_s, (int, float)):
        raise ValueError("gpu_placement_ready_timeout_s must be finite and positive")
    if not math.isfinite(ready_timeout_s) or ready_timeout_s <= 0:
        raise ValueError("gpu_placement_ready_timeout_s must be finite and positive")
    if policy == "legacy" and ready_timeout_s != 60.0:
        raise ValueError("gpu_placement_ready_timeout_s requires balanced_four_gpu")


def outside_resources(resources: Any) -> Any:
    """Deduct the entire bundle CPU reservation before source/PDF auto sizing."""
    from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources

    if resources.available_gpu_count() != 4 or resources.total_gpu_count() != 4:
        raise ValueError("balanced_four_gpu requires exactly four owned visible Ray GPUs")
    if resources.available_cpu_count() <= 16:
        raise ValueError("balanced_four_gpu requires CPU capacity outside its 16 reserved CPUs")
    return ClusterResources(
        total_resources=Resources(cpu_count=resources.total_cpu_count() - 16, gpu_count=4),
        available_resources=Resources(cpu_count=resources.available_cpu_count() - 16, gpu_count=4),
    )


def validate_role_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate actual graph roles, never configured-but-absent TS workers."""
    if len(specs) != 3 or {s["name"] for s in specs} != set(ROLE_BUNDLES):
        raise ValueError("balanced_four_gpu requires exactly PE2/OCR4/embed8 with table structure disabled")
    roles = {}
    for spec in specs:
        name = spec["name"]
        count = spec["concurrency"]
        if type(count) is not int or count != len(ROLE_BUNDLES[name]):
            raise ValueError(f"balanced_four_gpu requires fixed {name} concurrency {len(ROLE_BUNDLES[name])}")
        if (
            isinstance(spec["cpu"], bool)
            or isinstance(spec["gpu"], bool)
            or spec["cpu"] != 1.0
            or spec["gpu"] != 0.25
            or spec.get("remote")
        ):
            raise ValueError(f"balanced_four_gpu requires local {name} actors with CPU=1 and GPU=0.25")
        roles[name] = {"count": count, "CPU": 1.0, "GPU": 0.25, "bundles": list(ROLE_BUNDLES[name])}
    return {
        "schema_version": 1,
        "policy": "balanced_four_gpu",
        "strategy": "STRICT_PACK",
        "bundles": [{"CPU": 4.0, "GPU": 1.0} for _ in range(4)],
        "roles": roles,
        "reserved_resources": {"CPU": 16.0, "GPU": 4.0},
        "actor_resources": {"CPU": 14.0, "GPU": 3.5},
        "bundle_numbers_are_not_physical_GPU_indices": True,
    }


def _identity(value: Any) -> str:
    return value.hex() if callable(getattr(value, "hex", None)) else str(value)


def _gpu_uuid(value: Any) -> str:
    text = value.decode() if isinstance(value, bytes) else str(value)
    return "GPU-" + str(uuid.UUID(text.removeprefix("GPU-")))


def _model_devices(operator: Any) -> list[str]:
    devices = set()
    pending, seen = [(operator, 0)], set()
    while pending:
        value, depth = pending.pop()
        if id(value) in seen or depth > 6:
            continue
        seen.add(id(value))
        parameters = getattr(value, "parameters", None)
        if callable(parameters):
            devices.update(str(parameter.device) for parameter in parameters())
        for key in ("_operator", "_model", "model", "detector", "recognizer", "relational"):
            child = getattr(value, key, None)
            if child is not None:
                pending.append((child, depth + 1))
    return sorted(devices)


def _save(path: Path, value: dict[str, Any]) -> None:
    import yaml

    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        yaml.safe_dump(value, stream, sort_keys=True)
    temporary.replace(path)


def _process_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def _claim_actor_slot(directory: Path, record: dict[str, Any]) -> None:
    """Admit at most one constructor per role/slot for this owner generation."""
    role, slot = record["role"], record["slot"]
    if (
        role not in ROLE_BUNDLES
        or type(slot) is not int
        or not 0 <= slot < len(ROLE_BUNDLES[role])
        or record["actor_name"] != f"nrl-{record['owner']}-{role}-{slot}"
    ):
        raise RuntimeError("GPU actor constructor has an invalid owned slot")
    claims = directory / "claims"
    claims.mkdir(mode=0o700, exist_ok=True)
    claimed = claims / f"{role}-{slot}"
    try:
        claimed.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError("GPU placement actor slot was already claimed; recreation is not admitted") from error
    _save(claimed / "claim.yaml", record)


@dataclass(frozen=True)
class SlotCallback:
    """Serializable token only; mutable allocation/lock stays on its driver."""

    owner_token: str
    role: str
    process: int

    def __call__(self) -> dict[str, Any]:
        if os.getpid() != self.process or self.owner_token not in _OWNERS:
            raise RuntimeError("GPU placement callback is outside its registered owner/generation")
        return _OWNERS[self.owner_token].allocate(self.role)


class PlacementOperatorAdapter:
    """Check ownership before constructing the unchanged underlying operator."""

    def __init__(self, operator_class: type, operator_kwargs: dict[str, Any], placement: dict[str, Any]) -> None:
        import ray

        context = ray.get_runtime_context()
        name = context.get_actor_name()
        names = placement["names"]
        if name not in names:
            raise RuntimeError("GPU actor name is not an admitted placement slot")
        slot = names.index(name)
        pg_id = _identity(context.get_placement_group_id())
        accelerator_ids = context.get_accelerator_ids().get("GPU", [])
        if pg_id != placement["group_id"] or len(accelerator_ids) != 1:
            raise RuntimeError("GPU actor is outside its owned placement group or single-GPU slot")
        record = {
            "schema_version": 1,
            "owner": placement["owner"],
            "role": placement["role"],
            "slot": slot,
            "bundle": placement["bundles"][slot],
            "actor_name": name,
            "actor_id": _identity(context.get_actor_id()),
            "group_id": pg_id,
            "node_id": _identity(context.get_node_id()),
            "job_id": _identity(context.get_job_id()),
            "worker_id": _identity(context.get_worker_id()),
            "pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(os.getpid()),
            "accelerator_ids": list(accelerator_ids),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "state": "constructing",
        }
        # Read-only remote-argument inspection never claims a slot. A real
        # constructor claims it atomically before CUDA or operator/model work.
        _claim_actor_slot(Path(placement["directory"]), record)
        import torch

        gpu_uuid = _gpu_uuid(torch.cuda.get_device_properties(0).uuid)
        if gpu_uuid not in placement["allowed_GPU_UUIDs"]:
            raise RuntimeError("GPU actor is outside the admitted physical GPU UUID lease")
        record["GPU_UUID"] = gpu_uuid
        path = Path(placement["directory"]) / f"{record['actor_id']}-{os.getpid()}.yaml"
        self._placement_record = record
        self._placement_path = path
        _save(path, record)
        try:
            self._operator = operator_class(**operator_kwargs)
            record["model_devices"] = _model_devices(self._operator)
            if record["model_devices"] and not all(device.startswith("cuda:") for device in record["model_devices"]):
                raise RuntimeError("GPU placement did not verify CUDA model parameters")
            # HF embedders intentionally load on their first normal batch. Do
            # not add eager loading or a synthetic inference to placement.
            record["state"] = "ready" if record["model_devices"] else "constructed"
        except BaseException as exc:
            record.update(state="failed", error_type=type(exc).__name__)
            raise
        finally:
            _save(path, record)

    def __call__(self, batch: Any) -> Any:
        try:
            result = self._operator(batch)
            if self._placement_record["state"] != "ready":
                devices = _model_devices(self._operator)
                if not devices or not all(device.startswith("cuda:") for device in devices):
                    raise RuntimeError("GPU placement did not verify CUDA model parameters after the normal batch")
                self._placement_record.update(model_devices=devices, state="ready")
                _save(self._placement_path, self._placement_record)
            return result
        except BaseException as exc:
            self._placement_record.update(state="failed", error_type=type(exc).__name__)
            _save(self._placement_path, self._placement_record)
            raise


class BalancedPlacement:
    """One ingestion-owned non-detached group, shared by all executor graphs."""

    def __init__(self, ray: Any, executors: list[Any], resources: Any, *, ready_timeout_s: float = 60.0) -> None:
        from nemo_retriever.common.ray_runtime import require_owned_local_ray_runtime
        from nemo_retriever.graph.executor import _preflight_executor_nodes
        from nemo_retriever.graph.operator_resolution import resolve_graph
        from nemo_retriever.operators.gpu_operator import GPUOperator

        validate_policy("balanced_four_gpu", ready_timeout_s)
        require_owned_local_ray_runtime(ray)
        self.ray = ray
        self.timeout = float(ready_timeout_s)
        self.token = uuid.uuid4().hex
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.observed_names: set[str] = set()
        self.registered: set[str] = set()
        self.issued: list[str] = []
        self.pg = None
        self.timer_stop = threading.Event()
        self.timer: threading.Thread | None = None
        self.startup_error: str | None = None
        self.namespace = ray.get_runtime_context().namespace
        self.receipt: dict[str, Any] = {
            "schema_version": 1,
            "policy": "balanced_four_gpu",
            "status": "preflight",
            "owner": self.token,
            "creator_pid": self.pid,
            "creator_job_id": _identity(ray.get_runtime_context().get_job_id()),
            "creator_node_id": _identity(ray.get_runtime_context().get_node_id()),
            "creator_worker_id": _identity(ray.get_runtime_context().get_worker_id()),
            "terminal_iterator_closed": False,
        }
        alive = [node for node in ray.nodes() if node["Alive"]]
        if len(alive) != 1 or alive[0]["NodeID"] != _identity(ray.get_runtime_context().get_node_id()):
            raise ValueError("balanced_four_gpu requires one owned local Ray node")
        if ray.available_resources().get("GPU") != 4 or ray.cluster_resources().get("GPU") != 4:
            raise ValueError("balanced_four_gpu requires four wholly available Ray GPUs")
        self.allowed_uuids = _visible_gpu_uuids()
        outside = outside_resources(resources)
        specs, cpu_entries = [], []
        self.graphs = {}
        for executor in executors:
            graph = resolve_graph(executor.graph, resources)
            self.graphs[id(executor)] = graph
            cpu_nodes = []
            for node in executor._linearize(graph):
                if executor._stream_ingest_index([node]) is not None:
                    continue
                options = executor._node_overrides.get(node.name, {})
                if any(
                    key in options
                    for key in (
                        "scheduling_strategy",
                        "ray_remote_args_fn",
                        "name",
                        "lifetime",
                        "max_restarts",
                        "runtime_env",
                        "namespace",
                        "get_if_exists",
                    )
                ):
                    raise ValueError("balanced_four_gpu rejects custom actor scheduling/lifecycle overrides")
                gpu = executor._scheduled_num_gpus(node, options, resources.available_gpu_count())
                if gpu:
                    if not issubclass(node.operator_class, GPUOperator):
                        raise ValueError("balanced_four_gpu rejects GPU reservations on CPU-only operators")
                    if node.name in executor._auto_concurrency_nodes:
                        raise ValueError("balanced_four_gpu requires explicit fixed GPU worker counts")
                    specs.append(
                        {
                            "name": node.name,
                            "concurrency": options.get("concurrency", 1),
                            "cpu": options.get("num_cpus", executor._default_num_cpus),
                            "gpu": gpu,
                            "remote": executor._has_remote_endpoint(node),
                        }
                    )
                else:
                    count = options.get("concurrency", 1)
                    cpu = options.get("num_cpus", executor._default_num_cpus)
                    if (
                        isinstance(cpu, bool)
                        or not isinstance(cpu, (int, float))
                        or not math.isfinite(cpu)
                        or cpu <= 0
                        or type(count) is not int
                        or count <= 0
                    ):
                        raise ValueError(
                            "balanced_four_gpu requires finite positive CPUs and fixed outside actor counts"
                        )
                    cpu_nodes.append(node)
            cpu_entries.append((executor, cpu_nodes))
        self.plan = validate_role_specs(specs)
        # GPU actors consume only bundle CPUs. The existing shared admission
        # still accounts for source, normalization and every outside CPU actor.
        _preflight_executor_nodes(cpu_entries, outside.available_cpu_count(), 0, reserved_cpus=0)
        for executor in executors:
            executor._resources_preflight_complete = True
            executor._preflight_source_cpu_reservation = executor._source_cpu_reservation
            executor._preflight_cluster_resources = resources
            executor._placement = self
        self.receipt.update(
            plan=self.plan,
            outside_CPU_budget=outside.available_cpu_count(),
            allowed_GPU_UUIDs=self.allowed_uuids,
            ready_timeout_s=self.timeout,
            source_read_CPU=sum(executor._source_cpu_reservation for executor in executors),
            outside_CPU_actors=[
                {
                    "executor": index,
                    "name": node.name,
                    "class": node.operator_class.__name__,
                    "concurrency": executor._node_overrides.get(node.name, {}).get("concurrency", 1),
                    "CPU": executor._node_overrides.get(node.name, {}).get("num_cpus", executor._default_num_cpus),
                }
                for index, (executor, nodes) in enumerate(cpu_entries)
                for node in nodes
            ],
        )
        self.directory = Path(tempfile.mkdtemp(prefix="nrl-gpu-placement-"))
        self.directory.chmod(0o700)

    def __enter__(self) -> BalancedPlacement:
        from ray.util.placement_group import placement_group

        try:
            self.pg = placement_group(
                self.plan["bundles"], strategy="STRICT_PACK", name="nrl-" + self.token, lifetime=None
            )
            _OWNERS[self.token] = self
            self.ray.get(self.pg.ready(), timeout=self.timeout)
            self.receipt.update(group_id=_identity(self.pg.id), group_name="nrl-" + self.token, status="running")
            self.timer = threading.Thread(target=self._startup_watch, daemon=True)
            self.timer.start()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def register(self, role: str) -> SlotCallback:
        with self.lock:
            if role in self.registered or role not in ROLE_BUNDLES:
                raise RuntimeError("GPU placement role was reused or is unsupported")
            self.registered.add(role)
        return SlotCallback(self.token, role, self.pid)

    def actor_names(self, role: str) -> list[str]:
        return [f"nrl-{self.token}-{role}-{slot}" for slot in range(len(ROLE_BUNDLES[role]))]

    def allocate(self, role: str) -> dict[str, Any]:
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        with self.lock:
            if role not in self.registered:
                raise RuntimeError("GPU placement callback role is not registered")
            if self.startup_error:
                raise RuntimeError(self.startup_error)
            names = self.actor_names(role)
            # Ray calls this function for diagnostics as well as creation.
            # Only actual named registration or a permanent constructor claim
            # advances the pending slot. Do not retain get_actor strong refs.
            cursor = len(names) - 1
            for index, name in enumerate(names):
                if name not in self.observed_names:
                    claimed = (self.directory / "claims" / f"{role}-{index}").exists()
                    if not claimed:
                        try:
                            self.ray.get_actor(name, namespace=self.namespace)
                        except ValueError:
                            cursor = index
                            break
                    self.observed_names.add(name)
            name = names[cursor]
            if name not in self.issued:
                self.issued.append(name)
        return {
            "name": name,
            "get_if_exists": False,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=ROLE_BUNDLES[role][cursor],
                placement_group_capture_child_tasks=False,
            ),
        }

    def actor_payload(self, role: str) -> dict[str, Any]:
        return {
            "owner": self.token,
            "role": role,
            "names": self.actor_names(role),
            "bundles": list(ROLE_BUNDLES[role]),
            "directory": str(self.directory),
            "group_id": _identity(self.pg.id),
            "allowed_GPU_UUIDs": self.allowed_uuids,
        }

    def _records(self) -> list[dict[str, Any]]:
        import yaml

        records = {}
        for path in [*sorted((self.directory / "claims").glob("*/claim.yaml")), *sorted(self.directory.glob("*.yaml"))]:
            record = yaml.safe_load(path.read_text())
            name = record["actor_name"]
            previous = records.get(name)
            if previous is not None and any(previous[key] != record[key] for key in ("actor_id", "pid", "owner")):
                raise RuntimeError("GPU placement claim and constructor identity differ")
            records[name] = record
        return list(records.values())

    def _startup_watch(self) -> None:
        from ray.util.placement_group import remove_placement_group

        deadline = time.monotonic() + self.timeout
        while not self.timer_stop.wait(0.1):
            if len([r for r in self._records() if r["state"] == "ready"]) == 14:
                return
            if time.monotonic() >= deadline:
                self.startup_error = "GPU placement actor startup timed out"
                remove_placement_group(self.pg)
                return

    def iteration_closed(self) -> None:
        """Called after the public materializing Ray iterator has closed."""
        self.receipt["terminal_iterator_closed"] = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        from ray.util.placement_group import placement_group_table, remove_placement_group

        self.timer_stop.set()
        if self.timer is not None:
            self.timer.join(timeout=2)
        records, errors, forced = [], [], []
        removed, physical_exit, resources_released = False, False, False
        try:
            records = self._records()
            if exc_type is None:
                validate_runtime_records(
                    records,
                    self.token,
                    _identity(self.pg.id),
                    self.allowed_uuids,
                    creator_node_id=self.receipt.get("creator_node_id"),
                    creator_job_id=self.receipt.get("creator_job_id"),
                )
                if not self.receipt["terminal_iterator_closed"]:
                    errors.append("The terminal Ray Data iterator was not explicitly closed")
        except Exception as error:
            errors.append(str(error))
        if self.startup_error:
            errors.append(self.startup_error)
        try:
            process_records = [
                record
                for record in records
                if type(record.get("pid")) is int
                and type(record.get("process_start_ticks")) is int
                and isinstance(record.get("actor_name"), str)
            ]

            def live_records() -> list[dict[str, Any]]:
                return [
                    record
                    for record in process_records
                    if _process_start_ticks(record["pid"]) == record["process_start_ticks"]
                ]

            # Public iter_batches.close/exhaustion has already shut its executor
            # down. Observe physical exit without get_actor's extra strong refs.
            deadline = time.monotonic() + min(2.0, self.timeout)
            while live_records() and time.monotonic() < deadline:
                time.sleep(0.05)
            live_names = {record["actor_name"] for record in live_records()}
            observed_names = {record["actor_name"] for record in process_records}
            for name in self.issued:
                if name in observed_names and name not in live_names:
                    continue
                try:
                    actor = self.ray.get_actor(name, namespace=self.namespace)
                except ValueError:
                    continue
                self.ray.kill(actor, no_restart=True)
                forced.append(name)
            if self.pg is not None:
                remove_placement_group(self.pg)
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    removed = placement_group_table(self.pg).get("state") == "REMOVED"
                    physical_exit = (
                        len(process_records) == len(records)
                        and not live_records()
                        and set(self.issued) <= observed_names
                    )
                    resources_released = self.ray.available_resources().get("GPU", 0) >= 4
                    if removed and physical_exit and resources_released:
                        break
                    time.sleep(0.05)
                if not removed:
                    errors.append("Owned placement group removal was not confirmed")
                if not physical_exit:
                    errors.append("Owned actor physical process exit was not verified")
                if not resources_released:
                    errors.append("Owned GPU resources were not released before query admission")
            if forced:
                errors.append("Owned actors required forced cleanup")
        except Exception as error:
            errors.append("Owned cleanup failed: " + type(error).__name__)
            if self.pg is not None and not removed:
                try:
                    remove_placement_group(self.pg)
                    removed = placement_group_table(self.pg).get("state") == "REMOVED"
                except Exception:
                    errors.append("Owned placement group removal also failed")
        finally:
            self.receipt.update(
                status="passed" if not errors and exc_type is None else "failed",
                actors=records,
                cleanup={
                    "status": "passed" if not errors else "failed",
                    "forced_actor_names": forced,
                    "errors": errors,
                    "owned_group_removed": removed,
                    "actor_processes_exited": physical_exit,
                    "GPU_resources_released": resources_released,
                },
            )
            _OWNERS.pop(self.token, None)
            try:
                shutil.rmtree(self.directory)
            except OSError as error:
                errors.append("Owned receipt directory cleanup failed: " + type(error).__name__)
                self.receipt["status"] = self.receipt["cleanup"]["status"] = "failed"
        if errors and exc_type is None:
            raise RuntimeError("GPU placement verification/cleanup failed: " + "; ".join(errors))


def validate_runtime_records(
    records: list[dict[str, Any]],
    owner: str,
    group_id: str,
    uuids: list[str],
    *,
    creator_node_id: str | None = None,
    creator_job_id: str | None = None,
) -> None:
    if len(records) != 14:
        raise ValueError("GPU placement requires exactly 14 completed actor observations")
    observed = {}
    identities: dict[str, set[Any]] = {key: set() for key in ("actor_id", "worker_id", "pid", "actor_name")}
    for record in records:
        role, slot = record.get("role"), record.get("slot")
        if role not in ROLE_BUNDLES or type(slot) is not int or not 0 <= slot < len(ROLE_BUNDLES[role]):
            raise ValueError("GPU placement has an unknown role/slot")
        if (
            (role, slot) in observed
            or record.get("owner") != owner
            or record.get("group_id") != group_id
            or record.get("state") != "ready"
            or record.get("bundle") != ROLE_BUNDLES[role][slot]
            or record.get("GPU_UUID") not in uuids
            or type(record.get("pid")) is not int
            or type(record.get("process_start_ticks")) is not int
            or (creator_node_id is not None and record.get("node_id") != creator_node_id)
            or (creator_job_id is not None and record.get("job_id") != creator_job_id)
            or record.get("actor_name") != f"nrl-{owner}-{role}-{slot}"
            or not isinstance(record.get("accelerator_ids"), list)
            or len(record["accelerator_ids"]) != 1
            or not isinstance(record["accelerator_ids"][0], str)
            or not record["accelerator_ids"][0]
            or not isinstance(record.get("CUDA_VISIBLE_DEVICES"), str)
            or not record["CUDA_VISIBLE_DEVICES"]
            or not record.get("model_devices")
            or not all(device.startswith("cuda:") for device in record["model_devices"])
        ):
            raise ValueError("GPU placement actor ownership, readiness or UUID does not match")
        for key in ("actor_id", "worker_id", "node_id", "job_id", "group_id"):
            value = record.get(key)
            if not isinstance(value, str) or not value or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError("GPU placement requires nonempty hexadecimal Ray identity evidence")
        for key, seen in identities.items():
            if record[key] in seen:
                raise ValueError("GPU placement actor identities must be unique")
            seen.add(record[key])
        observed[(role, slot)] = record
    bundle_uuids = [{r["GPU_UUID"] for r in records if r["bundle"] == bundle} for bundle in range(4)]
    if any(len(values) != 1 for values in bundle_uuids) or len(set.union(*bundle_uuids)) != 4:
        raise ValueError("GPU placement did not demonstrate four distinct physical bundle GPU UUIDs")


def _visible_gpu_uuids() -> list[str]:
    import pynvml

    pynvml.nvmlInit()
    try:
        tokens = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(tokens) != 4 or any(not token.strip() for token in tokens):
            raise ValueError("balanced_four_gpu requires an explicit four-device CUDA_VISIBLE_DEVICES lease")
        uuids = []
        for token in tokens:
            token = token.strip()
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(int(token))
                if token.isdigit()
                else pynvml.nvmlDeviceGetHandleByUUID(token)
            )
            value = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(_gpu_uuid(value))
        if len(set(uuids)) != 4:
            raise ValueError("balanced_four_gpu requires four distinct physical GPU UUIDs")
        return uuids
    finally:
        pynvml.nvmlShutdown()
