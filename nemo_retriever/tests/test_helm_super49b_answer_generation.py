# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for Super-49B answer generation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALUES_YAML = _REPO_ROOT / "nemo_retriever/helm/values.yaml"
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"

_SUPER49B_KEY = "  llama_3_3_nemotron_super_49b_v1_5:"
_SUPER49B_SERVICE = "llama-3-3-nemotron-super-49b-v1-5"
_SUPER49B_REPOSITORY = "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
_SUPER49B_TAG = "2.0.5"
_SUPER49B_MODEL = "openai/nvidia/llama-3.3-nemotron-super-49b-v1.5"
_SUPER49B_PROFILE = "1146f49f84dff5dea09f5aa633cc70b92d7d972223d67878c841cd0fbccad4fb"


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    cmd = [
        helm,
        "template",
        "retriever",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _assert_helm_ok(self: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    self.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


class HelmSuper49BAnswerGenerationTests(TestCase):
    def test_values_define_super49b_as_optional_answer_nim(self) -> None:
        values = _read_required_file(_VALUES_YAML)

        self.assertIn(_SUPER49B_KEY, values)
        block = values[values.index(_SUPER49B_KEY) : values.index(_SUPER49B_KEY) + 1200]
        self.assertIn("enabled: false", block)
        self.assertIn(f"repository: {_SUPER49B_REPOSITORY}", block)
        self.assertIn(f'tag: "{_SUPER49B_TAG}"', block)
        self.assertIn("nvidia.com/gpu: 2", block)
        self.assertIn('size: "250Gi"', block)
        self.assertIn(_SUPER49B_PROFILE, block)

    def test_default_render_omits_super49b_and_disables_llm_answering(self) -> None:
        proc = _helm_template()
        _assert_helm_ok(self, proc)

        self.assertNotIn(f"name: {_SUPER49B_SERVICE}", proc.stdout)
        self.assertIn("llm:", proc.stdout)
        self.assertIn("enabled: false", proc.stdout)
        self.assertIn("api_base: null", proc.stdout)

    def test_super49b_opt_in_renders_nim_and_autowires_llm_config(self) -> None:
        proc = _helm_template(extra_args=("--set", "nimOperator.llama_3_3_nemotron_super_49b_v1_5.enabled=true"))
        _assert_helm_ok(self, proc)

        self.assertIn(f"name: {_SUPER49B_SERVICE}", proc.stdout)
        self.assertIn(f"repository: {_SUPER49B_REPOSITORY}", proc.stdout)
        self.assertIn(f"tag: {_SUPER49B_TAG}", proc.stdout)
        self.assertIn("nvidia.com/gpu: 2", proc.stdout)
        self.assertIn("profiles:", proc.stdout)
        self.assertIn(_SUPER49B_PROFILE, proc.stdout)
        self.assertIn("NIM_PASSTHROUGH_ARGS", proc.stdout)
        self.assertIn("--disable-custom-all-reduce", proc.stdout)
        self.assertIn("NCCL_IB_DISABLE", proc.stdout)
        self.assertIn("NCCL_P2P_DISABLE", proc.stdout)
        self.assertIn('api_base: "http://llama-3-3-nemotron-super-49b-v1-5:8000/v1"', proc.stdout)
        self.assertIn(f'model: "{_SUPER49B_MODEL}"', proc.stdout)
        self.assertIn('rag_system_prompt_prefix: "/no_think"', proc.stdout)
        self.assertIn("enabled: true", proc.stdout)

    def test_explicit_llm_api_base_wins_without_operator_nim(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.llm.enabled=true",
                "--set",
                "serviceConfig.llm.apiBase=http://external-llm:8000/v1",
                "--set",
                "serviceConfig.llm.model=openai/custom-answerer",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertNotIn(f"name: {_SUPER49B_SERVICE}", proc.stdout)
        self.assertIn('api_base: "http://external-llm:8000/v1"', proc.stdout)
        self.assertIn('model: "openai/custom-answerer"', proc.stdout)
        self.assertIn("rag_system_prompt_prefix: null", proc.stdout)

    def test_llm_api_key_secret_renders_env_not_configmap_value(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.llm.enabled=true",
                "--set",
                "serviceConfig.llm.apiBase=http://external-llm:8000/v1",
                "--set",
                "serviceConfig.llm.apiKeySecret.name=llm-secret",
                "--set",
                "serviceConfig.llm.apiKeySecret.key=OPENAI_API_KEY",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn("api_key: null", proc.stdout)
        self.assertIn("NEMO_RETRIEVER_LLM_API_KEY", proc.stdout)
        self.assertIn('name: "llm-secret"', proc.stdout)
        self.assertIn('key: "OPENAI_API_KEY"', proc.stdout)
        self.assertNotIn('api_key: "', proc.stdout)


if __name__ == "__main__":
    main()
