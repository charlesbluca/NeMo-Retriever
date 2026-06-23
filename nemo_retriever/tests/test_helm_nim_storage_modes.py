# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for NIMService storage-mode compatibility knobs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"
_PAGE_TEMPLATE = "templates/nims/nemotron-page-elements-v3.yaml"
_CONFIGMAP_TEMPLATE = "templates/configmap.yaml"
_PAGE_NAME = "nemotron-page-elements-v3"


def _helm_template_for(show_only: str, extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-storage-mode-regression",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
        "--show-only",
        show_only,
    ]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    return _helm_template_for(_PAGE_TEMPLATE, extra_args=extra_args)


def _assert_helm_ok(testcase: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    testcase.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


class NimStorageModeTemplateTests(TestCase):
    def test_default_page_elements_uses_nimcache_storage(self) -> None:
        proc = _helm_template()
        _assert_helm_ok(self, proc)

        self.assertIn("kind: NIMCache", proc.stdout)
        self.assertIn(f"name: {_PAGE_NAME}", proc.stdout)
        self.assertIn("storage:\n    nimCache:\n      name: nemotron-page-elements-v3", proc.stdout)
        self.assertIn('modelPuller: "nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0"', proc.stdout)
        self.assertNotIn("modelEndpoint:", proc.stdout)

    def test_page_elements_can_override_nimcache_model_puller(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.page_elements.modelPuller=nvcr.io/example/cache-puller:9.9.9",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn('modelPuller: "nvcr.io/example/cache-puller:9.9.9"', proc.stdout)
        self.assertIn("kind: NIMCache", proc.stdout)
        self.assertIn("repository: nvcr.io/nim/nvidia/nemotron-page-elements-v3", proc.stdout)

    def test_page_elements_can_skip_nimcache_with_service_emptydir(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.page_elements.storage.nimCache.enabled=false",
                "--set",
                "nimOperator.page_elements.storage.service.emptyDir.sizeLimit=25Gi",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertNotIn("kind: NIMCache", proc.stdout)
        self.assertIn("kind: NIMService", proc.stdout)
        self.assertIn(f"name: {_PAGE_NAME}", proc.stdout)
        self.assertIn("storage:\n    emptyDir:\n      sizeLimit: 25Gi", proc.stdout)
        self.assertNotIn("nimCache:", proc.stdout)

    def test_page_elements_invoke_path_overrides_configmap_auto_wire(self) -> None:
        proc = _helm_template_for(
            _CONFIGMAP_TEMPLATE,
            extra_args=(
                "--set",
                "nimOperator.page_elements.invokePath=/v1/page-elements",
            ),
        )
        _assert_helm_ok(self, proc)

        self.assertIn(
            'page_elements_invoke_url: "http://nemotron-page-elements-v3:8000/v1/page-elements"',
            proc.stdout,
        )
        self.assertIn(
            'table_structure_invoke_url: "http://nemotron-table-structure-v1:8000/v1/infer"',
            proc.stdout,
        )

    def test_page_elements_can_render_ngc_model_endpoint(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.page_elements.modelEndpoint=https://example.invalid/model-endpoint",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn('modelEndpoint: "https://example.invalid/model-endpoint"', proc.stdout)
        self.assertIn("kind: NIMCache", proc.stdout)


if __name__ == "__main__":
    main()
