#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ads import jobs
from ads.opctl.backend.ads_ml_job import MLJobOperatorBackend
from ads.opctl.backend.local import LocalOperatorBackend, OperatorLoader
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import BACKEND_NAME
from ads.opctl.operator.common.const import ENV_OPERATOR_ARGS, PACK_TYPE
from ads.opctl.operator.common.operator_loader import OperatorInfo
from ads.opctl.operator.runtime import runtime as operator_runtime
from ads.opctl.operator.runtime.runtime import ContainerRuntime, PythonRuntime


class TestLocalOperatorBackend:
    def setup_class(cls):
        # current directory and test files directory
        cls.CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_FILES_DIR = os.path.join(cls.CUR_DIR, "test_files")

        # mock backends
        cls.MOCK_BACKEND_CONFIG = {
            "local.container.config": operator_runtime.ContainerRuntime.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "local_container.yaml")
            ).to_dict(),
            "local.python.config": operator_runtime.PythonRuntime.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "local_python.yaml")
            ).to_dict(),
        }

    def setup_method(self):
        # mock operator config
        self.mock_config = {
            "kind": "operator",
            "type": "example",
            "version": "v1",
            "spec": {},
            "runtime": {},
            "infrastructure": {},
            "execution": {"oci_config": "test_oci_config"},
        }

        # mock operator info
        self.mock_operator_info = OperatorInfo(
            type="example",
            gpu="no",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path=os.path.join(self.TEST_FILES_DIR, "test_operator"),
            backends=[BACKEND_NAME.JOB.value, BACKEND_NAME.DATAFLOW.value],
        )

        # mock operator backend
        self.mock_backend = LocalOperatorBackend(
            config=self.mock_config, operator_info=self.mock_operator_info
        )

    def test__init(self):
        """Ensures that the local operator backend can be successfully initialized."""
        assert self.mock_backend.runtime_config == {}
        expected_operator_config = {
            **{
                key: value
                for key, value in self.mock_config.items()
                if key not in ("runtime", "infrastructure", "execution")
            }
        }
        assert self.mock_backend.operator_config == expected_operator_config
        assert self.mock_backend.operator_type == self.mock_config["type"]

        assert operator_runtime.ContainerRuntime.type in self.mock_backend._RUNTIME_MAP
        assert operator_runtime.PythonRuntime.type in self.mock_backend._RUNTIME_MAP

        self.mock_backend.operator_info = self.mock_operator_info

    @patch("runpy.run_module")
    def test__run_with_python(self, mock_run_module):
        """Tests running the operator within a local python environment."""
        self.mock_backend.runtime_config = PythonRuntime.init().to_dict()
        result = self.mock_backend._run_with_python()
        mock_run_module.assert_called_with(
            self.mock_operator_info.type, run_name="__main__"
        )
        assert result == 0

    @patch("runpy.run_module")
    def test__run_with_python_fail(self, mock_run_module):
        """Tests running the operator within a local python environment."""
        mock_run_module.side_effect = SystemExit(1)
        self.mock_backend.runtime_config = PythonRuntime.init().to_dict()
        result = self.mock_backend._run_with_python()
        mock_run_module.assert_called_with(
            self.mock_operator_info.type, run_name="__main__"
        )
        assert result == 1

    @patch("ads.opctl.backend.local.run_container")
    def test__run_with_container(self, mock_run_container):
        """Tests running the operator within a container."""
        self.mock_backend.runtime_config = ContainerRuntime.init(
            **{
                "image": "test_image",
                "env": [{"name": "test_env_key", "value": "test_env_value"}],
                "volume": ["host_value:container_value"],
            }
        ).to_dict()
        self.mock_backend._run_with_container()

        mock_run_container.assert_called_with(
            image="test_image",
            bind_volumes={"host_value": {"bind": "container_value"}},
            env_vars={
                "test_env_key": "test_env_value",
                ENV_OPERATOR_ARGS: json.dumps(self.mock_backend.operator_config),
            },
            command=f"'python3 -m {self.mock_operator_info.type}'",
        )

    @pytest.mark.parametrize(
        "mock_runtime_type, mock_runtime_config",
        [
            ("python", PythonRuntime().to_dict()),
            ("container", ContainerRuntime().to_dict()),
        ],
    )
    def test_run_success(self, mock_runtime_type, mock_runtime_config):
        """Test running the operator with success result"""
        self.mock_backend.runtime_config = mock_runtime_config
        self.mock_backend.operator_info = None

        mock_run_with = MagicMock(return_value=0)
        self.mock_backend._RUNTIME_MAP[mock_runtime_type] = mock_run_with

        with patch.object(OperatorLoader, "from_uri") as mock_operator_loader_from_uri:
            # mock objects
            mock_operator_loader_from_uri.return_value = MagicMock(
                load=MagicMock(return_value=self.mock_operator_info)
            )

            self.mock_backend.run()
            mock_run_with.assert_called()

            mock_run_with.return_value = 1
            with pytest.raises(RuntimeError):
                self.mock_backend.run()

    def test_run_fail(self):
        """Test running the operator with failed result"""
        with pytest.raises(RuntimeError):
            self.mock_backend.runtime_config = {"type": "undefined"}
            self.mock_backend.run()

    def test_init_fail(self):
        """Ensures that initiating starter config fails in case of wrong input params."""
        with pytest.raises(ValueError):
            self.mock_backend.init(runtime_type="unknown")

    @pytest.mark.parametrize(
        "mock_runtime_type, expected_result",
        [
            (
                "python",
                yaml.load(
                    "# This YAML specification was auto generated by the `ads operator init` "
                    "command.\n# The more details about the operator's runtime YAML "
                    "specification can be found in the ADS documentation:\n# "
                    "https://accelerated-data-science.readthedocs.io/en/latest "
                    "\n\n\nkind: operator.local\nspec: null\ntype: python\nversion: v1\n",
                    Loader=yaml.FullLoader,
                ),
            ),
            (
                "container",
                yaml.load(
                    "# This YAML specification was auto generated by the `ads operator "
                    "init` command.\n# The more details about the operator's runtime YAML "
                    "specification can be found in the ADS documentation:\n# "
                    "https://accelerated-data-science.readthedocs.io/en/latest \n\n\nkind: "
                    "operator.local\nspec:\n  env:\n  - name: operator\n    "
                    "value: example:v1\n  image: example:v1\n  "
                    "volume:\n  - :/root/.oci\ntype: container\nversion: v1\n",
                    Loader=yaml.FullLoader,
                ),
            ),
        ],
    )
    def test_init_success(self, mock_runtime_type, expected_result):
        """Tests generating a starter YAML specification for the operator local runtime."""
        assert (
            yaml.load(
                self.mock_backend.init(runtime_type=mock_runtime_type),
                Loader=yaml.FullLoader,
            )
            == expected_result
        )


class TestMLJobOperatorBackend:
    """Tests backend class to run operator on Data Science Jobs."""

    def setup_class(cls):
        # current directory and test files directory
        cls.CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_FILES_DIR = os.path.join(cls.CUR_DIR, "test_files")

        # mock backends
        cls.MOCK_BACKEND_CONFIG = {
            "job.container.config": jobs.Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "job_container.yaml")
            ).to_dict(),
            "job.python.config": jobs.Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "job_python.yaml")
            ).to_dict(),
        }

    def setup_method(self):
        self.mock_config = (
            ConfigProcessor(
                {
                    "kind": "operator",
                    "type": "example",
                    "version": "v1",
                    "spec": {},
                    "runtime": {},
                }
            )
            .step(ConfigMerger, **{})
            .config
        )

        # mock operator info
        self.mock_operator_info = OperatorInfo(
            type="example",
            gpu="no",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path=os.path.join(self.TEST_FILES_DIR, "test_operator"),
            backends=[BACKEND_NAME.JOB.value, BACKEND_NAME.DATAFLOW.value],
        )

        # mock operator backend
        self.mock_backend = MLJobOperatorBackend(
            config=self.mock_config, operator_info=self.mock_operator_info
        )

    def test__init(self):
        assert self.mock_backend.job is None
        assert self.mock_backend.runtime_config == {}

        expected_operator_config = {
            **{
                key: value
                for key, value in self.mock_config.items()
                if key not in ("runtime", "infrastructure", "execution")
            }
        }
        assert self.mock_backend.operator_config == expected_operator_config
        assert self.mock_backend.operator_type == self.mock_config["type"]
        assert self.mock_backend.operator_version == self.mock_config["version"]

        assert jobs.ContainerRuntime().type in self.mock_backend._RUNTIME_MAP
        assert jobs.PythonRuntime().type in self.mock_backend._RUNTIME_MAP

        self.mock_backend.operator_info = self.mock_operator_info

    def test__adjust_common_information(self):
        self.mock_backend.job = jobs.Job(name="{job", runtime=jobs.PythonRuntime({}))
        self.mock_backend._adjust_common_information()

        assert self.mock_backend.job.name == (
            f"job_{self.mock_operator_info.type.lower()}"
            f"_{self.mock_operator_info.version.lower()}"
        )

    def test__adjust_container_runtime(self):
        self.mock_backend.job = jobs.Job(
            name="{job", runtime=jobs.ContainerRuntime().with_image("test-image")
        )
        self.mock_backend._adjust_container_runtime()

        assert self.mock_backend.job.runtime.to_dict() == (
            {
                "kind": "runtime",
                "spec": {
                    "cmd": "python3 -m example",
                    "entrypoint": None,
                    "env": [
                        {"name": "OCI_IAM_TYPE", "value": "resource_principal"},
                        {"name": "OCIFS_IAM_TYPE", "value": "resource_principal"},
                        {
                            "name": "ENV_OPERATOR_ARGS",
                            "value": '{"kind": "operator", "type": "example", '
                            '"version": "v1", "spec": {}}',
                        },
                    ],
                    "image": "test-image",
                },
                "type": "container",
            }
        )

    @patch("time.time", return_value=1)
    def test__adjust_python_runtime(self, mock_time):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.mkdtemp", return_value=temp_dir):
                self.mock_backend.job = jobs.Job(
                    name="{job", runtime=jobs.PythonRuntime()
                )
                self.mock_backend._adjust_python_runtime()

                assert self.mock_backend.job.runtime.to_dict() == (
                    {
                        "kind": "runtime",
                        "type": "python",
                        "spec": {
                            "entrypoint": "example_1_run.sh",
                            "scriptPathURI": temp_dir,
                            "workingDir": os.path.basename(temp_dir.rstrip("/")),
                            "env": [
                                {"name": "OCI_IAM_TYPE", "value": "resource_principal"},
                                {
                                    "name": "OCIFS_IAM_TYPE",
                                    "value": "resource_principal",
                                },
                                {
                                    "name": "ENV_OPERATOR_ARGS",
                                    "value": '{"kind": "operator", "type": "example", "version": "v1", "spec": {}}',
                                },
                            ],
                        },
                    }
                )

    @pytest.mark.parametrize(
        "mock_runtime_type, mock_runtime_config",
        [
            (
                "python",
                jobs.Job(
                    name="test_name",
                    runtime=jobs.PythonRuntime(),
                    infrastructure=jobs.DataScienceJob(),
                ).to_dict(),
            ),
            (
                "container",
                jobs.Job(
                    name="test_name",
                    runtime=jobs.ContainerRuntime(),
                    infrastructure=jobs.DataScienceJob(),
                ).to_dict(),
            ),
        ],
    )
    @patch.object(jobs.Job, "create")
    @patch.object(jobs.Job, "run")
    def test_run_success(
        self, mock_job_run, mock_job_create, mock_runtime_type, mock_runtime_config
    ):
        mock_job_create.return_value = MagicMock(run=mock_job_run)
        """Test running the operator with success result"""
        self.mock_backend.runtime_config = mock_runtime_config
        self.mock_backend.operator_info = None

        mock_run_with = MagicMock()
        self.mock_backend._RUNTIME_MAP[mock_runtime_type] = mock_run_with

        with patch.object(OperatorLoader, "from_uri") as mock_operator_loader_from_uri:
            # mock objects
            mock_operator_loader_from_uri.return_value = MagicMock(
                load=MagicMock(return_value=self.mock_operator_info)
            )

            self.mock_backend.run()
            mock_run_with.assert_called()
            mock_job_run.assert_called()
            mock_job_create.assert_called()
