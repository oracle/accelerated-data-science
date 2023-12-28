#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
from unittest.mock import MagicMock, patch

import pytest


from ads.jobs import Job
from ads.opctl.config.base import ConfigProcessor
from ads.opctl.config.merger import ConfigMerger
from ads.opctl.constants import BACKEND_NAME
from ads.opctl.operator.common.backend_factory import BackendFactory
from ads.opctl.operator.common.const import PACK_TYPE
from ads.opctl.operator.common.operator_loader import OperatorInfo, OperatorLoader
from ads.opctl.operator.runtime import runtime as operator_runtime


class TestBackendFactory:
    """Test the backend factory."""

    def setup_class(cls):
        # current directory and test files directory
        cls.CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_FILES_DIR = os.path.join(cls.CUR_DIR, "test_files")

        # mock backends
        cls.MOCK_BACKEND = {
            "job.config": Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "job_python.yaml")
            ).to_dict(),
            "job.python.config": Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "job_python.yaml")
            ).to_dict(),
            "job.container.config": Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "job_container.yaml")
            ).to_dict(),
            "dataflow.config": Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "dataflow_dataflow.yaml")
            ).to_dict(),
            "dataflow.dataflow.config": Job.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "dataflow_dataflow.yaml")
            ).to_dict(),
            "local.config": operator_runtime.PythonRuntime.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "local_python.yaml")
            ).to_dict(),
            "local.container.config": operator_runtime.ContainerRuntime.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "local_container.yaml")
            ).to_dict(),
            "local.python.config": operator_runtime.PythonRuntime.from_yaml(
                uri=os.path.join(cls.TEST_FILES_DIR, "local_python.yaml")
            ).to_dict(),
        }

    def setup_method(self):
        # mock operator info with the all supported backends
        self.mock_operator_info = OperatorInfo(
            type="example",
            gpu="no",
            description="An example operator",
            version="v1",
            conda="example_v1",
            conda_type=PACK_TYPE.CUSTOM,
            path="/fake/path/to/operator",
            backends=[BACKEND_NAME.JOB.value, BACKEND_NAME.DATAFLOW.value],
        )

        # mock operator config
        self.mock_operator_config = {
            "kind": "operator",
            "type": "example",
            "version": "v1",
            "spec": {},
        }

        # expected backends
        self.mock_supported_backends = tuple(
            set(BackendFactory.BACKENDS + BackendFactory.LOCAL_BACKENDS)
            & set(self.mock_operator_info.backends)
        )

    @pytest.mark.parametrize(
        "backend, expected_backend_kind, expected_runtime_type",
        [
            ("job", "job", "python"),
            ("job.container", "job", "container"),
            ("dataflow.dataflow", "dataflow", "dataflow"),
            ("local.container", "operator.local", "container"),
            ("local.python", "operator.local", "python"),
            ("invalid", None, None),
            ("job.invalid", None, None),
            ("local.invalid", None, None),
        ],
    )
    def test_extract_backend(
        self, backend, expected_backend_kind, expected_runtime_type
    ):
        """Ensure that the backend and runtime type are extracted correctly."""
        if expected_backend_kind is None:
            with pytest.raises(RuntimeError):
                BackendFactory._extract_backend(backend)
        else:
            backend_kind, runtime_type = BackendFactory._extract_backend(backend)
            assert backend_kind == expected_backend_kind
            assert runtime_type == expected_runtime_type

    def test_validate_backend_and_runtime(self):
        """Ensure that the backend and runtime type are validated correctly."""
        backend_kind = "job"
        runtime_type = "python"
        supported_backends = ["job", "dataflow", "operator_local", "local"]
        assert (
            BackendFactory._validate_backend_and_runtime(
                backend_kind, runtime_type, supported_backends
            )
            == True
        )

        backend_kind = "invalid_backend"
        runtime_type = "python"
        supported_backends = ["job", "dataflow", "operator_local", "local"]
        with pytest.raises(RuntimeError):
            BackendFactory._validate_backend_and_runtime(
                backend_kind, runtime_type, supported_backends
            )

        backend_kind = "job"
        runtime_type = "invalid_runtime"
        supported_backends = ["job", "dataflow", "operator_local", "local"]
        with pytest.raises(RuntimeError):
            BackendFactory._validate_backend_and_runtime(
                backend_kind, runtime_type, supported_backends
            )

    def test_get_backend_fail(self):
        """Ensures that getting backend fails in case of wrong input data."""

        mock_config = MagicMock()
        mock_config.return_value = {"kind": "job", "type": "python"}

        with pytest.raises(RuntimeError):
            BackendFactory.backend(config=None)

        mock_config.return_value = {"kind": "value"}
        with pytest.raises(RuntimeError):
            BackendFactory.backend(config=mock_config)

        mock_config.return_value = {"kind": "operator"}
        with pytest.raises(RuntimeError):
            BackendFactory.backend(config=mock_config)

    @pytest.mark.parametrize(
        "mock_backend, expected_backend_kind, expected_runtime_type",
        [
            (None, "operator.local", "python"),
            ("job", "job", "python"),
            ("job.python", "job", "python"),
            ("job.container", "job", "container"),
            ("dataflow", "dataflow", "dataflow"),
            ("dataflow.dataflow", "dataflow", "dataflow"),
            ("local", "operator.local", "python"),
            ("local.container", "operator.local", "container"),
            ("local.python", "operator.local", "python"),
            ("job.config", "job", "python"),
            ("job.python.config", "job", "python"),
            ("job.container.config", "job", "container"),
            ("dataflow.config", "dataflow", "dataFlow"),
            ("dataflow.dataflow.config", "dataflow", "dataFlow"),
            ("local.config", "operator.local", "python"),
            ("local.container.config", "operator.local", "container"),
            ("local.python.config", "operator.local", "python"),
        ],
    )
    @patch.object(BackendFactory, "_validate_backend_and_runtime")
    @patch.object(BackendFactory, "_init_backend_config")
    def test_get_backend(
        self,
        mock_init_backend_config,
        mock_validate_backend_and_runtime,
        mock_backend,
        expected_backend_kind,
        expected_runtime_type,
    ):
        """Ensure that the backend is returned correctly."""

        mock_backend_config = self.MOCK_BACKEND[
            f"{expected_backend_kind.replace('operator.','').lower()}.{expected_runtime_type.lower()}.config"
        ]

        # check if mock backend is a config dict
        if mock_backend in self.MOCK_BACKEND:
            mock_backend = self.MOCK_BACKEND[mock_backend]

        # prepares mock config by applying the config merger
        # this step can be replaced with magic mock
        mock_config = ConfigProcessor(self.mock_operator_config).step(
            ConfigMerger, **{}
        )

        with patch.object(OperatorLoader, "from_uri") as mock_operator_loader_from_uri:
            # mock objects
            mock_operator_loader_from_uri.return_value = MagicMock(
                load=MagicMock(return_value=self.mock_operator_info)
            )
            mock_init_backend_config.return_value = {
                (expected_backend_kind, expected_runtime_type): mock_backend_config
            }

            # run test
            result_backend = BackendFactory.backend(
                config=mock_config, backend=mock_backend
            )

            # validate result
            mock_operator_loader_from_uri.assert_called_with(
                uri=self.mock_operator_config["type"]
            )
            mock_validate_backend_and_runtime.assert_called_with(
                backend_kind=expected_backend_kind,
                runtime_type=expected_runtime_type,
                supported_backends=self.mock_supported_backends,
            )

            if isinstance(mock_backend, str):
                mock_init_backend_config.assert_called_with(
                    operator_info=self.mock_operator_info,
                    backend_kind=expected_backend_kind,
                    **{},
                )

            # validate result_backend
            assert result_backend.operator_type == self.mock_operator_config["type"]
            assert result_backend.operator_info == self.mock_operator_info
