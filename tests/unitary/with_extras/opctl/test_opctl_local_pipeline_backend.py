#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pytest
from pathlib import Path
from pytest import fixture
from unittest import SkipTest
from unittest.mock import ANY, call, patch

try:
    from ads.opctl.backend.local import LocalPipelineBackend
except (ImportError, AttributeError) as e:
    raise SkipTest("OCI MLPipeline is not available. Skipping the MLPipeline tests.")


class TestLocalPipelineBackend:
    @fixture(autouse=True)
    def before_each(self):
        LocalPipelineBackend.DEFAULT_STATUS_POLL_INTERVAL_SECONDS = 0

    def pipeline_config_base(self):
        return {
            "kind": "pipeline",
            "execution": {
                "backend": "local",
                "debug": False,
                "oci_config": "~/.oci/config",
                "oci_profile": "BOAT",
            },
            "infrastructure": {"max_parallel_containers": 1},
        }

    def pipeline_config_with_files(self, tmp_path):
        os.makedirs(os.path.join(tmp_path, "conda", "slug"))
        os.makedirs(os.path.join(tmp_path, "src"))
        os.makedirs(os.path.join(tmp_path, "work"))
        Path(os.path.join(tmp_path, "src", "script.sh")).touch()
        Path(os.path.join(tmp_path, "work", "myscript.py")).touch()
        with open(
            os.path.join(tmp_path, "conda", "slug", "pack_manifest.yaml"), "w"
        ) as f:
            f.write("manifest:\n  name: conda env\n")
        with open(os.path.join(tmp_path, "src", "notebook.ipynb"), "w") as f:
            f.write(
                '{"cells": [], "metadata": {},  "nbformat": 4, "nbformat_minor": 4}'
            )
        config = self.pipeline_config_base()
        config["execution"]["conda_pack_folder"] = os.path.join(tmp_path, "conda")
        config["execution"]["source_folder"] = os.path.join(tmp_path, "src")
        return config

    def test_job_steps_not_supported(self):
        config = self.pipeline_config_base()
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "dataScienceJob",
                    "spec": {
                        "description": "A job step",
                        "jobId": "ocid1.datasciencejob.oc1.iad.<unique_ocid>",
                        "name": "job_step",
                    },
                }
            ]
        }
        with pytest.raises(
            ValueError,
            match="Step job_step has unsupported kind. "
            "Local pipeline execution only supports pipeline steps with kind customScript.",
        ):
            LocalPipelineBackend(config).run()

    def test_git_runtime_not_supported(self):
        config = self.pipeline_config_base()
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "Test step",
                        "name": "git_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "gitPython",
                            "spec": {
                                "conda": {"slug": "foo_p38_cpu_v7", "type": "service"},
                                "url": "http://my.git.server/my.project.git",
                                "branch": "master",
                            },
                        },
                    },
                }
            ]
        }
        with pytest.raises(
            ValueError,
            match="Step git_step has unsupported runtime. "
            "Supported values are: script, python, notebook",
        ):
            LocalPipelineBackend(config).run()

    @patch(
        "ads.opctl.backend.local.LocalBackend._activate_conda_env_and_run",
        return_value=0,
    )
    def test_local_pipeline_parses_runtimes(self, activate_and_run, tmp_path):
        config = self.pipeline_config_with_files(tmp_path)
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "script_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"FOO": "BAR"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "notebook step",
                        "name": "notebook_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "notebook",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "notebookEncoding": "utf-8",
                                "notebookPathURI": "notebook.ipynb",
                                "env": {"ABC": "123"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "python step",
                        "name": "python_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "python",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "myscript.py",
                                "workingDir": os.path.join(tmp_path, "work"),
                                "env": {"COLOR": "RED"},
                            },
                        },
                    },
                },
            ]
        }
        LocalPipelineBackend(config).run()
        calls = [
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"FOO": "BAR"},
            ),
            call(
                "ml-job",
                "slug",
                "python /etc/datascience/operators/src/notebook.py ",
                ANY,
                {"ABC": "123"},
            ),
            call(
                "ml-job",
                "slug",
                "python /etc/datascience/operators/work/myscript.py ",
                ANY,
                {"COLOR": "RED"},
            ),
        ]
        activate_and_run.assert_has_calls(calls, any_order=True)
        assert activate_and_run.call_count == 3

    @patch("ads.opctl.backend.local.ThreadPoolExecutor")
    def test_max_parallel_containers_respected(self, executor, tmp_path):
        config = self.pipeline_config_with_files(tmp_path)
        config["infrastructure"]["max_parallel_containers"] = 1
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "script_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"FOO": "BAR"},
                            },
                        },
                    },
                },
            ]
        }
        LocalPipelineBackend(config).run()
        executor.assert_called_with(max_workers=1)

    @patch("ads.opctl.backend.local.ThreadPoolExecutor")
    @patch("ads.opctl.backend.local.os.cpu_count")
    def test_max_parallel_containers_default_when_not_specified(
        self, cpu_count, executor, tmp_path
    ):
        config = self.pipeline_config_with_files(tmp_path)
        del config["infrastructure"]["max_parallel_containers"]
        cpu_count.side_effect = [3, 15]
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "script_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                            },
                        },
                    },
                },
            ]
        }
        LocalPipelineBackend(config).run()
        executor.assert_called_with(max_workers=3)

        LocalPipelineBackend(config).run()
        executor.assert_called_with(max_workers=4)

    @patch("ads.opctl.backend.local.LocalBackend")
    def test_pipeline_step_source_folder(self, local_backend, tmp_path):
        config = self.pipeline_config_with_files(tmp_path)
        config["spec"] = {
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "script_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"FOO": "BAR"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "python step",
                        "name": "python_step",
                        "runtime": {
                            "kind": "runtime",
                            "type": "python",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "myscript.py",
                                "workingDir": os.path.join(tmp_path, "work"),
                                "env": {"COLOR": "RED"},
                            },
                        },
                    },
                },
            ]
        }
        LocalPipelineBackend(config).run()

        config["execution"]["source_folder"] = None
        LocalPipelineBackend(config).run()

        call_args_list = local_backend.call_args_list
        assert len(call_args_list) == 4
        args, kwargs = call_args_list[0]
        assert args[0]["execution"]["source_folder"] == os.path.join(
            tmp_path, "src"
        ), "source folder for step should match what is specified in the pipeline config."
        args, kwargs = call_args_list[1]
        assert args[0]["execution"]["source_folder"] == os.path.join(
            tmp_path, "work"
        ), "source folder for python runtime step should match the step's work dir."
        args, kwargs = call_args_list[2]
        assert (
            args[0]["execution"]["source_folder"] == os.getcwd()
        ), "source folder for step should default to current working directory when not specified in pipeline config."
        args, kwargs = call_args_list[3]
        assert args[0]["execution"]["source_folder"] == os.path.join(
            tmp_path, "work"
        ), "source folder for python runtime step should match the step's work dir."

    @patch(
        "ads.opctl.backend.local.LocalBackend._activate_conda_env_and_run",
        return_value=0,
    )
    def test_steps_in_correct_order(self, activate_and_run, tmp_path):
        config = self.pipeline_config_with_files(tmp_path)
        config["spec"] = {
            "dag": [
                "(step_1, step_2, step_3) >> step_4",
                "step_1 >> step_5",
                "step_5 >> step_7",
                "step_5 >> step_8",
                "(step_7, step_8) >> step_9",
            ],
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_9",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "9"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_8",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "8"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_7",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "7"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_6",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "6"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_5",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "5"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_4",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "4"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_3",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "3"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_2",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "2"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_1",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "1"},
                            },
                        },
                    },
                },
            ],
        }
        LocalPipelineBackend(config).run()
        # The steps are processed in order in a for loop, and the test config uses no parallelization, so we can
        # determine the specific order that steps should be run in.
        calls = [
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "6"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "3"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "2"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "1"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "5"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "4"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "8"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "7"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "9"},
            ),
        ]
        activate_and_run.assert_has_calls(calls)
        assert activate_and_run.call_count == 9

    @patch("ads.opctl.backend.local.LocalBackend._activate_conda_env_and_run")
    def test_waiting_steps_skipped_when_step_fails(self, activate_and_run, tmp_path):
        activate_and_run.side_effect = [0, 2]
        config = self.pipeline_config_with_files(tmp_path)
        config["spec"] = {
            "dag": ["step_1 >> step_2", "step_2 >> step_3", "step_3 >> step_4"],
            "stepDetails": [
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_1",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "1"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_2",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "2"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_3",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "3"},
                            },
                        },
                    },
                },
                {
                    "kind": "customScript",
                    "spec": {
                        "description": "script step",
                        "name": "step_4",
                        "runtime": {
                            "kind": "runtime",
                            "type": "script",
                            "spec": {
                                "conda": {"slug": "slug", "type": "service"},
                                "scriptPathURI": "script.sh",
                                "env": {"STEP": "4"},
                            },
                        },
                    },
                },
            ],
        }
        LocalPipelineBackend(config).run()
        calls = [
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "1"},
            ),
            call(
                "ml-job",
                "slug",
                "cd /etc/datascience/operators/src && /bin/bash script.sh ",
                ANY,
                {"STEP": "2"},
            ),
        ]
        activate_and_run.assert_has_calls(calls)
        assert activate_and_run.call_count == 2
