#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging

logger = logging.getLogger(__name__)
try:
    from ads.jobs.builders.runtimes.python_runtime import (
        PythonRuntime,
        GitPythonRuntime,
        NotebookRuntime,
        ScriptRuntime,
        DataFlowRuntime,
        DataFlowNotebookRuntime,
    )
    from ads.jobs.builders.runtimes.pytorch_runtime import PyTorchDistributedRuntime
    from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
    from ads.jobs.ads_job import Job
    from ads.jobs.builders import infrastructure
    from ads.jobs.builders.infrastructure.dsc_job import (
        DataScienceJob,
        DataScienceJobRun,
    )
    from ads.jobs.builders.infrastructure.dataflow import DataFlow, DataFlowRun
except AttributeError as e:
    import oci.data_science

    if not hasattr(oci.data_science.models, "Job"):
        logger.warning(
            "The OCI SDK you installed does not support Data Science Jobs. ADS Jobs API will not work."
        )
    else:
        raise e

__all__ = [
    "Job",
    "DataScienceJob",
    "DataScienceJobRun",
    "PythonRuntime",
    "GitPythonRuntime",
    "NotebookRuntime",
    "ScriptRuntime",
    "ContainerRuntime",
    "PyTorchDistributedRuntime",
    "DataFlow",
    "DataFlowRun",
    "DataFlowRuntime",
    "DataFlowNotebookRuntime",
]
