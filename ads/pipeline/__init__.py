#!/usr/bin/env python
# -*- coding: utf-8; -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

logger = logging.getLogger(__name__)
try:
    from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
    from ads.jobs.builders.runtimes.python_runtime import (
        GitPythonRuntime,
        NotebookRuntime,
        PythonRuntime,
        ScriptRuntime,
    )
    from ads.pipeline.ads_pipeline import Pipeline
    from ads.pipeline.ads_pipeline_run import (
        PIPELINE_RUN_TERMINAL_STATE,
        LogType,
        PipelineRun,
        LogType,
        ShowMode,
    )
    from ads.pipeline.ads_pipeline_step import PipelineStep
    from ads.pipeline.builders import infrastructure
    from ads.pipeline.builders.infrastructure.custom_script import CustomScriptStep
    from ads.pipeline.visualizer.base import GraphOrientation

except AttributeError as e:
    import oci.data_science

    if not hasattr(oci.data_science.models, "Pipeline"):
        logger.warning(
            "The OCI SDK you installed does not support Data Science Pipleines. ADS Pipelines API will not work."
        )
    else:
        raise e
