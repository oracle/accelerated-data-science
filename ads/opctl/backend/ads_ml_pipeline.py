#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, Union
from ads.common.auth import create_signer, AuthContext
from ads.common.oci_client import OCIClientFactory
from ads.opctl.backend.base import Backend
from ads.opctl.backend.ads_ml_job import JobRuntimeFactory
from ads.opctl.decorator.common import print_watch_command
from ads.pipeline import Pipeline, PipelineRun, PipelineStep, CustomScriptStep

from ads.jobs import PythonRuntime


class PipelineBackend(Backend):
    def __init__(self, config: Dict) -> None:
        """
        Initialize a MLPipeline object given config dictionary.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config
        self.oci_auth = create_signer(
            config["execution"].get("auth"),
            config["execution"].get("oci_config", None),
            config["execution"].get("oci_profile", None),
        )
        self.auth_type = config["execution"].get("auth")
        self.profile = config["execution"].get("oci_profile", None)
        self.client = OCIClientFactory(**self.oci_auth).data_science

    @print_watch_command
    def apply(self) -> Dict:
        """
        Create Pipeline and Pipeline Run from YAML.
        """
        with AuthContext(auth=self.auth_type, profile=self.profile):
            pipeline = Pipeline.from_dict(self.config)
            pipeline.create()
            pipeline_run = pipeline.run()
            print("PIPELINE OCID:", pipeline.id)
            print("PIPELINE RUN OCID:", pipeline_run.id)
            return {"job_id": pipeline.id, "run_id": pipeline_run.id}

    @print_watch_command
    def run(self) -> Dict:
        """
        Create Pipeline and Pipeline Run from OCID.
        """
        pipeline_id = self.config["execution"]["ocid"]
        with AuthContext(auth=self.auth_type, profile=self.profile):
            pipeline = Pipeline.from_ocid(ocid=pipeline_id)
            pipeline_run = pipeline.run()
            print("PIPELINE OCID:", pipeline.id)
            print("PIPELINE RUN OCID:", pipeline_run.id)
            return {"job_id": pipeline.id, "run_id": pipeline_run.id}

    def delete(self) -> None:
        """
        Delete Pipeline or Pipeline Run from OCID.
        """
        if self.config["execution"].get("id"):
            pipeline_id = self.config["execution"]["id"]
            with AuthContext(auth=self.auth_type, profile=self.profile):
                Pipeline.from_ocid(pipeline_id).delete()
                print(f"Pipeline {pipeline_id} has been deleted.")
        elif self.config["execution"].get("run_id"):
            run_id = self.config["execution"]["run_id"]
            with AuthContext(auth=self.auth_type, profile=self.profile):
                PipelineRun.from_ocid(run_id).delete()
                print(f"Pipeline run {run_id} has been deleted.")

    def cancel(self) -> None:
        """
        Cancel Pipeline Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]
        with AuthContext(auth=self.auth_type, profile=self.profile):
            PipelineRun.from_ocid(run_id).cancel()
            print(f"Pipeline run {run_id} has been cancelled.")

    def watch(self) -> None:
        """
        Watch Pipeline Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]
        log_type = self.config["execution"].get("log_type")
        interval = self.config["execution"].get("interval")
        with AuthContext(auth=self.auth_type, profile=self.profile):
            PipelineRun.from_ocid(run_id).watch(
                interval=interval,
                log_type=log_type
            )

    def init(
        self,
        uri: Union[str, None] = None,
        overwrite: bool = False,
        runtime_type: Union[str, None] = None,
        **kwargs: Dict,
    ) -> Union[str, None]:
        """Generates a starter YAML specification for an MLPipeline.

        Parameters
        ----------
        overwrite: (bool, optional). Defaults to False.
            Overwrites the result specification YAML if exists.
        uri: (str, optional), Defaults to None.
            The filename to save the resulting specification template YAML.
        runtime_type: (str, optional). Defaults to None.
                The resource runtime type.
        **kwargs: Dict
            The optional arguments.

        Returns
        -------
        Union[str, None]
            The YAML specification for the given resource if `uri` was not provided.
            `None` otherwise.
        """

        with AuthContext(auth=self.auth_type, profile=self.profile):
            # define a pipeline step
            pipeline_step = (
                PipelineStep("pipeline_step_name_1")
                .with_description("A step running a python script")
                .with_infrastructure(CustomScriptStep().init())
                .with_runtime(
                    JobRuntimeFactory.get_runtime(
                        key=runtime_type or PythonRuntime().type
                    ).init()
                )
            )

            # define a pipeline
            pipeline = (
                Pipeline(
                    name="Pipeline Name",
                    spec=(self.config.get("infrastructure", {}) or {}),
                )
                .with_step_details([pipeline_step])
                .with_dag(["pipeline_step_name_1"])
                .init()
            )

            note = (
                "# This YAML specification was auto generated by the `ads opctl init` command.\n"
                "# The more details about the jobs YAML specification can be found in the ADS documentation:\n"
                "# https://accelerated-data-science.readthedocs.io/en/latest/user_guide/pipeline/quick_start.html \n\n"
            )

            return pipeline.to_yaml(
                uri=uri,
                overwrite=overwrite,
                note=note,
                filter_by_attribute_map=True,
                **kwargs,
            )
