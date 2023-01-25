#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict
import ads
from ads.common.auth import create_signer, AuthContext
from ads.common.oci_client import OCIClientFactory
from ads.opctl.backend.base import Backend
from ads.pipeline import Pipeline, PipelineRun


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

    def apply(self) -> None:
        """
        Create Pipeline and Pipeline Run from YAML.
        """
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            pipeline = Pipeline.from_dict(self.config)
            pipeline.create()
            pipeline_run = pipeline.run()
            print("PIPELINE OCID:", pipeline.id)
            print("PIPELINE RUN OCID:", pipeline_run.id)

    def run(self) -> None:
        """
        Create Pipeline and Pipeline Run from OCID.
        """
        pipeline_id = self.config["execution"]["ocid"]
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            pipeline = Pipeline.from_ocid(ocid=pipeline_id)
            pipeline_run = pipeline.run()
            print("PIPELINE RUN OCID:", pipeline_run.id)

    def delete(self) -> None:
        """
        Delete Pipeline or Pipeline Run from OCID.
        """
        if self.config["execution"].get("id"):
            pipeline_id = self.config["execution"]["id"]
            with AuthContext():
                ads.set_auth(auth=self.auth_type, profile=self.profile)
                Pipeline.from_ocid(pipeline_id).delete()
                print(f"Pipeline {pipeline_id} has been deleted.")
        elif self.config["execution"].get("run_id"):
            run_id = self.config["execution"]["run_id"]
            with AuthContext():
                ads.set_auth(auth=self.auth_type, profile=self.profile)
                PipelineRun.from_ocid(run_id).delete()
                print(f"Pipeline run {run_id} has been deleted.")

    def cancel(self) -> None:
        """
        Cancel Pipeline Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            PipelineRun.from_ocid(run_id).cancel()
            print(f"Pipeline run {run_id} has been cancelled.")

    def watch(self) -> None:
        """
        Watch Pipeline Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]
        log_type = self.config["execution"]["log_type"]
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            PipelineRun.from_ocid(run_id).watch(log_type=log_type)
