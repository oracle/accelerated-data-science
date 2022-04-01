#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from distutils import dir_util
from typing import Tuple, Dict
import shlex

from ads.common.auth import get_signer
from ads.common.oci_client import OCIClientFactory
from ads.jobs import (
    Job,
    DataScienceJobRun,
    DataScienceJob,
    ScriptRuntime,
    ContainerRuntime,
)
from ads.opctl import logger
from ads.opctl.backend.base import Backend
from ads.opctl.config.resolver import ConfigResolver
from ads.opctl.utils import (
    publish_image,
    OCIAuthContext,
)
from ads.opctl.constants import DEFAULT_IMAGE_SCRIPT_DIR

REQUIRED_FIELDS = [
    "project_id",
    "compartment_id",
    "subnet_id",
    "block_storage_size_in_GBs",
    "shape_name",
]


class MLJobBackend(Backend):
    def __init__(self, config: Dict) -> None:
        """
        Initialize a MLJobBackend object given config dictionary.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config
        self.oci_auth = get_signer(
            config["execution"].get("oci_config", None),
            config["execution"].get("oci_profile", None),
        )
        self.profile = config["execution"].get("oci_profile", None)
        self.client = OCIClientFactory(**self.oci_auth).data_science

    def run(self) -> None:
        with OCIAuthContext(profile=self.profile):
            if self.config["execution"].get("job_id", None):
                job_id = self.config["execution"]["job_id"]
                run_id = (
                    Job.from_datascience_job(self.config["execution"]["job_id"])
                    .run()
                    .id
                )
            else:
                payload = self._create_payload()
                src_folder = self.config["execution"].get("source_folder")
                if self.config["execution"].get("conda_type") and self.config[
                    "execution"
                ].get("conda_slug"):
                    job_id, run_id = self._run_with_conda_pack(payload, src_folder)
                elif self.config["execution"].get("image"):
                    job_id, run_id = self._run_with_image(payload)
                else:
                    raise ValueError(
                        "Either conda info or image name should be provided."
                    )
            print("JOB OCID:", job_id)
            print("JOB RUN OCID:", run_id)
            return {"job_id": job_id, "run_id": run_id}

    def delete(self):
        if self.config["execution"].get("job_id"):
            job_id = self.config["execution"]["job_id"]
            with OCIAuthContext(profile=self.profile):
                Job.from_datascience_job(job_id).delete()
        elif self.config["execution"].get("run_id"):
            run_id = self.config["execution"]["run_id"]
            with OCIAuthContext(profile=self.profile):
                DataScienceJobRun.from_ocid(run_id).delete()

    def cancel(self):
        run_id = self.config["execution"]["run_id"]
        with OCIAuthContext(profile=self.profile):
            DataScienceJobRun.from_ocid(run_id).cancel()

    def watch(self):
        run_id = self.config["execution"]["run_id"]

        with OCIAuthContext(profile=self.profile):
            run = DataScienceJobRun.from_ocid(run_id)
            run.watch()

    def _create_payload(self) -> Job:
        infra = self.config.get("infrastructure", {})
        if any(k not in infra for k in REQUIRED_FIELDS):
            missing = [k for k in REQUIRED_FIELDS if k not in infra]
            raise ValueError(
                f"Following fields are missing but are required for OCI ML Jobs: {missing}. Please run `ads opctl configure`."
            )

        ml_job = (
            DataScienceJob()
            .with_project_id(infra["project_id"])
            .with_compartment_id(infra["compartment_id"])
            .with_job_type("DEFAULT")
            .with_job_infrastructure_type("STANDALONE")
            .with_shape_name(infra["shape_name"])
            .with_block_storage_size(infra["block_storage_size_in_GBs"])
            .with_subnet_id(infra["subnet_id"])
        )

        log_group_id = infra.get("log_group_id")
        log_id = infra.get("log_id")

        if log_group_id:
            ml_job.with_log_group_id(log_group_id)
        if log_id:
            ml_job.with_log_id(log_id)
        return Job(
            name=self.config["execution"].get("job_name"),
            infrastructure=ml_job,
        )

    def _run_with_conda_pack(self, payload: Job, src_folder: str) -> Tuple[str, str]:
        payload.with_runtime(
            ScriptRuntime().with_environment_variable(
                **self.config["execution"]["env_vars"]
            )
        )
        if self.config["execution"].get("conda_type") == "service":
            payload.runtime.with_service_conda(self.config["execution"]["conda_slug"])
        else:
            payload.runtime.with_custom_conda(self.config["execution"]["conda_uri"])

        if ConfigResolver(self.config)._is_ads_operator():
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "operators"), exist_ok=True)
                dir_util.copy_tree(
                    src_folder,
                    os.path.join(td, "operators", os.path.basename(src_folder)),
                )
                curr_dir = os.path.dirname(os.path.abspath(__file__))
                shutil.copy(
                    os.path.join(curr_dir, "..", "operators", "run.py"),
                    os.path.join(td, "operators"),
                )
                payload.runtime.with_source(
                    os.path.join(td, "operators"), entrypoint="operators/run.py"
                )
                payload.runtime.set_spec(
                    "args", shlex.split(self.config["execution"]["command"] + " -r")
                )
                job = payload.create()
                job_id = job.id
                run_id = job.run().id
        else:
            with tempfile.TemporaryDirectory() as td:
                dir_util.copy_tree(
                    src_folder, os.path.join(td, os.path.basename(src_folder))
                )
                payload.runtime.with_source(
                    os.path.normpath(os.path.join(td, os.path.basename(src_folder))),
                    entrypoint=os.path.join(
                        os.path.basename(src_folder),
                        self.config["execution"]["entrypoint"],
                    ),
                )
                if self.config["execution"].get("command"):
                    payload.runtime.set_spec(
                        "args", shlex.split(self.config["execution"]["command"])
                    )
                job = payload.create()
                job_id = job.id
                run_id = job.run().id
        return job_id, run_id

    def _run_with_image(self, payload: Job) -> Tuple[str, str]:
        payload.with_runtime(
            ContainerRuntime().with_environment_variable(
                **self.config["execution"]["env_vars"]
            )
        )
        image = self.config["execution"]["image"]
        if ":" not in image:
            image += ":latest"
        payload.runtime.with_image(image)
        if os.path.basename(image) == image:
            logger.warn("Did you include registry in image name?")

        if ConfigResolver(self.config)._is_ads_operator():
            command = f"python {os.path.join(DEFAULT_IMAGE_SCRIPT_DIR, 'operators/run.py')} -r "
        else:
            command = ""
            # running a non-operator image
            if self.config["execution"].get("entrypoint"):
                payload.runtime.with_entrypoint(self.config["execution"]["entrypoint"])

        if self.config["execution"].get("command"):
            command += f"{self.config['execution']['command']}"
        if len(command) > 0:
            payload.runtime.with_cmd(",".join(shlex.split(command)))

        job = payload.create()
        job_id = job.id
        run_id = job.run().id
        return job_id, run_id
