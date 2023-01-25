#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy
import os
import shutil
import tempfile
from distutils import dir_util
from typing import Tuple, Dict
import shlex

import ads
from ads.common.auth import create_signer, AuthContext
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
from ads.opctl.constants import DEFAULT_IMAGE_SCRIPT_DIR
from jinja2 import Environment, PackageLoader
from ads.opctl.distributed.common.cluster_config_helper import (
    ClusterConfigToJobSpecConverter,
)

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
        Create Job and Job Run from YAML.
        """
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            job = Job.from_dict(self.config)
            job.create()
            job_run = job.run()
            print("JOB OCID:", job.id)
            print("JOB RUN OCID:", job_run.id)

    def run(self) -> None:
        """
        Create Job and Job Run from OCID or cli parameters.
        """
        # TODO Check that this still runs smoothly for distributed
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            if self.config["execution"].get("ocid", None):
                job_id = self.config["execution"]["ocid"]
                run_id = (
                    Job.from_datascience_job(self.config["execution"]["ocid"]).run().id
                )
            else:
                payload = self._create_payload()  # create job with infrastructure
                src_folder = self.config["execution"].get("source_folder")
                if self.config["execution"].get("conda_type") and self.config[
                    "execution"
                ].get("conda_slug"):
                    # add conda runtime
                    job_id, run_id = self._run_with_conda_pack(payload, src_folder)
                elif self.config["execution"].get("image"):
                    # add docker image runtime
                    job_id, run_id = self._run_with_image(payload)
                else:
                    raise ValueError(
                        "Either conda info or image name should be provided."
                    )
            print("JOB OCID:", job_id)
            print("JOB RUN OCID:", run_id)
            return {"job_id": job_id, "run_id": run_id}

    def init_operator(self):
        # TODO: check if folder is empty, check for force overwrite
        # TODO: check that command is being run from advanced-ds repo (important until ads released)

        operator_folder = self.config["execution"].get("operator_folder_path")
        os.makedirs(operator_folder, exist_ok=True)

        operator_folder_name = os.path.basename(os.path.normpath(operator_folder))
        docker_tag = f"{os.path.join(self.config['infrastructure'].get('docker_registry'), operator_folder_name)}:latest"

        self.config["execution"]["operator_folder_name"] = operator_folder_name
        self.config["execution"]["docker_tag"] = docker_tag

        operator_slug = self.config["execution"].get("operator_slug")
        self._jinja_write(operator_slug, operator_folder)

        # DONE
        print(
            "\nInitialization Successful.\n"
            f"All code should be written in main.py located at: {os.path.join(operator_folder, 'main.py')}\n"
            f"Additional libraries should be added to environment.yaml located at: {os.path.join(operator_folder, 'environment.yaml')}\n"
            "Any changes to main.py will require re-building the docker image, whereas changes to args in the"
            " runtime section of the yaml file do not. Write accordingly.\n"
            "Run this cluster with:\n"
            f"\tdocker build -t {docker_tag} -f {os.path.join(operator_folder, 'Dockerfile')} .\n"
            f"\tads opctl publish-image {docker_tag} \n"
            f"\tads opctl run -f {os.path.join(operator_folder, operator_slug + '.yaml')} \n"
        )
        return operator_folder

    def delete(self):
        """
        Delete Job or Job Run from OCID.
        """
        if self.config["execution"].get("id"):
            job_id = self.config["execution"]["id"]
            with AuthContext():
                ads.set_auth(auth=self.auth_type, profile=self.profile)
                Job.from_datascience_job(job_id).delete()
                print(f"Job {job_id} has been deleted.")
        elif self.config["execution"].get("run_id"):
            run_id = self.config["execution"]["run_id"]
            with AuthContext():
                ads.set_auth(auth=self.auth_type, profile=self.profile)
                DataScienceJobRun.from_ocid(run_id).delete()
                print(f"Job run {run_id} has been deleted.")

    def cancel(self):
        """
        Cancel Job Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            DataScienceJobRun.from_ocid(run_id).cancel()
            print(f"Job run {run_id} has been cancelled.")

    def watch(self):
        """
        Watch Job Run from OCID.
        """
        run_id = self.config["execution"]["run_id"]

        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            run = DataScienceJobRun.from_ocid(run_id)
            run.watch()

    def _jinja_write(self, operator_slug, operator_folder):
        # TODO AH: fill in templates with relevant details
        env = Environment(
            loader=PackageLoader("ads", f"opctl/operators/{operator_slug}")
        )

        for setup_file in [
            "Dockerfile",
            "environment.yaml",
            "main.py",
            "run.py",
            "start_scheduler.sh",
            "start_worker.sh",
            "dask_cluster.yaml",
        ]:
            template = env.get_template(setup_file + ".jinja2")
            with open(os.path.join(operator_folder, setup_file), "w") as ff:
                ff.write(template.render(config=self.config))

    def _create_payload(self, infra=None, name=None) -> Job:
        if not infra:
            infra = self.config.get("infrastructure", {})
        # if any(k not in infra for k in REQUIRED_FIELDS):
        #    missing = [k for k in REQUIRED_FIELDS if k not in infra]
        #    raise ValueError(
        #        f"Following fields are missing but are required for OCI ML Jobs: {missing}. Please run `ads opctl configure`."
        #    )

        ml_job = DataScienceJob(spec=infra if "spec" not in infra else infra["spec"])

        log_group_id = infra.get("log_group_id")
        log_id = infra.get("log_id")

        if log_group_id:
            ml_job.with_log_group_id(log_group_id)
        if log_id:
            ml_job.with_log_id(log_id)
        if not name:
            try:
                name = infra.get("displayName") or self.config["execution"].get(
                    "job_name"
                )
            except:
                pass

        return Job(
            name=name,
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


class MLJobDistributedBackend(MLJobBackend):
    DIAGNOSTIC_COMMAND = "python -m ads.opctl.diagnostics -t distributed"

    def __init__(self, config: Dict) -> None:
        """
        Initialize a MLJobDistributedBackend object given config dictionary.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        super().__init__(config=config)
        self.job = None

    def prepare_job_config(self, cluster_info):
        job_conf_helper = ClusterConfigToJobSpecConverter(cluster_info)
        jobdef_conf = job_conf_helper.job_def_info()
        infrastructure = cluster_info.infrastructure
        if jobdef_conf.get("name"):
            infrastructure["spec"]["displayName"] = jobdef_conf.get("name")
        job = self._create_payload(infrastructure["spec"])
        envVars = {}
        envVars.update(
            cluster_info.cluster.config.envVars
        )  # Add user provided environment variables
        envVars.update(
            jobdef_conf.get("envVars") or {}
        )  # Update with `OCI__` environment variables

        job.with_runtime(ContainerRuntime().with_environment_variable(**envVars))
        job.runtime.with_image(image=jobdef_conf["image"])
        self.job = job
        if os.path.basename(jobdef_conf["image"]) == jobdef_conf["image"]:
            logger.warning("Did you include registry in image name?")
        main_jobrun_conf = job_conf_helper.job_run_info("main")
        main_jobrun_conf["envVars"]["RANK"] = "0"
        main_jobrun_conf["name"] = main_jobrun_conf.get("name") or "main"

        worker_jobrun_conf = job_conf_helper.job_run_info("worker")
        worker_jobrun_conf_list = []
        if worker_jobrun_conf:
            for i in range(cluster_info.cluster.worker.replicas):
                conf = copy.deepcopy(worker_jobrun_conf)
                conf["envVars"]["RANK"] = str(i + 1)
                conf["name"] = (
                    conf.get("name", worker_jobrun_conf["envVars"]["OCI__MODE"])
                    + "_"
                    + str(i)
                )
                worker_jobrun_conf_list.append(conf)
        ps_jobrun_conf = job_conf_helper.job_run_info("ps")
        ps_jobrun_conf_list = []
        if ps_jobrun_conf:
            for i in range(cluster_info.cluster.ps.replicas):
                conf = copy.deepcopy(ps_jobrun_conf)
                conf["name"] = (
                    conf.get("name", worker_jobrun_conf["envVars"]["OCI__MODE"])
                    + "_"
                    + str(i)
                )
                ps_jobrun_conf_list.append(conf)

        worker_jobrun_conf_list.extend(ps_jobrun_conf_list)
        return main_jobrun_conf, worker_jobrun_conf_list

    @staticmethod
    def generate_worker_name(worker_jobrun_conf, i):
        return f"{worker_jobrun_conf['name']}-{i}"

    def run_diagnostics(self, cluster_info, dry_run=False, **kwargs):
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            main_jobrun_conf, worker_jobrun_conf_list = self.prepare_job_config(
                cluster_info=cluster_info
            )
            self.job.runtime.with_entrypoint(["/bin/bash", "--login", "-c"])
            self.job.runtime.with_cmd(MLJobDistributedBackend.DIAGNOSTIC_COMMAND)
            if dry_run:  # If dry run, print the job yaml on the console.
                print(
                    "-----------------------------Entering dryrun mode----------------------------------"
                )
                print(f"Creating Job with payload: \n{self.job}")
                print("+" * 200)

                print(f"Creating Main Job Run with following details:")
                print(f"Name: {main_jobrun_conf['name']}")
                print(f"Additional Environment Variables: ")
                main_env_Vars = main_jobrun_conf.get("envVars", {})
                for k in main_env_Vars:
                    print(f"\t{k}:{main_env_Vars[k]}")
                print("~" * 200)

                print(
                    "-----------------------------Ending dryrun mode----------------------------------"
                )
                return None
            else:
                job = self.job.create()

                # Start main job
                conf = dict(main_jobrun_conf)
                main_jobrun = job.run(
                    conf["name"],
                    env_var=conf["envVars"],
                    # freeform_tags={"distributed_training": "oracle-ads"},
                )
                self.job = job
                main_jobrun.watch()
                return job, main_jobrun

    def run(self, cluster_info, dry_run=False) -> None:
        """
        * Creates Job Definition  and starts main and worker jobruns from that job definition
        * The Job Definition will contain all the environment variables defined at the cluster/spec/config level, environment variables defined by the user at runtime/spec/env level and `OCI__` derived from the yaml specification
        * The Job Run will have overrides provided by the user under cluster/spec/{main|worker}/config section and `OCI__MODE`={MASTER|WORKER} depending on the run type
        """
        with AuthContext():
            ads.set_auth(auth=self.auth_type, profile=self.profile)
            main_jobrun_conf, worker_jobrun_conf_list = self.prepare_job_config(
                cluster_info=cluster_info
            )
            if dry_run:  # If dry run, print the job yaml on the console.
                print(
                    "-----------------------------Entering dryrun mode----------------------------------"
                )
                print(f"Creating Job with payload: \n{self.job}")
                print(
                    "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                )

                print(f"Creating Main Job Run with following details:")
                print(f"Name: {main_jobrun_conf['name']}")
                print(f"Additional Environment Variables: ")
                main_env_Vars = main_jobrun_conf.get("envVars", {})
                for k in main_env_Vars:
                    print(f"\t{k}:{main_env_Vars[k]}")
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                )
                if cluster_info.cluster.worker:
                    print(f"Creating Job Runs with following details:")
                    for i in range(len(worker_jobrun_conf_list)):
                        worker_jobrun_conf = worker_jobrun_conf_list[i]
                        print("Name: " + worker_jobrun_conf.get("name"))
                        print("Additional Environment Variables: ")
                        worker_env_Vars = worker_jobrun_conf.get("envVars", {})
                        for k in worker_env_Vars:
                            print(f"\t{k}:{worker_env_Vars[k]}")

                print(
                    "-----------------------------Ending dryrun mode----------------------------------"
                )
                return None

            else:
                job = self.job.create()

                # Start main job
                conf = dict(main_jobrun_conf)
                main_jobrun = job.run(
                    conf["name"],
                    env_var=conf["envVars"],
                    # freeform_tags={"distributed_training": "oracle-ads"},
                )

                # Start worker job
                worker_jobruns = []
                if cluster_info.cluster.worker:
                    for i in range(len(worker_jobrun_conf_list)):
                        worker_jobrun_conf = worker_jobrun_conf_list[i]
                        conf = dict(worker_jobrun_conf)
                        jobrun = job.run(
                            worker_jobrun_conf.get("name"),
                            env_var=conf["envVars"],
                        )
                        worker_jobruns.append(jobrun)
                self.job = job
                return job, main_jobrun, worker_jobruns
