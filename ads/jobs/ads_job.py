#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from typing import List, Union, Dict
from urllib.parse import urlparse

import fsspec
from ads.common.auth import default_signer
from ads.jobs.builders.base import Builder
from ads.jobs.builders.infrastructure.dataflow import DataFlow, DataFlowRun
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob, DataScienceJobRun
from ads.jobs.builders.runtimes.pytorch_runtime import PyTorchDistributedRuntime
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.jobs.builders.runtimes.python_runtime import (
    DataFlowRuntime,
    GitPythonRuntime,
    NotebookRuntime,
    PythonRuntime,
    Runtime,
    ScriptRuntime,
)


class Job(Builder):
    """Represents a Job defined by infrastructure and runtime.

    Examples
    --------
    Here is an example for creating and running a job::

        from ads.jobs import Job, DataScienceJob, PythonRuntime

        # Define an OCI Data Science job to run a python script
        job = (
            Job(name="<job_name>")
            .with_infrastructure(
                DataScienceJob()
                # Configure logging for getting the job run outputs.
                .with_log_group_id("<log_group_ocid>")
                # Log resource will be auto-generated if log ID is not specified.
                .with_log_id("<log_ocid>")
                # If you are in an OCI data science notebook session,
                # the following configurations are not required.
                # Configurations from the notebook session will be used as defaults.
                .with_compartment_id("<compartment_ocid>")
                .with_project_id("<project_ocid>")
                .with_subnet_id("<subnet_ocid>")
                .with_shape_name("VM.Standard.E3.Flex")
                # Shape config details are applicable only for the flexible shapes.
                .with_shape_config_details(memory_in_gbs=16, ocpus=1)
                # Minimum/Default block storage size is 50 (GB).
                .with_block_storage_size(50)
            )
            .with_runtime(
                PythonRuntime()
                # Specify the service conda environment by slug name.
                .with_service_conda("pytorch110_p38_cpu_v1")
                # The job artifact can be a single Python script, a directory or a zip file.
                .with_source("local/path/to/code_dir")
                # Environment variable
                .with_environment_variable(NAME="Welcome to OCI Data Science.")
                # Command line argument, arg1 --key arg2
                .with_argument("arg1", key="arg2")
                # Set the working directory
                # When using a directory as source, the default working dir is the parent of code_dir.
                # Working dir should be a relative path beginning from the source directory (code_dir)
                .with_working_dir("code_dir")
                # The entrypoint is applicable only to directory or zip file as source
                # The entrypoint should be a path relative to the working dir.
                # Here my_script.py is a file in the code_dir/my_package directory
                .with_entrypoint("my_package/my_script.py")
                # Add an additional Python path, relative to the working dir (code_dir/other_packages).
                .with_python_path("other_packages")
                # Copy files in "code_dir/output" to object storage after job finishes.
                .with_output("output", "oci://bucket_name@namespace/path/to/dir")
                # Tags
                .with_freeform_tag(my_tag="my_value")
                .with_defined_tag(**{"Operations": {"CostCenter": "42"}})
            )
        )
        # Create and Run the job
        run = job.create().run()
        # Stream the job run outputs
        run.watch()

    If you are in an OCI notebook session and you would like to use the same infrastructure
    configurations, the infrastructure configuration can be simplified.
    Here is another example of creating and running a jupyter notebook as a job::

        from ads.jobs import Job, DataScienceJob, NotebookRuntime

        # Define an OCI Data Science job to run a jupyter Python notebook
        job = (
            Job(name="<job_name>")
            .with_infrastructure(
                # The same configurations as the OCI notebook session will be used.
                DataScienceJob()
                .with_log_group_id("<log_group_ocid>")
                .with_log_id("<log_ocid>")
            )
            .with_runtime(
                NotebookRuntime()
                .with_notebook("path/to/notebook.ipynb")
                .with_service_conda(tensorflow28_p38_cpu_v1")
                # Saves the notebook with outputs to OCI object storage.
                .with_output("oci://bucket_name@namespace/path/to/dir")
            )
        ).create()
        # Run and monitor the job
        run = job.run().watch()
        # Download the notebook and outputs to local directory
        run.download(to_dir="path/to/local/dir/")


    See Also
    --------
    https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/jobs/index.html

    """

    _INFRASTRUCTURE_MAPPING = {
        **{item().type: item for item in [DataScienceJob]},
        "dataFlow": DataFlow,
    }
    _RUNTIME_MAPPING = {
        item().type: item
        for item in [
            PythonRuntime,
            GitPythonRuntime,
            ContainerRuntime,
            ScriptRuntime,
            NotebookRuntime,
            PyTorchDistributedRuntime,
            DataFlowRuntime,
        ]
    }

    @staticmethod
    def from_datascience_job(job_id) -> "Job":
        """Loads a data science job from OCI.

        Parameters
        ----------
        job_id : str
            OCID of an existing data science job.

        Returns
        -------
        Job
            A job instance.

        """
        dsc_infra = DataScienceJob.from_id(job_id)
        job = (
            Job(name=dsc_infra.name)
            .with_infrastructure(dsc_infra)
            .with_runtime(dsc_infra.runtime)
        )
        return job

    @staticmethod
    def datascience_job(compartment_id: str = None, **kwargs) -> List["DataScienceJob"]:
        """Lists the existing data science jobs in the compartment.

        Parameters
        ----------
        compartment_id : str
            The compartment ID for listing the jobs.
            This is optional if running in an OCI notebook session.
            The jobs in the same compartment of the notebook session will be returned.

        Returns
        -------
        list
            A list of Job objects.
        """
        return [
            Job(name=dsc_job.name)
            .with_infrastructure(dsc_job)
            .with_runtime(dsc_job.runtime)
            for dsc_job in DataScienceJob.list_jobs(compartment_id, **kwargs)
        ]

    @staticmethod
    def from_dataflow_job(job_id: str) -> "Job":
        """
        Create a Data Flow job given a job id.

        Parameters
        ----------
        job_id: str
            id of the job
        Returns
        -------
        Job
            a Job instance
        """
        df = DataFlow.from_id(job_id)
        job = Job(name=df.name).with_infrastructure(df).with_runtime(df.runtime)
        return job

    @staticmethod
    def dataflow_job(compartment_id: str = None, **kwargs) -> List["Job"]:
        """
        List data flow jobs under a given compartment.

        Parameters
        ----------
        compartment_id: str
            compartment id
        kwargs
            additional keyword arguments

        Returns
        -------
        List[Job]
            list of Job instances
        """
        return [
            Job(name=df.name).with_infrastructure(df).with_runtime(df.runtime)
            for df in DataFlow.list_jobs(compartment_id, **kwargs)
        ]

    def __init__(self, name: str = None, infrastructure=None, runtime=None) -> None:
        """Initializes a job.

        The infrastructure and runtime can be configured when initializing the job,
         or by calling with_infrastructure() and with_runtime().

        The infrastructure should be a subclass of ADS job Infrastructure, e.g., DataScienceJob, DataFlow.
        The runtime should be a subclass of ADS job Runtime, e.g., PythonRuntime, NotebookRuntime.

        Parameters
        ----------
        name : str, optional
            The name of the job, by default None.
            If it is None, a default name may be generated by the infrastructure,
            depending on the implementation of the infrastructure.
            For OCI data science job, the default name contains the job artifact name and a timestamp.
            If no artifact, a randomly generated easy to remember name with timestamp will be generated,
            like 'strange-spider-2022-08-17-23:55.02'.
        infrastructure : Infrastructure, optional
            Job infrastructure, by default None
        runtime : Runtime, optional
            Job runtime, by default None.

        """
        super().__init__()
        if name:
            self.set_spec("name", name)
        if infrastructure:
            self.with_infrastructure(infrastructure)
        if runtime:
            self.with_runtime(runtime)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            "job"
        """
        return "job"

    @property
    def id(self) -> str:
        """The ID of the job.
        For jobs running on OCI, this is the OCID.

        Returns
        -------
        str
            ID of the job.
        """
        if self.infrastructure and hasattr(self.infrastructure, "job_id"):
            return self.infrastructure.job_id
        return None

    @property
    def name(self) -> str:
        """The name of the job.
        For jobs running on OCI, this is the display name.

        Returns
        -------
        str
            The name of the job.
        """
        return self.get_spec("name")

    @property
    def infrastructure(self) -> Union[DataScienceJob, DataFlow]:
        """The job infrastructure.

        Returns
        -------
        Infrastructure
            Job infrastructure.
        """
        return self.get_spec("infrastructure")

    @property
    def runtime(self) -> Runtime:
        """The job runtime.

        Returns
        -------
        Runtime
            The job runtime
        """
        return self.get_spec("runtime")

    def with_infrastructure(self, infrastructure) -> "Job":
        """Sets the infrastructure for the job.

        Parameters
        ----------
        infrastructure : Infrastructure
            Job infrastructure.

        Returns
        -------
        Job
            The job instance (self)
        """
        return self.set_spec("infrastructure", infrastructure)

    def with_runtime(self, runtime) -> "Job":
        """Sets the runtime for the job.

        Parameters
        ----------
        runtime : Runtime
            Job runtime.

        Returns
        -------
        Job
            The job instance (self)
        """
        return self.set_spec("runtime", runtime)

    def with_name(self, name: str) -> "Job":
        """Sets the job name.

        Parameters
        ----------
        name : str
            Job name.

        Returns
        -------
        Job
            The job instance (self)
        """
        return self.set_spec("name", name)

    def build(self) -> "Job":
        """Load default values from the environment for the job infrastructure."""
        super().build()

        build_method = getattr(self.infrastructure, "build", None)
        if build_method and callable(build_method):
            build_method()
        else:
            raise NotImplementedError
        return self

    def create(self, **kwargs) -> "Job":
        """Creates the job on the infrastructure.

        Returns
        -------
        Job
            The job instance (self)
        """
        infra = self.get_spec("infrastructure")
        infra.name = self.name
        self.infrastructure.create(self.runtime, **kwargs)
        self.set_spec("name", self.infrastructure.name)
        return self

    def run(
        self,
        name=None,
        args=None,
        env_var=None,
        freeform_tags=None,
        defined_tags=None,
        wait=False,
        **kwargs
    ) -> Union[DataScienceJobRun, DataFlowRun]:
        """Runs the job.

        Parameters
        ----------
        name : str, optional
            Name of the job run, by default None.
            The infrastructure handles the naming of the job run.
            For data science job, if a name is not provided,
            a default name will be generated containing the job name and the timestamp of the run.
            If no artifact, a randomly generated easy to remember name
            with timestamp will be generated, like 'strange-spider-2022-08-17-23:55.02'.
        args : str, optional
            Command line arguments for the job run, by default None.
            This will override the configurations on the job.
            If this is None, the args from the job configuration will be used.
        env_var : dict, optional
            Additional environment variables for the job run, by default None
        freeform_tags : dict, optional
            Freeform tags for the job run, by default None
        defined_tags : dict, optional
            Defined tags for the job run, by default None
        wait : bool, optional
            Indicate if this method call should wait for the job run.
            By default False, this method returns as soon as the job run is created.
            If this is set to True, this method will stream the job logs and wait until it finishes,
            similar to `job.run().watch()`.
        kwargs
            additional keyword arguments

        Returns
        -------
        Job Run Instance
            A job run instance, depending on the infrastructure.

        Examples
        --------
        To run a job and override the configurations::

            job_run = job.run(
                name="<my_job_run_name>",
                args="new_arg --new_key new_val",
                env_var={"new_env": "new_val"},
                freeform_tags={"new_tag": "new_tag_val"},
                defined_tags={"Operations": {"CostCenter": "42"}}
            )

        """
        return self.infrastructure.run(
            name=name,
            args=args,
            env_var=env_var,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
            wait=wait,
            **kwargs
        )

    def run_list(self, **kwargs) -> list:
        """Gets a list of runs of the job.

        Returns
        -------
        list
            A list of job run instances, the actual object type depends on the infrastructure.
        """
        return self.infrastructure.run_list(**kwargs)

    def delete(self) -> None:
        """Deletes the job from the infrastructure."""
        self.infrastructure.delete()

    def status(self) -> str:
        """Status of the job

        Returns
        -------
        str
            Status of the job
        """
        return getattr(self.infrastructure, "status", None)

    def to_dict(self, **kwargs: Dict) -> Dict:
        """Serialize the job specifications to a dictionary.

        Parameters
        ----------
        **kwargs: Dict
            The additional arguments.
            - filter_by_attribute_map: bool
                If True, then in the result will be included only the fields
                presented in the `attribute_map`.

        Returns
        -------
        Dict
            A dictionary containing job specifications.
        """
        spec = {"name": self.name}
        if self.runtime:
            spec["runtime"] = self.runtime.to_dict(**kwargs)
        if self.infrastructure:
            spec["infrastructure"] = self.infrastructure.to_dict(**kwargs)
        if self.id:
            spec["id"] = self.id
        return {
            "kind": self.kind,
            # "apiVersion": self.api_version,
            "spec": spec,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "Job":
        """Initializes a job from a dictionary containing the configurations.

        Parameters
        ----------
        config : dict
            A dictionary containing the infrastructure and runtime specifications.

        Returns
        -------
        Job
            A job instance

        Raises
        ------
        NotImplementedError
            If the type of the infrastructure or runtime is not supported.
        """
        if not isinstance(config, dict):
            raise ValueError("The config data for initializing the job is invalid.")
        spec = config.get("spec")

        mappings = {
            "infrastructure": cls._INFRASTRUCTURE_MAPPING,
            "runtime": cls._RUNTIME_MAPPING,
        }
        job = cls()

        for key, value in spec.items():
            if key in mappings:
                mapping = mappings[key]
                child_config = value
                if child_config.get("type") not in mapping:
                    raise NotImplementedError(
                        f"{key.title()} type: {child_config.get('type')} is not supported."
                    )
                job.set_spec(
                    key, mapping[child_config.get("type")].from_dict(child_config)
                )
            else:
                job.set_spec(key, value)

        return job

    def download(self, to_dir: str, output_uri=None, **storage_options):
        """Downloads files from remote output URI to local.

        Parameters
        ----------
        to_dir : str
            Local directory to which the files will be downloaded to.

        output_uri : (str, optional). Default is None.
            The remote URI from which the files will be downloaded.
            Defaults to None.
            If output_uri is not specified, this method will try to get the output_uri from the runtime.

        storage_options :
            Extra keyword arguments for particular storage connection.
            This method uses fsspec to download the files from remote URI.
            storage_options will to be passed into fsspec.open_files().

        Returns
        -------
        Job
            The job instance (self)

        Raises
        ------
        AttributeError
            The output_uri is not specified and the runtime is not configured with output_uri.
        """
        if not output_uri and self.runtime:
            output_uri = getattr(self.runtime, "output_uri", None)

        if not output_uri:
            raise AttributeError(
                "Please specify the output_uri or set it with a compatible runtime."
            )

        scheme = urlparse(output_uri).scheme
        if scheme == "oci" and not storage_options:
            storage_options = default_signer()
        fs = fsspec.filesystem(scheme, **storage_options)
        fs.get(self.runtime.output_uri, to_dir, recursive=True)
        return self
