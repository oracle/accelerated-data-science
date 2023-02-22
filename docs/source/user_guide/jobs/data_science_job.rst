Quick Start
***********

.. admonition:: OCI Data Science Policies

  Before creating a job, ensure that you have policies configured for Data Science resources.
  See also: :doc:`policies` and  `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_.

In ADS, a job is defined by :doc:`infrastructure` and :doc:`runtime`.
The Data Science Job infrastructure is configured through a :py:class:`~ads.jobs.DataScienceJob` instance.
The runtime can be an instance of :py:class:`~ads.jobs.PythonRuntime`,
:py:class:`~ads.jobs.GitPythonRuntime`,
:py:class:`~ads.jobs.NotebookRuntime`,
:py:class:`~ads.jobs.ScriptRuntime`, or
:py:class:`~ads.jobs.ContainerRuntime`


Create and Run a Job
====================

Here is an example to define and run a Python :py:class:`~ads.jobs.Job`:

.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, PythonRuntime

    job = (
      Job(name="My Job")
      .with_infrastructure(
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        # The following infrastructure configurations are optional
        # if you are in an OCI data science notebook session.
        # The configurations of the notebook session will be used as defaults.
        .with_compartment_id("<compartment_ocid>")
        .with_project_id("<project_ocid>")
        # For default networking, no need to specify subnet ID
        .with_subnet_id("<subnet_ocid>")
        .with_shape_name("VM.Standard.E3.Flex")
        # Shape config details are applicable only for the flexible shapes.
        .with_shape_config_details(memory_in_gbs=16, ocpus=1)
        .with_block_storage_size(50)
      )
      .with_runtime(
        PythonRuntime()
        # Specify the service conda environment by slug name.
        .with_service_conda("pytorch19_p37_cpu_v1")
        # The job artifact can be a single Python script, a directory or a zip file.
        .with_source("local/path/to/code_dir")
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
      )
    )

    # Create the job on OCI Data Science
    job.create()
    # Start a job run
    run = job.run()
    # Stream the job run outputs
    run.watch()

  .. code-tab:: yaml
    :caption: YAML

    kind: job
    spec:
      name: "My Job"
      infrastructure:
        kind: infrastructure
        type: dataScienceJob
        spec:
          blockStorageSize: 50
          compartmentId: <compartment_ocid>
          jobInfrastructureType: STANDALONE
          jobType: DEFAULT
          logGroupId: <log_group_ocid>
          logId: <log_ocid>
          projectId: <project_ocid>
          shapeConfigDetails:
            memoryInGBs: 16
            ocpus: 1
          shapeName: VM.Standard.E3.Flex
          subnetId: <subnet_ocid>
      runtime:
        kind: runtime
        type: python
        spec:
          conda:
            slug: pytorch19_p37_cpu_v1
            type: service
          entrypoint: my_package/my_script.py
          outputDir: output
          outputUri: oci://bucket_name@namespace/path/to/dir
          pythonPath:
          - other_packages
          scriptPathURI: local/path/to/code_dir
          workingDir: code_dir

For more details, see :doc:`infrastructure` configurations and see :doc:`runtime` configurations.

In :py:class:`~ads.jobs.PythonRuntime`,
the ``entrypoint`` can be a Python script, a Python function or a Jupyter notebook.

Once the job is created, the job OCID can be accessed through ``job.id``.
Once the job run is created, the job run OCID can be accessed through ``run.id``.

The :py:meth:`~ads.jobs.DataScienceJobRun.watch` method is useful to monitor the progress of the job run.
It will stream the logs to terminal and return once the job is finished.
Logging configurations are required for this method to show logs.

Here is an example of the logs:

.. code-block:: text

    2021-10-28 17:17:58 - Job Run ACCEPTED
    2021-10-28 17:18:07 - Job Run ACCEPTED, Infrastructure provisioning.
    2021-10-28 17:19:19 - Job Run ACCEPTED, Infrastructure provisioned.
    2021-10-28 17:20:48 - Job Run ACCEPTED, Job run bootstrap starting.
    2021-10-28 17:23:41 - Job Run ACCEPTED, Job run bootstrap complete. Artifact execution starting.
    2021-10-28 17:23:50 - Job Run IN_PROGRESS, Job run artifact execution in progress.
    2021-10-28 17:23:50 - <Log Message>
    2021-10-28 17:23:50 - <Log Message>
    2021-10-28 17:23:50 - ...


YAML
====

A job can also be defined using YAML, as shown in the "YAML" tab.
Here are some examples to load/save the YAML job configurations:

.. code-block:: python

  # Load a job from a YAML file
  job = Job.from_yaml(uri="oci://bucket_name@namespace/path/to/job.yaml")
  # Save a job to a YAML file
  job.to_yaml(uri="oci://bucket_name@namespace/path/to/job.yaml")

  # Save a job to YAML in a string
  yaml_string = job.to_yaml()

  # Load a job from a YAML string
  job = Job.from_yaml("""
  kind: job
  spec:
    infrastructure:
    kind: infrastructure
      ...
  """")

The ``uri`` can be a local file path or a remote location supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_, including OCI object storage.

With the YAML file, you can create and run the job with ADS CLI:

.. code-block:: bash

  ads opctl run -f your_job.yaml

For more details on ``ads opctl``, see :doc:`../cli/opctl/_template/jobs`.


Loading Existing Job or Job Run
===============================

You can load an existing job or job run using the OCID from OCI:

.. code-block:: python

  from ads.jobs import Job, DataScienceJobRun

  # Load a job
  job = Job.from_datascience_job("<job_ocid>")

  # Load a job run
  job_run = DataScienceJobRun.from_ocid("<job_run_ocid>"")


List Existing Jobs or Job Runs
==============================

To get a list of existing jobs in a specific compartment:

.. code-block:: python

  from ads.jobs import Job

  # Load a job
  jobs = Job.datascience_job("<compartment_ocid>")

With a ``Job`` object, you can get a list of job runs:

.. code-block:: python

  # Gets a list of job runs for a specific job.
  runs = job.run_list()

Deleting a Job or Job Run
=========================

You can delete a job or job run by calling the ``delete()`` method.

.. code-block:: python

  # Delete a job and the corresponding job runs.
  job.delete()
  # Delete a job run
  run.delete()

You can also cancel a job run:

.. code-block:: python

  run.cancel()
