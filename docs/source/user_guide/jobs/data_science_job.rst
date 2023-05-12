Quick Start
***********

.. admonition:: Prerequisite

  Before creating a job, ensure that you have policies configured for Data Science resources.

  See :doc:`policies` and  `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_.


Define a Job
============

In ADS, a job is defined by :doc:`infra_and_runtime`.
The Data Science Job infrastructure is configured through a :py:class:`~ads.jobs.DataScienceJob` instance.
The runtime can be an instance of:

.. include:: ../jobs/runtime_types.rst

Here is an example to define and run a Python :py:class:`~ads.jobs.Job`.

Note that a job can be defined either using Python APIs or YAML.
See the next section for how to load and save the job with YAML.

.. include:: ../jobs/tabs/quick_start_job.rst

The :py:class:`~ads.jobs.PythonRuntime` is designed for :doc:`Running a Python Workload <run_python>`.
The source code is specified by :py:meth:`~ads.jobs.PythonRuntime.with_source` (``path/to/script.py``).
It can be a script, a Jupyter notebook, a folder or a zip file.
The source code location can be a local or remote, including HTTP URL and OCI Object Storage.
An `example Python script <https://github.com/oracle-samples/oci-data-science-ai-samples/blob/master/jobs/python/job%2Bsamples/greeting-env-cmd.py>`_
is available on `Data Science AI Sample GitHub Repository <https://github.com/oracle-samples/oci-data-science-ai-samples>`_.

For more details, see :doc:`infra_and_runtime` configurations.
You can also :doc:`run_notebook`, :doc:`run_script` and :doc:`run_git`.


YAML
====

A job can be defined using YAML, as shown in the "YAML" tab in the example above.
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
  """)

The ``uri`` can be a local file path or a remote location supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_, including OCI object storage.

With the YAML file, you can create and run the job with ADS CLI:

.. code-block:: bash

  ads opctl run -f your_job.yaml

For more details on ``ads opctl``, see :doc:`../cli/opctl/_template/jobs`.

The job infrastructure, runtime and job run also support YAML serialization/deserialization.


Run a Job and Monitor outputs
=============================

Once the job is defined or loaded from YAML, you can call the :py:meth:`~ads.jobs.Job.create` method
to create the job on OCI. To start a job run, you can call the :py:meth:`~ads.jobs.Job.run` method,
which returns a :py:class:`~ads.jobs.DataScienceJobRun` instance.
Once the job or job run is created, the job OCID can be accessed through ``job.id`` or ``run.id``.

.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()

The :py:meth:`~ads.jobs.DataScienceJobRun.watch` method is useful to monitor the progress of the job run.
It will stream the logs to terminal and return once the job is finished.
Logging configurations are required for this method to show logs. Here is an example of the logs:

.. code-block:: text

  Job OCID: <job_ocid>
  Job Run OCID: <job_run_ocid>
  2023-02-27 15:58:01 - Job Run ACCEPTED
  2023-02-27 15:58:11 - Job Run ACCEPTED, Infrastructure provisioning.
  2023-02-27 15:59:06 - Job Run ACCEPTED, Infrastructure provisioned.
  2023-02-27 15:59:29 - Job Run ACCEPTED, Job run bootstrap starting.
  2023-02-27 16:01:08 - Job Run ACCEPTED, Job run bootstrap complete. Artifact execution starting.
  2023-02-27 16:01:18 - Job Run IN_PROGRESS, Job run artifact execution in progress.
  2023-02-27 16:01:11 - Good morning, your environment variable has value of (Welcome to OCI Data Science.)
  2023-02-27 16:01:11 - Job Run 02-27-2023-16:01:11
  2023-02-27 16:01:11 - Job Done.
  2023-02-27 16:01:22 - Job Run SUCCEEDED, Job run artifact execution succeeded. Infrastructure de-provisioning.


Load Existing Job or Job Run
============================

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

  # Get a list of jobs in a specific compartment.
  jobs = Job.datascience_job("<compartment_ocid>")

With a ``Job`` object, you can get a list of job runs:

.. code-block:: python

  # Gets a list of job runs for a specific job.
  runs = job.run_list()

Delete a Job or Job Run
=======================

You can delete a job or job run by calling the ``delete()`` method.

.. code-block:: python

  # Delete a job and the corresponding job runs.
  job.delete()
  # Delete a job run
  run.delete()

You can also cancel a job run:

.. code-block:: python

  run.cancel()


Variable Substitution
=====================

When defining a job or starting a job run,
you can use environment variable substitution for the names and ``output_uri`` argument of
the :py:meth:`~ads.jobs.PythonRuntime.with_output` method.

For example, the following job specifies the name based on the environment variable ``DATASET_NAME``,
and ``output_uri`` based on the environment variables ``JOB_RUN_OCID``:

.. include:: ../jobs/tabs/name_substitution.rst

Note that ``JOB_RUN_OCID`` is an environment variable provided by the service after the job run is created.
It is available for the ``output_uri`` but cannot be used in the job name.

See also:

* :ref:`Saving Outputs <runtime_outputs>`
* `Service Provided Environment Variables <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-env-vars.htm>`_
