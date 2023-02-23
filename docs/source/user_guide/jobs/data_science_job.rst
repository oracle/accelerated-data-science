Quick Start
***********

.. admonition:: OCI Data Science Policies

  Before creating a job, ensure that you have policies configured for Data Science resources.
  See also: :doc:`policies` and  `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`_.

.. include:: ../jobs/components/toc_local.rst

Create and Run a Job
====================

In ADS, a job is defined by :doc:`infrastructure` and :doc:`runtime`.
The Data Science Job infrastructure is configured through a :py:class:`~ads.jobs.DataScienceJob` instance.
The runtime can be an instance of:

.. include:: ../jobs/components/runtime_types.rst

Here is an example to define and run a Python :py:class:`~ads.jobs.Job`:

.. include:: ../jobs/tabs/python_runtime.rst

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

A job can be defined using YAML, as shown in the "YAML" tab.
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
See also :ref:`Saving Outputs <_runtime_outputs>`
