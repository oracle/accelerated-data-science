Infrastructure and Runtime
**************************

This page describes the configurations of **Infrastructure** and **Runtime** defining a Data Science Job.


Example
=======

The following example configures the infrastructure and runtime to run a Python script.

.. include:: ../jobs/tabs/quick_start_job.rst

Infrastructure
==============

The Data Science Job infrastructure is defined by a :py:class:`~ads.jobs.DataScienceJob` instance.
For example:

.. include:: ../jobs/tabs/infra_config.rst

When creating a :py:class:`~ads.jobs.DataScienceJob` instance, the following configurations are required:

* Compartment ID
* Project ID
* Compute Shape

The following configurations are optional:

* Block Storage Size, defaults to 50 (GB)
* Log Group ID
* Log ID

For more details about the mandatory and optional parameters, see :py:class:`~ads.jobs.DataScienceJob`.

Using Configurations from Notebook
----------------------------------

If you are creating a job from an OCI Data Science
`Notebook Session <https://docs.oracle.com/en-us/iaas/data-science/using/manage-notebook-sessions.htm>`_,
the same infrastructure configurations from the notebook session will be used as defaults, including:

* Compartment ID
* Project ID
* Subnet ID
* Compute Shape
* Block Storage Size

You can initialize the :py:class:`~ads.jobs.DataScienceJob`
with the logging configurations and override the other options as needed. For example:

.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        # Use a GPU shape for the job,
        # regardless of the shape used by the notebook session
        .with_shape_name("VM.GPU3.1")
        # compartment ID, project ID, subnet ID and block storage will be
        # the same as the ones set in the notebook session
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: infrastructure
    type: dataScienceJob
    spec:
      logGroupId: <log_group_ocid>
      logId: <log_ocid>
      shapeName: VM.GPU3.1


Compute Shapes
--------------

The :py:class:`~ads.jobs.DataScienceJob` class provides two static methods to obtain the support compute shapes:

* You can get a list of currently supported compute shapes by calling
  :py:meth:`~ads.jobs.DataScienceJob.instance_shapes`.
* can get a list of shapes that are available for fast launch by calling
  :py:meth:`~ads.jobs.DataScienceJob.fast_launch_shapes`.
  Specifying a fast launch shape will allow your job to start as fast as possible.

Networking
----------

Data Science Job offers two types of networking: default networking (managed egress) and custom networking.
Default networking allows job runs to access public internet through a NAT gateway and OCI service through
a service gateway, both are configured automatically. Custom networking requires you to specify a subnet ID.
You can control the network access through the subnet and security lists.

If you specified a subnet ID, your job will be configured to have custom networking.
Otherwise, default networking will be used.

Note that when you are in a Data Science Notebook Session,
the same networking configuration is be used by default.
You can specify the networking manually by calling :py:meth:`~ads.jobs.DataScienceJob.with_job_infrastructure_type()`.
For example, if you are using custom networking in the notebook session
but you would like to use default networking for the job:

.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        # Use default networking,
        # regardless of the networking used by the notebook session
        .with_job_infrastructure_type("ME_STANDALONE")
        # compartment ID, project ID, compute shape and block storage will be
        # the same as the ones set in the notebook session
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: infrastructure
    type: dataScienceJob
    spec:
      jobInfrastructureType: ME_STANDALONE
      logGroupId: <log_group_ocid>
      logId: <log_ocid>


Logging
-------

Logging is not required to create the job.
However, it is highly recommended to enable logging for debugging and monitoring.

In the preceding example, both the log OCID and corresponding log group OCID are specified
with the :py:class:`~ads.jobs.DataScienceJob` instance.
If your administrator configured the permission for you to search for logging resources,
you can skip specifying the log group OCID because ADS can automatically retrieve it.

If you specify only the log group OCID and no log OCID,
a new Log resource is automatically created within the log group to store the logs,
see also `ADS Logging <../logging/logging.html>`_.

With logging configured, you can call :py:meth:`~ads.jobs.DataScienceJobRun.watch` method to stream the logs.

Mounting File Systems
---------------------

Data Science Job supports mounting multiple types of file systems,
see `Data Science Job Mounting File Systems <place_holder>`_. A maximum number of 5 file systems are
allowed to be mounted for each Data Science Job. You can specify a list of file systems to be mounted
by calling :py:meth:`~ads.jobs.DataScienceJob.with_storage_mount()`. For each file system to be mounted,
you need to pass a dictionary with `src` and `dest` as keys. For example, you can pass
*<mount_target_ip_address>@<export_path>* as the value for `src` to mount OCI File Storage. The value of
`dest` must be the folder to which you want to mount the file system. See example below.

.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        .with_storage_mount(
          {
            "src" : "<mount_target_ip_address>@<export_path>",
            "dest" : "<destination_directory_name>"
          }
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: infrastructure
    type: dataScienceJob
    spec:
      logGroupId: <log_group_ocid>
      logId: <log_ocid>
      storageMount:
      - src: <mount_target_ip_address>@<export_path>
        dest: <destination_directory_name>

Runtime
=======

The *runtime* of a job defines the source code of your workload, environment variables, CLI arguments
and other configurations for the environment to run the workload.

Depending on the source code, ADS provides different types of *runtime* for defining a data science job,
including:

.. include:: ../jobs/runtime_types.rst


Environment Variables
---------------------

You can set environment variables for a runtime by calling
:py:meth:`~ads.jobs.PythonRuntime.with_environment_variable()`.
Environment variables enclosed by ``${...}`` will be substituted. For example:

.. include:: ../jobs/tabs/runtime_envs.rst

.. code-block:: python

  for k, v in runtime.environment_variables.items():
      print(f"{k}: {v}")

will show the following environment variables for the runtime:

.. code-block:: text

  HOST: 10.0.0.1
  PORT: 443
  URL: http://10.0.0.1:443/path/
  ESCAPED_URL: http://${HOST}:${PORT}/path/
  MISSING_VAR: This is ${UNDEFINED}
  VAR_WITH_DOLLAR: $10
  DOUBLE_DOLLAR: $10

Note that:

* You can use ``$$`` to escape the substitution.
* Undefined variable enclosed by ``${...}`` will be ignored.
* Double dollar signs ``$$`` will be substituted by a single one ``$``.

See also:
`Service Provided Environment Variables <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-env-vars.htm>`_


.. _runtime_args:

Command Line Arguments
----------------------

The command line arguments for running your script or function can be configured by calling
:py:meth:`~ads.jobs.PythonRuntime.with_argument()`. For example:

.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_argument(
            "arg1", "arg2",
            key1="val1",
            key2="val2"
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: runtime
    type: python
    spec:
      scriptPathURI: oci://bucket_name@namespace/path/to/script.py
      args:
      - arg1
      - arg2
      - --key1
      - val1
      - --key2
      - val2

will configured the job to call your script by:

.. code-block:: bash

  python script.py arg1 arg2 --key1 val1 --key2 val2

You can call :py:meth:`~ads.jobs.PythonRuntime.with_argument()` multiple times to set the arguments
to your desired order. You can check ``runtime.args`` to see the added arguments.

Here are a few more examples:

.. include:: ../jobs/tabs/runtime_args.rst

Conda Environment
-----------------

You can configure a
`Conda Environment <https://docs.oracle.com/en-us/iaas/data-science/using/conda_understand_environments.htm>`_
for running your workload. You can use the slug name to specify a
`conda environment provided by the data science service
<https://docs.oracle.com/en-us/iaas/data-science/using/conda_viewing.htm#conda-dsenvironments>`_.
For example, to use the TensorFlow conda environment:

.. include:: ../jobs/tabs/runtime_service_conda.rst

You can also use a custom conda environment published to OCI Object Storage
by passing the ``uri`` to :py:meth:`~ads.jobs.PythonRuntime.with_custom_conda`,
for example:

.. include:: ../jobs/tabs/runtime_custom_conda.rst

By default, ADS will try to determine the region based on the authenticated API key or resource principal.
If your custom conda environment is stored in a different region,
you can specify the ``region`` when calling :py:meth:`~ads.jobs.PythonRuntime.with_custom_conda`.

For more details on custom conda environment, see
`Publishing a Conda Environment to an Object Storage Bucket in Your Tenancy
<https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm>`__.


Override Configurations
-----------------------

When you call :py:meth:`ads.jobs.Job.run`, a new job run will be started with the configuration defined in the **job**.
You may want to override the configuration with custom variables. For example,
you can customize job run display name, override command line argument, specify additional environment variables,
and add free form tags:

.. code-block:: python3

  job_run = job.run(
      name="<my_job_run_name>",
      args="new_arg --new_key new_val",
      env_var={"new_env": "new_val"},
      freeform_tags={"new_tag": "new_tag_val"}
  )
