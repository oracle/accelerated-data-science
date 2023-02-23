Runtime
*******

The *runtime* of a job defines the source code of your workload, environment variables, CLI arguments
and other configurations for the environment to run the workload.

.. include:: ../jobs/components/toc_local.rst

Depending on the source code, ADS provides different types of *runtime* for defining a data science job,
including:

.. include:: ../jobs/components/runtime_types.rst


Environment Variables
=====================

You can set environment variables for a runtime by calling
:py:meth:`~ads.jobs.PythonRuntime.with_environment_variable()`.
Environment variables enclosed by ``${...}`` will be substituted. For example:

.. code-block:: python

  from ads.jobs import PythonRuntime

  runtime = (
      PythonRuntime()
      .with_environment_variable(
          HOST="10.0.0.1",
          PORT="443",
          URL="http://${HOST}:${PORT}/path/",
          ESCAPED_URL="http://$${HOST}:$${PORT}/path/",
          MISSING_VAR="This is ${UNDEFINED}",
          VAR_WITH_DOLLAR="$10",
          DOUBLE_DOLLAR="$$10"
      )
  )

  for k, v in runtime.environment_variables.items():
      print(f"{k}: {v}")

will show the following environment variables for the runtime:

.. code-block:: text

  HOST: 10.0.0.1
  PORT: 443
  URL: http://10.0.0.1:443/path/
  ESCAPED_URL: http://${HOST}:${PORT}/path/
  MISSING_VAR: This is This is ${UNDEFINED}
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
======================

The command line arguments for running your script or function can be configured by calling
:py:meth:`~ads.jobs.PythonRuntime.with_argument()`. For example:

.. code-block:: python

  from ads.jobs import PythonRuntime

  runtime = (
      PythonRuntime()
      .with_source("oci://bucket_name@namespace/path/to/script.py")
      .with_arguments(
          "arg1", "arg2",
          key1="val1",
          key2="val2"
      )
  )

will configured the job to call your script by:

.. code-block:: bash

  python script.py arg1 arg2 --key1 val1 --key2 val2

You can call :py:meth:`~ads.jobs.PythonRuntime.with_argument()` multiple times to set the arguments
to your desired order. You can check ``runtime.args`` to see the added arguments.

Here are a few more examples:

.. code-block:: python

  runtime = PythonRuntime().with_argument(key1="val1", key2="val2").with_argument("pos1")
  print(runtime.args)
  # ["--key1", "val1", "--key2", "val2", "pos1"]

  runtime = PythonRuntime()
  runtime.with_argument("pos1")
  runtime.with_argument(key1="val1", key2="val2.1 val2.2")
  runtime.with_argument("pos2")
  print(runtime.args)
  # ['pos1', '--key1', 'val1', '--key2', 'val2.1 val2.2', 'pos2']

  runtime = PythonRuntime()
  runtime.with_argument("pos1")
  runtime.with_argument(key1=None, key2="val2")
  runtime.with_argument("pos2")
  print(runtime.args)
  # ["pos1", "--key1", "--key2", "val2", "pos2"]

Conda Environment
=================

Except for :py:class:`~ads.jobs.ContainerRuntime`,
all the other runtime options allow you to configure a
`Conda Environment <https://docs.oracle.com/en-us/iaas/data-science/using/conda_understand_environments.htm>`_
for your workload. You can use the slug name to specify a
`conda environment provided by the data science service
<https://docs.oracle.com/en-us/iaas/data-science/using/conda_viewing.htm#conda-dsenvironments>`_.
For example, to use the TensorFlow conda environment:

.. code-block:: python

  from ads.jobs import PythonRuntime

  runtime = (
    PythonRuntime()
    .with_source("oci://bucket_name@namespace/path/to/script.py")
    # Use slug name for conda environment provided by data science service
    .with_service_conda("tensorflow28_p38_cpu_v1")
  )

You can also use a custom conda environment published to OCI Object Storage
by passing the ``uri`` to :py:meth:`~ads.jobs.PythonRuntime.with_custom_conda`,
for example:

.. code-block:: python

    runtime = (
        PythonRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_custom_conda("oci://bucket@namespace/conda_pack/pack_name")
    )

By default, ADS will try to determine the region based on the authenticated API key or resource principal.
If your custom conda environment is stored in a different region,
you can specify the ``region`` when calling :py:meth:`~ads.jobs.PythonRuntime.with_custom_conda`.

For more details on custom conda environment, see
`Publishing a Conda Environment to an Object Storage Bucket in Your Tenancy
<https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm>`__.


Override Configurations
=======================

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
