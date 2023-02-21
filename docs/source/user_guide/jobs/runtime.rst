Runtime
*******

The *runtime* of a job defines the source code of your workload, environment variables, CLI arguments
and other configurations for the environment to run the workload.

Depending on the source code, ADS provides different types of *runtime* for defining a data science job,
including:

* :py:class:`~ads.jobs.builders.runtimes.python_runtime.PythonRuntime`
  for Python code stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_
* :py:class:`~ads.jobs.builders.runtimes.python_runtime.GitPythonRuntime`
  for Python code from a Git repository.
* :py:class:`~ads.jobs.builders.runtimes.python_runtime.NotebookRuntime`
  for a single Jupyter notebook stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_
* :py:class:`~ads.jobs.builders.runtimes.python_runtime.ScriptRuntime`
  for bash or shell scripts stored locally, OCI object storage, or other remote location supported by
  `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_
* :py:class:`~ads.jobs.builders.runtimes.python_runtime.ContainerRuntime` for container images.


Conda Environment
=================

Except for :py:class:`~ads.jobs.builders.runtimes.python_runtime.ContainerRuntime`,
all the other runtime options allow you to configure a
`Conda Environment <https://docs.oracle.com/en-us/iaas/data-science/using/conda_understand_environments.htm>`_
for your workload. You can use the slug name to specify a
`conda environment provided by the data science service <https://docs.oracle.com/en-us/iaas/data-science/using/conda_viewing.htm#conda-dsenvironments>`_.
For example, to use the TensorFlow conda environment:

.. code-block:: python3

    from ads.jobs import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        # Use slug name for conda environment provided by data science service
        .with_service_conda("tensorflow28_p38_cpu_v1")
    )

You can also use a custom conda environment published to OCI Object Storage
by passing the ``uri`` to the ``with_custom_conda()`` method, for example:

.. code-block:: python3

    runtime = (
        ScriptRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_custom_conda("oci://bucket@namespace/conda_pack/pack_name")
    )

For more details on custom conda environment, see
`Publishing a Conda Environment to an Object Storage Bucket in Your Tenancy <https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm>`__.


