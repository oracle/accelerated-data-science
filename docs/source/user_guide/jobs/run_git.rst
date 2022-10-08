Run a Git Repo
**************

The ADS ``GitPythonRuntime`` class allows you to run source code from a Git repository as a Data Science job. The next example shows how to run a
`PyTorch Neural Network Example to train third order polynomial predicting y=sin(x) <https://github.com/pytorch/tutorials/blob/master/beginner_source/examples_nn/polynomial_nn.py>`__.

Python
======

To configure the ``GitPythonRuntime``, you must specify the source code ``url`` and entrypoint ``path``. Similar to ``PythonRuntime``, you can specify a service conda environment, environment variables, and CLI arguments. In this example, the ``pytorch19_p37_gpu_v1`` service conda environment is used.  Assuming you are running this example in an Data Science notebook session, only log ID and log group ID need to be configured for the ``DataScienceJob`` object, see `Data Science Jobs <data_science_job.html>`__ for more details about configuring the infrastructure.

.. code-block:: python3

  from ads.jobs import Job, DataScienceJob, GitPythonRuntime

  job = (
    Job()
    .with_infrastructure(
      DataScienceJob()
      .with_log_group_id("<log_group_ocid>")
      .with_log_id("<log_ocid>")
      # The following infrastructure configurations are optional
      # if you are in an OCI data science notebook session.
      # The configurations of the notebook session will be used as defaults
      .with_compartment_id("<compartment_ocid>")
      .with_project_id("<project_ocid>")
      .with_subnet_id("<subnet_ocid>")
      .with_shape_name("VM.Standard.E3.Flex")
      .with_shape_config_details(memory_in_gbs=16, ocpus=1) # Applicable only for the flexible shapes
      .with_block_storage_size(50)
    )
    .with_runtime(
      GitPythonRuntime()
      .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
      .with_service_conda("pytorch19_p37_gpu_v1")
      .with_source("https://github.com/pytorch/tutorials.git")
      .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
      .with_output(
        output_dir="~/Code/tutorials/beginner_source/examples_nn",
        output_uri="oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX"
      )
    )
  )

  # Create the job with OCI
  job.create()
  # Run the job and stream the outputs
  job_run = job.run().watch()


The default branch from the Git repository is used unless you specify a different ``branch`` or ``commit`` in the ``.with_source()`` method.

For a public repository, we recommend the "http://" or "https://" URL.  Authentication may be required for the SSH URL even if the repository is
public.

To use a private repository, you must first save an SSH key to an `OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`__ as a secret, and provide the ``secret_ocid`` to the ``with_source()`` method, see `Managing Secret with Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Tasks/managingsecrets.htm>`__.  For example, you could use `GitHub Deploy
Key <https://docs.github.com/en/developers/overview/managing-deploy-keys#deploy-keys>`__.

The entry point specifies how the source code is invoked.  The ``.with_entrypiont()`` has the following arguments:

* ``func``: Optional. The function in the script specified by ``path`` to call. If you don't specify it, then the script specified by ``path`` is run as a Python script in a subprocess.
* ``path``: Required. The relative path for the script, module, or file to start the job.

With the ``GitPythonRuntime`` class, you can save the output files from the job run to Object Storage using ``with_output()``. By default, the source code is cloned to the ``~/Code`` directory. In the example, the files in the ``example_nn`` directory are copied to the Object Storage specified by the ``output_uri`` parameter. The ``output_uri`` parameter should have this format:

``oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX``

The ``GitPythonRuntime`` also supports these additional configurations:

* The ``.with_python_path()`` method allows you to add additional Python paths to the runtime. By default, the code directory checked out from Git is added to ``sys.path``. Additional Python paths are appended before the code directory is appended.
* The ``.with_argument()`` method allows you to pass arguments to invoke the script or function. For running a script, the arguments are passed in as CLI arguments. For running a function, the ``list`` and ``dict`` JSON serializable objects are supported and are passed into the function.

The ``GitPythonRuntime`` method updates metadata in the free form tags of the job run after the job run finishes. The following tags are added automatically:

* ``commit``: The Git commit ID.
* ``method``: The entry function or method.
* ``module``: The entry script or module.
* ``outputs``: The prefix of the output files in Object Storage.
* ``repo``: The URL of the Git repository.

The new values overwrite any existing tags. If you want to skip the metadata update, set ``skip_metadata_update`` to ``True`` when initializing the runtime:

.. code-block:: python3

  runtime = GitPythonRuntime(skip_metadata_update=True)

YAML
====

You could create the preceding example job with the following YAML file:

.. code-block:: yaml

  kind: job
  spec:
    infrastructure:
      kind: infrastructure
      type: dataScienceJob
      spec:
        logGroupId: <log_group_ocid>
        logId: <log_ocid>
        compartmentId: <compartment_ocid>
        projectId: <project_ocid>
        subnetId: <subnet_ocid>
        shapeName: VM.Standard.E3.Flex
        shapeConfigDetails:
          memoryInGBs: 16
          ocpus: 1
        blockStorageSize: 50
    name: git_example
    runtime:
      kind: runtime
      type: gitPython
      spec:
        entrypoint: beginner_source/examples_nn/polynomial_nn.py
        outputDir: ~/Code/tutorials/beginner_source/examples_nn
        outputUri: oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX
        url: https://github.com/pytorch/tutorials.git
        conda:
          slug: pytorch19_p37_gpu_v1
          type: service
        env:
          - name: GREETINGS
            value: Welcome to OCI Data Science

**GitPythonRuntime YAML Schema**

.. code-block:: yaml

  kind:
    required: true
    type: string
    allowed:
      - runtime
  type:
    required: true
    type: string
    allowed:
      - gitPython
  spec:
    required: true
    type: dict
    schema:
      args:
        type: list
        nullable: true
        required: false
        schema:
          type: string
      branch:
        nullable: true
        required: false
        type: string
      commit:
        nullable: true
        required: false
        type: string
      codeDir:
        required: false
        type: string
      conda:
        nullable: false
        required: false
        type: dict
        schema:
          slug:
            required: true
            type: string
          type:
            required: true
            type: string
            allowed:
              - service
      entryFunction:
        nullable: true
        required: false
        type: string
      entrypoint:
        required: false
        type:
          - string
          - list
      env:
        nullable: true
        required: false
        type: list
        schema:
          type: dict
          schema:
            name:
              type: string
            value:
              type:
              - number
              - string
      outputDir:
        required: false
        type: string
      outputUri:
        required: false
        type: string
      pythonPath:
        nullable: true
        required: false
        type: list
      url:
        required: false
        type: string

