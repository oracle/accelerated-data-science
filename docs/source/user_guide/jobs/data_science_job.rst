Data Science Job
****************

This section shows how you can use the ADS jobs APIs to run OCI Data Science jobs.  You can use similar APIs to `Run a OCI DataFlow Application <run_data_flow.html>`__.

Before creating a job, ensure that you have policies configured for Data Science resources, see `About Data Science Policies <https://docs.oracle.com/en-us/iaas/data-science/using/policies.htm>`__.

Infrastructure
==============

The Data Science job infrastructure is defined by a ``DataScienceJob`` instance.  When creating a job, you specify the compartment ID, project ID, subnet ID, Compute shape, Block Storage size, log group ID, and log ID in the ``DataScienceJob`` instance.  For example:

.. code-block:: python3

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_compartment_id("<compartment_ocid>")
        .with_project_id("<project_ocid>")
        .with_subnet_id("<subnet_ocid>")
        .with_shape_name("VM.Standard.E3.Flex")
        .with_shape_config_details(memory_in_gbs=16, ocpus=1) # Applicable only for the flexible shapes
        .with_block_storage_size(50)
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
    )

If you are using these API calls in a Data Science `Notebook Session <https://docs.oracle.com/en-us/iaas/data-science/using/manage-notebook-sessions.htm>`__, and you want to use the same infrastructure configurations as the notebook session, you can initialize the ``DataScienceJob`` with only the logging configurations:

.. code-block:: python3

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
    )

In some cases, you may want to override the shape and block storage size.  For example, if you are testing your code in a CPU notebook session, but want to run the job in a GPU VM:

.. code-block:: python3

    from ads.jobs import DataScienceJob

    infrastructure = (
        DataScienceJob()
        .with_shape_name("VM.GPU2.1")
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
    )

Data Science jobs support the following shapes:

====================  ==========  ===========
Shape Name            Core Count  Memory (GB)
====================  ==========  ===========
VM.Optimized3.Flex    18          256
VM.Standard3.Flex     32          512
VM.Standard.E4.Flex   16          1024
VM.Standard2.1        1           15
VM.Standard2.2        2           30
VM.Standard2.4        4           60
VM.Standard2.8        8           120
VM.Standard2.16       16          240
VM.Standard2.24       24          320
VM.GPU2.1             12          72
VM.GPU3.1             6           90
VM.GPU3.2             12          180
VM.GPU3.4             24          360
====================  ==========  ===========

You can get a list of currently supported shapes by calling ``DataScienceJob.instance_shapes()``.

Logs
====

In the preceding examples, both the log OCID and corresponding log group OCID are specified in the ``DataScienceJob`` instance.  If your administrator configured the permission for you to search for logging resources, you can skip specifying the log group OCID because ADS automatically retrieves it.

If you specify only the log group OCID and no log OCID, a new Log resource is automatically created within the log group to store the logs, see `ADS Logging <../logging/logging.html>`__.

Runtime
=======

A job can have different types of *runtime* depending on the source code you want to run:

* ``ScriptRuntime`` allows you to run Python, Bash, and Java scripts from a single source file (``.zip`` or ``.tar.gz``) or code directory, see `Run a Script <run_script.html>`__ and `Run a ZIP file or folder <run_zip.html>`__.
* ``PythonRuntime`` allows you to run Python code with additional options, including setting a working directory, adding python paths, and copying output files, see `Run a ZIP file or folder <run_zip.html>`__.
* ``NotebookRuntime`` allows you to run a JupyterLab Python notebook, see `Run a Notebook <run_notebook.html>`__.
* ``GitPythonRuntime`` allows you to run source code from a Git repository, see `Run from Git <run_git.html>`__.

All of these runtime options allow you to configure a `Data Science Conda Environment <https://docs.oracle.com/en-us/iaas/data-science/using/conda_understand_environments.htm>`__ for running your code. For example, to define a python script as a job runtime with a TensorFlow conda environment you could use:

.. code-block:: python3

    from ads.jobs import ScriptRuntime

    runtime = (
        ScriptRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_service_conda("tensorflow26_p37_cpu_v2")
    )

You can store your source code in a local file path or location supported by `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__, including OCI Object Storage.

You can also use a custom conda environment published to OCI Object Storage by passing the ``uri`` to the ``with_custom_conda()`` method, for example:

.. code-block:: python3

    runtime = (
        ScriptRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_custom_conda("oci://bucket@namespace/conda_pack/pack_name")
    )

For more details on custom conda environment, see `Publishing a Conda Environment to an Object Storage Bucket in Your Tenancy <https://docs.oracle.com/en-us/iaas/data-science/using/conda_publishs_object.htm>`__.

You can also configure the environment variables, command line arguments, and free form tags for runtime:

.. code-block:: python3

    runtime = (
        ScriptRuntime()
        .with_source("oci://bucket_name@namespace/path/to/script.py")
        .with_service_conda("tensorflow26_p37_cpu_v2")
        .with_environment_variable(ENV="value")
        .with_argument("argument", key="value")
        .with_freeform_tag(tag_name="tag_value")
    )

With the preceding arguments, the script is started as ``python script.py argument --key value``.

Define a Job
============

With ``runtime`` and ``infrastructure``, you can define a job and give it a name:

.. code-block:: python3

    from ads.jobs import Job

    job = (
        Job(name="<job_display_name>")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )

If the job name is not specified, a name is generated automatically based on the name of the job artifact and a time stamp.

Alternatively, a job can also be defined with keyword arguments:

.. code-block:: python3

    job = Job(
        name="<job_display_name>",
        infrastructure=infrastructure,
        runtime=runtime
    )

Create and Run
==============

You can call the ``create()`` method of a job instance to create a job.  After the job is created, you can call the ``run()`` method to create and start a job run.  The ``run()`` method returns a ``DataScienceJobRun``.  You can monitor the job run output by calling the ``watch()`` method of the ``DataScienceJobRun`` instance:

.. code-block:: python3

    # Create a job
    job.create()
    # Run a job, a job run will be created and started
    job_run = job.run()
    # Stream the job run outputs
    job_run.watch()

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

Override Configuration
======================

When you run ``job.run()``, the job is run with the default configuration. You may want to override this default configuration with custom variables.  You can specify a custom job run display name, override command line argument, add additional environment variables, or free form tags as in this example:

.. code-block:: python3

  job_run = job.run(
    name="<my_job_run_name>",
    args="new_arg --new_key new_val",
    env_var={"new_env": "new_val"},
    freeform_tags={"new_tag": "new_tag_val"}
  )

YAML Serialization
==================

A job instance can be serialized to a YAML file by calling ``to_yaml()``, which returns the YAML as a string.  You can easily share the YAML with others, and reload the configurations by calling ``from_yaml()``.  The ``to_yaml()`` and ``from_yaml()`` methods also take an optional ``uri`` argument for saving and loading the YAML file.  This argument can be any URI to the file location supported by `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__, including Object Storage. For example:

.. code-block:: python3

    # Save the job configurations to YAML file
    job.to_yaml(uri="oci://bucket_name@namespace/path/to/job.yaml")

    # Load the job configurations from YAML file
    job = Job.from_yaml(uri="oci://bucket_name@namespace/path/to/job.yaml")

    # Save the job configurations to YAML in a string
    yaml_string = job.to_yaml()

    # Load the job configurations from a YAML string
    job = Job.from_yaml("""
    kind: job
    spec:
        infrastructure:
        kind: infrastructure
            ...
    """")

Here is an example of a YAML file representing the job defined in the preceding examples:

.. code-block:: yaml

    kind: job
    spec:
      name: <job_display_name>
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
      runtime:
        kind: runtime
        type: script
        spec:
          conda:
            slug: tensorflow26_p37_cpu_v2
            type: service
          scriptPathURI: oci://bucket_name@namespace/path/to/script.py

**ADS Job YAML schema**

.. code-block:: yaml

    kind:
      required: true
      type: string
      allowed:
        - job
    spec:
      required: true
      type: dict
      schema:
        id:
          required: false
        infrastructure:
          required: false
        runtime:
          required: false
        name:
          required: false
          type: string

**Data Science Job Infrastructure YAML Schema**

.. code-block:: yaml

    kind:
      required: true
      type: "string"
      allowed:
        - "infrastructure"
    type:
      required: true
      type: "string"
      allowed:
        - "dataScienceJob"
    spec:
      required: true
      type: "dict"
      schema:
        blockStorageSize:
          default: 50
          min: 50
          required: false
          type: "integer"
        compartmentId:
          required: false
          type: "string"
        displayName:
          required: false
          type: "string"
        id:
          required: false
          type: "string"
        logGroupId:
          required: false
          type: "string"
        logId:
          required: false
          type: "string"
        projectId:
          required: false
          type: "string"
        shapeName:
          required: false
          type: "string"
        subnetId:
          required: false
          type: "string"
        shapeConfigDetails:
          required: false
          type: "dict"

