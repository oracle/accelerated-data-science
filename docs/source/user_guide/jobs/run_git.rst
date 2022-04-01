Run a Git Repo
--------------
       
The ADS ``GitPythonRuntime`` class allows you to run source code from a Git 
repository as a Data Science job. The next example shows how to run a 
`Pytorch Neural Network Example to train third order polynomial predicting y=sin(x) <https://github.com/pytorch/tutorials/blob/master/beginner_source/examples_nn/polynomial_nn.py>`__.

To configure the ``GitPythonRuntime``, you must specify the source
code ``url`` and entrypoint ``path``. Similar to ``PythonRuntime``,
you can specify a service conda environment, environment variables, and
CLI arguments. In this example, the ``pytorch19_p37_gpu_v1`` service 
conda environment is used.
Assuming you are running this example in an Data Science notebook session,
only log ID and log group ID need to be configured for the ``DataScienceJob`` object, 
see `Data Science Jobs <data_science_job.html>`__ for more details about configuring the infrastructure.

Python
~~~~~~

.. code:: ipython3

  from ads.jobs import Job, DataScienceJob, GitPythonRuntime

  infrastructure = (
    DataScienceJob()
    .with_log_id(<"log_id">)
    .with_log_group_id(<"log_group_id">)
  )
   
  runtime = (
    GitPythonRuntime()
    .with_source("https://github.com/pytorch/tutorials.git")
    .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
    .with_service_conda("pytorch19_p37_gpu_v1")
  )

The default branch from the Git repository is used 
unless you specify a different ``branch`` or
``commit`` using the ``.with_source()`` method if needed.

For a public repository, we recommend the "http://" or "https://" URL.
Authentication may be required for the SSH URL even if the repository is
public.

To use a private repository, you must first save an SSH key
to an `OCI Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm>`__
as a secret, and provide the ``secret_ocid`` to the ``with_source()``
method, see `Managing Secret with
Vault <https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Tasks/managingsecrets.htm>`__.
For example, you could use `GitHub Deploy
Key <https://docs.github.com/en/developers/overview/managing-deploy-keys#deploy-keys>`__.

The entry point specifies how the source code is invoked.
The ``.with_entrypiont()`` has the following arguments: 

* ``path``: Required. The relative path for the script, module, or file to start the job.
* ``func``: Optional. The function in the script specified by ``path`` to call. If you don't specify it, then the script specified by ``path`` is run as a Python script in a subprocess.

With the ``GitPythonRuntime`` class, you can save the output files from
the job run to Object Storage. By default, the source code is cloned to
the ``~/Code`` directory. However, in the next example the files in the ``example_nn`` 
directory are copied to the Object Storage specified by the ``output_uri`` 
parameter. The ``output_uri`` parameter should have this format:

``oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX``

.. code:: ipython3

  runtime.with_output(
    output_dir="~/Code/tutorials/beginner_source/examples_nn",
    output_uri="oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX"
  )

.. code:: ipython3

  job = (
    Job(name="git_example")
    .with_infrastructure(infrastructure)
    .with_runtime(runtime)
  ).create()

After the job is created, you can run it, and then monitor the job run 
using the ``.watch()`` API:

.. code:: ipython3

  run = job.run().watch()

The ``GitPythonRuntime`` also supports these additional configurations: 

* The ``.with_python_path()`` method allows you to add additional Python paths 
  to the runtime. By default, the code directory checked out from Git is added
  to ``sys.path``. Additional Python paths are appended
  before the code directory is appended. 
* The ``.with_argument()`` method allows you to pass arguments to invoke the 
  script or function. For running a script, the arguments are passed in as 
  CLI arguments. For running a function, the ``list``
  and ``dict`` JSON serializable objects are supported and are passed into the function.

For example:

.. code:: ipython3

  runtime = (
    GitPythonRuntime()
    .with_source("YOUR_GIT_URL")
    .with_entrypoint(path="YOUR_MODULE_PATH", func="YOUR_FUNCTION")
    .with_service_conda("pytorch19_p37_gpu_v1")
    .with_argument("val", ["a", "b"], key=dict(k="v"))
  )

The ``GitPythonRuntime`` method updates metadata in the free form tags of the
job run after the job run finishes. The following tags are added
automatically: 

-  ``repo``: The URL of the Git repository.
-  ``commit``: The Git commit ID.
-  ``module``: The entry script or module.
-  ``method``: The entry function or method.
-  ``outputs``: The prefix of the output files in Object Storage.

The new values overwrite any existing tags. If you want to
skip the metadata update, set ``skip_metadata_update`` to ``True`` when 
initializing the runtime:

.. code:: ipython3

  runtime = GitPythonRuntime(skip_metadata_update=True)


YAML
~~~~

You could create the preceding example job with the following YAML file:

.. code:: yaml

  kind: job
  spec:
    infrastructure:
      kind: infrastructure
      spec:
        jobInfrastructureType: STANDALONE
        jobType: DEFAULT
        logGroupId: <log_group_id>
        logId: <log_id>
      type: dataScienceJob
    name: git_example
    runtime:
      kind: runtime
      spec:
        conda:
          slug: pytorch19_p37_gpu_v1
          type: service
        entrypoint: beginner_source/examples_nn/polynomial_nn.py
        outputDir: ~/Code/tutorials/beginner_source/examples_nn
        outputUri: oci://BUCKET_NAME@BUCKET_NAMESPACE/PREFIX
        url: https://github.com/pytorch/tutorials.git
      type: gitPython


**GitPythonRuntime YAML Schema**


.. code:: yaml

  kind:
    allowed:
      - runtime
    required: true
    type: string
  spec:
    required: true
    schema:
      args:
        nullable: true
        required: false
        schema:
          type: string
        type: list
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
        schema:
          slug:
            required: true
            type: string
          type:
            allowed:
              - service
            required: true
            type: string
        type: dict
      entryFunction:
        nullable: true
        required: false
        type: string
      pythonPath:
        nullable: true
        required: false
        type: list
      entrypoint:
        required: false
        type:
          - string
          - list
      env:
        required: false
        schema:
          type: dict
        type: list
      freeform_tag:
        required: false
        type: dict
      outputDir:
        required: false
        type: string
      outputUri:
        required: false
        type: string
      url:
        required: false
        type: string
    type: dict
  type:
    allowed:
      - gitPython
    required: true
    type: string

