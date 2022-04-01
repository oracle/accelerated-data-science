Run a ZIP file or folder
------------------------

ScriptRuntime
~~~~~~~~~~~~~

The ``ScriptRuntime`` class is designed for you to define job artifacts and configurations supported by OCI Data Science jobs natively.
It can be used with any script types that is supported by the OCI Data Science jobs, including a ZIP or compressed tar file or folder.
See `Preparing Job Artifacts <https://docs.oracle.com/en-us/iaas/data-science/using/jobs-artifact.htm>`__ for more details.
In the job run, the working directory is the user's home directory. For example ``/home/datascience``.

Python
++++++

If you are in a notebook session, ADS can automatically fetch the
infrastructure configurations, and use them in the job. If you aren't
in a notebook session or you want to customize the infrastructure,
you can specify them using the methods in the ``DataScienceJob`` class.

With the ``ScriptRuntime``, you can pass in a path to a ZIP file or directory.
For a ZIP file, the path can be any URI supported by
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__,
including OCI Object Storage.

You must specify the ``entrypoint``, which is the relative path from the ZIP file or
directory to the script starting your program. Note that the ``entrypoint`` contains the
name of the directory, since the directory itself is also zipped as the job artifact.

.. code:: ipython3

  from ads.jobs import Job, DataScienceJob, ScriptRuntime

  job = (
    Job()
    .with_infrastructure(
        DataScienceJob()
        .with_log_id("<log_id>")
        .with_log_group_id("<log_group_id>")
    )
    .with_runtime(
        ScriptRuntime()
        .with_source("path/to/zip_or_dir", entrypoint="zip_or_dir/main.py")
        .with_service_conda("pytorch19_p37_cpu_v1")
    )
  )

.. code:: ipython3

	# Create the job with OCI
	job.create()
	# Run the job and stream the outputs
	job_run = job.run().watch()


YAML
++++

You could use the following YAML example to create the same job with ``ScriptRuntime``:

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
	  runtime:
	    kind: runtime
	    spec:
        conda:
          slug: pytorch19_p37_cpu_v1
          type: service
	      entrypoint: zip_or_dir/main.py
	      scriptPathURI: path/to/zip_or_dir
	    type: script



PythonRuntime
~~~~~~~~~~~~~

The ``PythonRuntime`` class allows you to run Python code with ADS enhanced features like configuring the working directory and Python path.
It also allows you to copy the output files to OCI Object Storage. This is especially useful for Python code involving multiple files and packages in the job artifact.

The ``PythonRuntime`` uses an ADS generated driver script as the entry point for the job run. It performs additional
operations before and after invoking your code. You can examine the driver script by downloading the job artifact from the OCI Console.

Python
++++++

Relative to ``ScriptRunTime`` the ``PythonRuntime`` has 3 additional methods:

* ``.with_working_dir()``: Specify the working directory to use when running a job. By default, the working directory is also added to the Python paths. This should be a relative path from the parent of the job artifact directory.
* ``.with_python_path()``: Add one or more Python paths to use when running a job. The paths should be relative paths from the working directory.
* ``.with_output()``: Specify the output directory and a remote URI (for example, an OCI Object Storage URI) in the job run. Files in the output directory are copied to the remote output URI after the job run finishes successfully.

Following is an example of creating a job with ``PythonRuntime``:

.. code:: ipython3

  from ads.jobs import Job, DataScienceJOb, PythonRuntime

  job = (
    Job()
    .with_infrastructure(
      DataScienceJob()
      .with_log_id(<"log_id">)
      .with_log_group_id(<"log_group_id">)
    )
    .with_runtime(
      PythonRuntime()
      .with_service_conda("pytorch19_p37_cpu_v1")
      # The job artifact directory is named "zip_or_dir"
      .with_source("local/path/to/zip_or_dir", entrypoint="zip_or_dir/my_package/entry.py")
      # Change the working directory to be inside the job artifact directory
      # Working directory a relative path from the parent of the job artifact directory
      # Working directory is also added to Python paths
      .with_working_dir("zip_or_dir")
      # Add an additional Python path
      # The "my_python_packages" folder is under "zip_or_dir" (working directory)
      .with_python_path("my_python_packages")
      # Files in "output" directory will be copied to OCI object storage once the job finishes
      # Here we assume "output" is a folder under "zip_or_dir" (working directory)
      .with_output("output", "oci://bucket_name@namespace/path/to/dir")
    )
  )

YAML
++++

You could use the following YAML to create the same job with ``PythonRuntime``:

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
	  runtime:
	    kind: runtime
	    spec:
        conda:
          slug: pytorch19_p37_cpu_v1
          type: service
	      entrypoint: zip_or_dir/my_package/entry.py
	      scriptPathURI: path/to/zip_or_dir
        workingDir: zip_or_dir
        outputDir: zip_or_dir/output
        outputUri: oci://bucket_name@namespace/path/to/dir
        pythonPath:
        - "zip_or_dir/python_path"
	    type: python

**PythonRuntime YAML Schema**

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
      env:
        required: false
        schema:
          type: dict
        type: list
      freeform_tag:
        required: false
        type: dict
      scriptPathURI:
        required: true
        type: string
      entrypoint:
        required: false
        type: string
      outputDir:
        required: false
        type: string
      outputUri:
        required: false
        type: string
      workingDir:
        required: false
        type: string
      pythonPath:
        required: false
        type: list
    type: dict
  type:
    allowed:
      - script
    required: true
    type: string
