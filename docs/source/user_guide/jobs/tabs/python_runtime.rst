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
            # Environment variable
            .with_environment_variable(NAME="Welcome to OCI Data Science.")
            # Command line argument, arg1 --key arg2
            .with_argument("arg1", key="arg2")
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
          args:
          - arg1
          - --key
          - arg2
          conda:
            slug: pytorch19_p37_cpu_v1
            type: service
          entrypoint: my_package/my_script.py
          env:
          - name: NAME
            value: Welcome to OCI Data Science.
          outputDir: output
          outputUri: oci://bucket_name@namespace/path/to/dir
          pythonPath:
          - other_packages
          scriptPathURI: local/path/to/code_dir
          workingDir: code_dir
          workingDir: code_dir


.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()
