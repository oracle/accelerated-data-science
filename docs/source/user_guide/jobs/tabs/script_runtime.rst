.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, ScriptRuntime

    job = (
        Job(name="My Job")
        .with_infrastructure(
            DataScienceJob()
            # Configure logging for getting the job run outputs.
            .with_log_group_id("<log_group_ocid>")
            # Log resource will be auto-generated if log ID is not specified.
            .with_log_id("<log_ocid>")
            # If you are in an OCI data science notebook session,
            # the following configurations are not required.
            # Configurations from the notebook session will be used as defaults.
            .with_compartment_id("<compartment_ocid>")
            .with_project_id("<project_ocid>")
            .with_subnet_id("<subnet_ocid>")
            .with_shape_name("VM.Standard.E3.Flex")
            # Shape config details are applicable only for the flexible shapes.
            .with_shape_config_details(memory_in_gbs=16, ocpus=1)
            # Minimum/Default block storage size is 50 (GB).
            .with_block_storage_size(50)
        )
        .with_runtime(
            ScriptRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch110_p38_cpu_v1")
            # The job artifact can be a single Python script, a directory or a zip file.
            .with_source("local/path/to/code_dir")
            # Environment variable
            .with_environment_variable(NAME="Welcome to OCI Data Science.")
            # Command line argument
            .with_argument("100 linux \"hi there\"")
            # The entrypoint is applicable only to directory or zip file as source
            # The entrypoint should be a path relative to the working dir.
            # Here my_script.sh is a file in the code_dir/my_package directory
            .with_entrypoint("my_package/my_script.sh")
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
        type: script
        spec:
          args:
          - 100 linux "hi there"
          conda:
            slug: pytorch110_p38_cpu_v1
            type: service
          entrypoint: my_package/my_script.sh
          env:
          - name: NAME
            value: Welcome to OCI Data Science.
          scriptPathURI: local/path/to/code_dir

.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()
