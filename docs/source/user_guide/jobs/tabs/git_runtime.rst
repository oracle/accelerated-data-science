.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, GitPythonRuntime

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
            GitPythonRuntime()
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch19_p37_gpu_v1")
            # Specify the git repository
            # Optionally, you can specify the branch or commit
            .with_source("https://github.com/pytorch/tutorials.git")
            # Entrypoint is a relative path from the root of the git repo.
            .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
            # Copy files in "beginner_source/examples_nn" to object storage after job finishes.
            .with_output(
              output_dir="beginner_source/examples_nn",
              output_uri="oci://bucket_name@namespace/path/to/dir"
            )
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
        type: gitPython
        spec:
          conda:
            slug: pytorch19_p37_gpu_v1
            type: service
          entrypoint: beginner_source/examples_nn/polynomial_nn.py
          env:
          - name: GREETINGS
            value: Welcome to OCI Data Science
          outputDir: beginner_source/examples_nn
          outputUri: oci://bucket_name@namespace/path/to/dir
          url: https://github.com/pytorch/tutorials.git


.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()
