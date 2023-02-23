.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, GitPythonRuntime

    job = (
      Job(name="Training MNIST with PyTorch")
      .with_infrastructure(
        DataScienceJob()
        .with_log_group_id("<log_group_ocid>")
        .with_log_id("<log_ocid>")
        .with_shape_name("VM.GPU3.1")
        # The following infrastructure configurations are optional
        # if you are in an OCI data science notebook session.
        # The configurations of the notebook session will be used as defaults.
        .with_compartment_id("<compartment_ocid>")
        .with_project_id("<project_ocid>")
        .with_block_storage_size(50)
      )
      .with_runtime(
        GitPythonRuntime(skip_metadata_update=True)
        .with_source(url="https://github.com/pytorch/examples.git", branch="main")
        .with_entrypoint(path="mnist/main.py")
        .with_service_conda("pytorch110_p38_gpu_v1")
        # Pass the arguments as: main.py --epochs 10 --save-model
        .with_argument(**{"epochs": 10, "save-model": None})
        .with_output(
          "mnist_rnn.pt",
          output_uri="oci://bucket_name@namespace/path/to/dir"
        )
      )
    )

    # Create the job on OCI Data Science
    job.create()
    # Start a job run
    run = job.run()
    # Stream the job run outputs
    run.watch()

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