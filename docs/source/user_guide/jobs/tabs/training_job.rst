.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, GitPythonRuntime

    job = (
        Job(name="Training RNN with PyTorch")
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
            # Default block storage size is 50GB
            .with_block_storage_size(50)
        )
        .with_runtime(
            GitPythonRuntime(skip_metadata_update=True)
            # Use service conda pack
            .with_service_conda("pytorch110_p38_gpu_v1")
            # Specify training source code from GitHub
            .with_source(url="https://github.com/pytorch/examples.git", branch="main")
            # Entrypoint is a relative path from the root of the Git repository
            .with_entrypoint("word_language_model/main.py")
            # Pass the arguments as: "--epochs 5 --save model.pt --cuda"
            .with_argument(epochs=5, save="model.pt", cuda=None)
            # Set working directory, which will also be added to PYTHONPATH
            .with_working_dir("word_language_model")
            # Save the output to OCI object storage
            # output_dir is relative to working directory
            .with_output(output_dir=".", output_uri="oci://bucket@namespace/prefix")
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
          args:
          - --epochs
          - '5'
          - --save
          - model.pt
          - --cuda
          branch: main
          conda:
            slug: pytorch110_p38_gpu_v1
            type: service
          entrypoint: word_language_model/main.py
          outputDir: .
          outputUri: oci://bucket@namespace/prefix
          skipMetadataUpdate: true
          url: https://github.com/pytorch/examples.git
          workingDir: word_language_model


.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()