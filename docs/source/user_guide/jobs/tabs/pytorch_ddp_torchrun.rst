.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, PyTorchDistributedRuntime

    job = (
        Job(name="PyTorch DDP Job")
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
            .with_shape_name("VM.GPU.A10.1")
            # Minimum/Default block storage size is 50 (GB).
            .with_block_storage_size(50)
        )
        .with_runtime(
            PyTorchDistributedRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch20_p39_gpu_v1")
            .with_git(url="https://github.com/pytorch/examples.git", commit="d91085d2181bf6342ac7dafbeee6fc0a1f64dcec")
            .with_dependency("distributed/minGPT-ddp/requirements.txt")
            .with_inputs({
              "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt": "data/input.txt"
            })
            .with_output("data", "oci://bucket_name@namespace/path/to/dir")
            .with_command("torchrun distributed/minGPT-ddp/mingpt/main.py data_config.path=data/input.txt trainer_config.snapshot_path=data/snapshot.pt")
            .with_replica(2)
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: job
    apiVersion: v1.0
    spec:
      name: PyTorch-MinGPT
      infrastructure:
        kind: infrastructure
        spec:
          blockStorageSize: 50
          compartmentId: "{{ compartment_id }}"
          logGroupId: "{{ log_group_id }}"
          logId: "{{ log_id }}"
          projectId: "{{ project_id }}"
          subnetId: "{{ subnet_id }}"
          shapeName: VM.GPU.A10.1
        type: dataScienceJob
      runtime:
        kind: runtime
        type: pyTorchDistributed
        spec:
          replicas: 2
          conda:
            type: service
            slug: pytorch110_p38_gpu_v1
          dependencies:
            pipRequirements: distributed/minGPT-ddp/requirements.txt
          inputs:
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt": "data/input.txt"
          outputDir: data
          outputUri: oci://bucket_name@namespace/path/to/dir
          git:
            url: https://github.com/pytorch/examples.git
            commit: d91085d2181bf6342ac7dafbeee6fc0a1f64dcec
          command: >-
            torchrun distributed/minGPT-ddp/mingpt/main.py
            data_config.path=data/input.txt
            trainer_config.snapshot_path=data/snapshot.pt
