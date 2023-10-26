.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, PyTorchDistributedRuntime

    job = (
        Job(name="LLAMA2-Fine-Tuning")
        .with_infrastructure(
            DataScienceJob()
            .with_log_group_id("<log_group_ocid>")
            .with_log_id("<log_ocid>")
            .with_compartment_id("<compartment_ocid>")
            .with_project_id("<project_ocid>")
            .with_subnet_id("<subnet_ocid>")
            .with_shape_name("VM.GPU.A10.2")
            .with_block_storage_size(256)
        )
        .with_runtime(
            PyTorchDistributedRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda("pytorch20_p39_gpu_v2")
            .with_git(
              url="https://github.com/facebookresearch/llama-recipes.git",
              commit="1aecd00924738239f8d86f342b36bacad180d2b3"
            )
            .with_dependency(
              pip_pkg=" ".join([
                "--extra-index-url https://download.pytorch.org/whl/cu118 torch==2.1.0",
                "git+https://github.com/huggingface/peft.git@15a013af5ff5660b9377af24d3eee358213d72d4"
                "appdirs==1.4.4",
                "llama-recipes==0.0.1",
                "py7zr==0.20.6",
              ])
            )
            .with_output("/home/datascience/outputs", "oci://bucket@namespace/outputs/$JOB_RUN_OCID")
            .with_command(" ".join([
              "torchrun examples/finetuning.py",
              "--enable_fsdp",
              "--pure_bf16",
              "--batch_size_training 1",
              "--model_name $MODEL_NAME",
              "--dist_checkpoint_root_folder /home/datascience/outputs",
              "--dist_checkpoint_folder fine-tuned"
            ]))
            .with_replica(2)
            .with_environment_variable(
              MODEL_NAME="meta-llama/Llama-2-7b-hf",
              HUGGING_FACE_HUB_TOKEN="<access_token>",
              LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/conda/lib",
            )
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: job
    apiVersion: v1.0
    spec:
      name: LLAMA2-Fine-Tuning
      infrastructure:
        kind: infrastructure
        spec:
          blockStorageSize: 256
          compartmentId: "<compartment_ocid>"
          logGroupId: "<log_group_id>"
          logId: "<log_id>"
          projectId: "<project_id>"
          subnetId: "<subnet_id>"
          shapeName: VM.GPU.A10.2
        type: dataScienceJob
      runtime:
        kind: runtime
        type: pyTorchDistributed
        spec:
          git:
            url: https://github.com/facebookresearch/llama-recipes.git
            commit: 1aecd00924738239f8d86f342b36bacad180d2b3
          command: >-
            torchrun llama_finetuning.py
            --enable_fsdp
            --pure_bf16
            --batch_size_training 1
            --model_name $MODEL_NAME
            --dist_checkpoint_root_folder /home/datascience/outputs
            --dist_checkpoint_folder fine-tuned
          replicas: 2
          conda:
            type: service
            slug: pytorch20_p39_gpu_v2
          dependencies:
            pipPackages: >-
              --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.1.0
              git+https://github.com/huggingface/peft.git@15a013af5ff5660b9377af24d3eee358213d72d4
              llama-recipes==0.0.1
              appdirs==1.4.4
              py7zr==0.20.6
          outputDir: /home/datascience/outputs
          outputUri: oci://bucket@namespace/outputs/$JOB_RUN_OCID
          env:
            - name: MODEL_NAME
              value: meta-llama/Llama-2-7b-hf
            - name: HUGGING_FACE_HUB_TOKEN
              value: "<access_token>"
            - name: LD_LIBRARY_PATH
              value: /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/conda/lib
