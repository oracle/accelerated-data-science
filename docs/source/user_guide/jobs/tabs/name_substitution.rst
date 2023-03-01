.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, PythonRuntime

    job = (
        Job(name="Training on ${DATASET_NAME}")
        .with_infrastructure(
            DataScienceJob()
            .with_log_group_id("<log_group_ocid>")
            .with_log_id("<log_ocid>")
            .with_compartment_id("<compartment_ocid>")
            .with_project_id("<project_ocid>")
            .with_shape_name("VM.Standard.E3.Flex")
            .with_shape_config_details(memory_in_gbs=16, ocpus=1)
        )
        .with_runtime(
            PythonRuntime()
            .with_service_conda("pytorch110_p38_gpu_v1")
            .with_environment_variable(DATASET_NAME="MyData")
            .with_source("local/path/to/training_script.py")
            .with_output("output", "oci://bucket_name@namespace/prefix/${JOB_RUN_OCID}")
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: job
    spec:
      name: Training on ${DATASET_NAME}
      infrastructure:
        kind: infrastructure
        type: dataScienceJob
        spec:
          compartmentId: <compartment_ocid>
          logGroupId: <log_group_ocid>
          logId: <log_ocid>
          projectId: <project_ocid>
          shapeConfigDetails:
            memoryInGBs: 16
            ocpus: 1
          shapeName: VM.Standard.E3.Flex
      runtime:
        kind: runtime
        type: python
        spec:
          conda:
            slug: pytorch110_p38_cpu_v1
            type: service
          env:
          - name: DATASET_NAME
            value: MyData
          outputDir: output
          outputUri: oci://bucket_name@namespace/prefix/${JOB_RUN_OCID}
          scriptPathURI: local/path/to/training_script.py
