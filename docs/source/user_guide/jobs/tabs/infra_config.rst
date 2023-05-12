.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, GitPythonRuntime

    infrastructure = (
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
        # A maximum number of 5 file systems are allowed to be mounted for a job.
        .with_storage_mount(
          {
            "src" : "<mount_target_ip_address>@<export_path>",
            "dest" : "<destination_directory_name>"
          }
        )
    )

  .. code-tab:: yaml
    :caption: YAML

    kind: infrastructure
    type: dataScienceJob
    spec:
      blockStorageSize: 50
      compartmentId: <compartment_ocid>
      logGroupId: <log_group_ocid>
      logId: <log_ocid>
      projectId: <project_ocid>
      shapeConfigDetails:
        memoryInGBs: 16
        ocpus: 1
      shapeName: VM.Standard.E3.Flex
      subnetId: <subnet_ocid>
      storageMount:
      - src: <mount_target_ip_address>@<export_path>
        dest: <destination_directory_name>
