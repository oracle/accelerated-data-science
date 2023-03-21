.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, NotebookRuntime

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
            NotebookRuntime()
            .with_notebook(
                path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                encoding='utf-8'
            )
            .with_service_conda("tensorflow28_p38_cpu_v1")
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_exclude_tag(["ignore", "remove"])
            .with_output("oci://bucket_name@namespace/path/to/dir")
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
        type: notebook
        spec:
          conda:
            slug: tensorflow28_p38_cpu_v1
            type: service
          env:
          - name: GREETINGS
            value: Welcome to OCI Data Science
          excludeTags:
          - ignore
          - remove
          notebookEncoding: utf-8
          notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
          outputUri: oci://bucket_name@namespace/path/to/dir

.. code-block:: python

  # Create the job on OCI Data Science
  job.create()
  # Start a job run
  run = job.run()
  # Stream the job run outputs
  run.watch()
  # Download the notebook back to local
  run.download("/path/to/local/dir")
