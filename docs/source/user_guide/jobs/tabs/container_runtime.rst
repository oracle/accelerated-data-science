.. tabs::

  .. code-tab:: python
    :caption: Python

    from ads.jobs import Job, DataScienceJob, ContainerRuntime

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
        ContainerRuntime()
        .with_image("<region>.ocir.io/<your_tenancy>/<your_image>")
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_entrypoint(["/bin/sh", "-c"])
        .with_cmd("sleep 5 && echo $GREETINGS")
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
        type: container
        spec:
          cmd: sleep 5 && echo $GREETINGS
          entrypoint:
          - /bin/sh
          - -c
          env:
          - name: GREETINGS
            value: Welcome to OCI Data Science
          image: <region>.ocir.io/<your_tenancy>/<your_image>