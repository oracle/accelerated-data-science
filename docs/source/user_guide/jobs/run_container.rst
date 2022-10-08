Run a Container
***************

The ADS ``ContainerRuntime`` class allows you to run a container image using OCI data science jobs.

To use the ``ContainerRuntime``, you need to first push the image to `OCI container registry <https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryoverview.htm>`_. See `Creating a Repository <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ and `Pushing Images Using the Docker CLI <https://docs.oracle.com/en-us/iaas/Content/Registry/Tasks/registrycreatingarepository.htm>`_ for more details.

Python
======

To configure ``ContainerRuntime``, you must specify the container ``image``. Similar to other runtime, you can add environment variables. You can optionally specify the `entrypoint` and `cmd` for running the container (See `Understand how CMD and ENTRYPOINT interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`_).

.. code-block:: python3

    from ads.jobs import Job, DataScienceJob, ContainerRuntime

    job = (
        Job()
        .with_infrastructure(
            DataScienceJob()
            .with_log_group_id("<log_group_ocid>")
            .with_log_id("<log_ocid>")
            # The following infrastructure configurations are optional
            # if you are in an OCI data science notebook session.
            # The configurations of the notebook session will be used as defaults
            .with_compartment_id("<compartment_ocid>")
            .with_project_id("<project_ocid>")
            .with_subnet_id("<subnet_ocid>")
            .with_shape_name("VM.Standard.E3.Flex")
            .with_shape_config_details(memory_in_gbs=16, ocpus=1) # Applicable only for the flexible shapes
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

    # Create the job with OCI
    job.create()
    # Run the job and stream the outputs
    job_run = job.run().watch()

YAML
====

You could use the following YAML to create the same job:

.. code-block:: yaml

    kind: job
    spec:
      name: container-job
      infrastructure:
        kind: infrastructure
        type: dataScienceJob
        spec:
          logGroupId: <log_group_ocid>
          logId: <log_ocid>
          compartmentId: <compartment_ocid>
          projectId: <project_ocid>
          subnetId: <subnet_ocid>
          shapeName: VM.Standard.E3.Flex
          shapeConfigDetails:
            memoryInGBs: 16
            ocpus: 1
          blockStorageSize: 50
      runtime:
        kind: runtime
        type: container
        spec:
          image: iad.ocir.io/<your_tenancy>/<your_image>
          cmd:
          - sleep 5 && echo $GREETINGS
          entrypoint:
          - /bin/sh
          - -c
          env:
          - name: GREETINGS
            value: Welcome to OCI Data Science

**ContainerRuntime Schema**

.. code-block:: yaml

    kind:
      required: true
      type: string
      allowed:
        - runtime
    type:
      required: true
      type: string
      allowed:
        - container
    spec:
      type: dict
      required: true
      schema:
        image:
          required: true
          type: string
        entrypoint:
          required: false
          type:
          - string
          - list
        cmd:
          required: false
          type:
          - string
          - list
        env:
          nullable: true
          required: false
          type: list
          schema:
            type: dict
            schema:
              name:
                type: string
              value:
                type:
                - number
                - string

