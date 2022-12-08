Run a Script
************

This example shows you how to create a job running "Hello World" Python scripts.  Although Python scripts are used here, you could also run Bash or Shell scripts.  The Logging service log and log group are defined in the infrastructure.  The output of the script appear in the logs.

Python
======

Suppose you would like to run the following "Hello World" python script named ``job_script.py``.

.. code-block:: python3

  print("Hello World")

First, initiate a job with a job name:

.. code-block:: python3

  from ads.jobs import Job
  job = Job(name="Job Name")

Next, you specify the desired infrastructure to run the job. If you are in a notebook session, ADS can automatically fetch the infrastructure configurations and use them for the job. If you aren't in a notebook session or you want to customize the infrastructure, you can specify them using the methods from the ``DataScienceJob`` class:

.. code-block:: python3

  from ads.jobs import DataScienceJob

  job.with_infrastructure(
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

In this example, it is a Python script so the ``ScriptRuntime()`` class is used to define the name of the script using the ``.with_source()`` method:

.. code-block:: python3

    from ads.jobs import ScriptRuntime
    job.with_runtime(
      ScriptRuntime().with_source("job_script.py")
    )

Finally, you create and run the job, which gives you access to the
``job_run.id``:

.. code-block:: python3

    job.create()
    job_run = job.run()

Additionally, you can acquire the job run using the OCID:

.. code-block:: python3

    from ads.jobs import DataScienceJobRun
    job_run = DataScienceJobRun.from_ocid(job_run.id)

The ``.watch()`` method is useful to monitor the progress of the job run:

.. code-block:: python3

    job_run.watch()

After the job has been created and runs successfully, you can find
the output of the script in the logs if you configured logging.

YAML
====

You could also initialize a job directly from a YAML string.  For example, to create a job identical to the preceding example, you could simply run the following:

.. code-block:: python3

  job = Job.from_string(f"""
  kind: job
  spec:
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
    name: <resource_name>
    runtime:
      kind: runtime
      type: python
      spec:
        scriptPathURI: job_script.py
  """)


Command Line Arguments
======================

If the Python script that you want to run as a job requires CLI arguments,
use the ``.with_argument()`` method to pass the arguments to the job.

Python
------

Suppose you want to run the following python script named ``job_script_argument.py``:

.. code-block:: python3

    import sys
    print("Hello " + str(sys.argv[1]) + " and " + str(sys.argv[2]))

This example runs a job with CLI arguments:

.. code-block:: python3

  job = Job()
  job.with_infrastructure(
    DataScienceJob()
    .with_log_id("<log_id>")
    .with_log_group_id("<log_group_id>")
  )

  # The CLI argument can be passed in using `with_argument` when defining the runtime
  job.with_runtime(
    ScriptRuntime()
      .with_source("job_script_argument.py")
      .with_argument("<first_argument>", "<second_argument>")
    )

  job.create()
  job_run = job.run()

After the job run is created and run, you can use the ``.watch()`` method to monitor
its progress:

.. code-block:: python3

    job_run.watch()

This job run prints out ``Hello <first_argument> and <second_argument>``.

YAML
----

You could create the preceding example job with the following YAML file:

.. code-block:: yaml

	kind: job
	spec:
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
      type: python
	    spec:
	      args:
	      - <first_argument>
	      - <second_argument>
	      scriptPathURI: job_script_argument.py


Environment Variables
=====================

Similarly, if the script you want to run requires environment variables, you also pass them in using the ``.with_environment_variable()`` method. The key-value pair of the environment variable are passed in using the ``.with_environment_variable()`` method, and are accessed in the Python script using the ``os.environ`` dictionary.

Python
------

Suppose you want to run the following python script named ``job_script_env.py``:

.. code-block:: python3

  import os
  import sys
  print("Hello " + os.environ["KEY1"] + " and " + os.environ["KEY2"])""")

This example runs a job with environment variables:

.. code-block:: python3

  job = Job()
  job.with_infrastructure(
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
    .with_shape_config_details(memory_in_gbs=16, ocpus=1)
    .with_block_storage_size(50)
  )

  job.with_runtime(
    ScriptRuntime()
    .with_source("job_script_env.py")
    .with_environment_variable(KEY1="<first_value>", KEY2="<second_value>")
  )
  job.create()
  job_run = job.run()

You can watch the progress of the job run using the ``.watch()`` method:

.. code-block:: python3

  job_run.watch()

This job run prints out ``Hello <first_value> and <second_value>``.

YAML
----

You could create the preceding example job with the following YAML file:

.. code-block:: yaml

    kind: job
    spec:
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
        type: python
        spec:
          env:
          - name: KEY1
            value: <first_value>
          - name: KEY2
            value: <second_value>
          scriptPathURI: job_script_env.py


**ScriptRuntime YAML Schema**

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
      - script
  spec:
    required: true
    type: dict
    schema:
      args:
        nullable: true
        required: false
        type: list
        schema:
          type: string
      conda:
        nullable: false
        required: false
        type: dict
        schema:
          slug:
            required: true
            type: string
          type:
            allowed:
              - service
            required: true
            type: string
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
      scriptPathURI:
        required: true
        type: string
      entrypoint:
        required: false
        type: string

