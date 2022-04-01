Run a Script
------------

This example shows you how to create a job running "Hello World" Python scripts.
Although Python scripts are used here, you could also run Bash or Shell scripts.
The Logging service log and log group are defined in the infrastructure.
The output of the script appear in the logs.

Python
~~~~~~
Suppose you would like to run the following "Hello World" python script named ``job_script.py``.

.. code:: ipython3

  print("Hello World")

First, initiate a job with a job name:

.. code:: ipython3

  from ads.jobs import Job
  job = Job(name="Job Name")

Next, you specify the desired infrastructure to run the job. If
you are in a notebook session, ADS can automatically fetch the
infrastructure configurations and use them for the job. If you aren't 
in a notebook session or you want to customize the infrastructure, 
you can specify them using the methods from the ``DataScienceJob`` class:

.. code:: ipython3

  from ads.jobs import DataScienceJob

  job.with_infrastructure(
    DataScienceJob()
    .with_log_id("<log_id>")
    .with_log_group_id("<log_group_id>")
  )

In this example, it is a Python script so the ``ScriptRuntime()`` class is used to define the
name of the script using the ``.with_source()`` method:

.. code:: ipython3

    from ads.jobs import ScriptRuntime
    job.with_runtime(
      ScriptRuntime().with_source("job_script.py")
    )

Finally, you create and run the job, which gives you access to the
``job_run.id``:

.. code:: ipython3

    job.create()
    job_run = job.run() 

Additionally, you can acquire the job run using the OCID:

.. code:: ipython3

    from ads.jobs import DataScienceJobRun
    job_run = DataScienceJobRun.from_ocid(job_run.id)

The ``.watch()`` method is useful to monitor the progress of the job run:

.. code:: ipython3

    job_run.watch() 

After the job has been created and runs successfully, you can find
the output of the script in the logs if you configured logging.

YAML
~~~~

You could also initialize a job directly from a YAML string.
For example, to create a job identical to the preceding example, you
could simply run the following:

.. code:: ipython3

  job = Job.from_string(f"""
  kind: job
  spec:
    infrastructure:
      kind: infrastructure
      spec:
        jobInfrastructureType: STANDALONE
        jobType: DEFAULT
        logGroupId: <log_group_id>
        logId: <log_id>
      type: dataScienceJob
    name: <resource_name>
    runtime:
      kind: runtime
      spec:
        scriptPathURI: job_script.py
      type: python
  """)


Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

If the Python script that you want to run as a job requires CLI arguments, 
use the ``.with_argument()`` method to pass the arguments to the job.

Python
++++++

Suppose you want to run the following python script named ``job_script_argument.py``:

.. code:: ipython3

    import sys
    print("Hello " + str(sys.argv[1]) + " and " + str(sys.argv[2]))

This example runs a job with CLI arguments:

.. code:: ipython3

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

.. code:: ipython3

    job_run.watch()

This job run prints out ``Hello <first_argument> and <second_argument>``.

YAML
++++

You can define a job with a YAML string. In order to define a job identical
to the preceding job, you could use the following before running ``job.create()`` and ``job.run()``:

.. code:: ipython3

	job = Job.from_yaml(f"""
	kind: job
	spec:
	  infrastructure:
	    kind: infrastructure
	    spec:
	      jobInfrastructureType: STANDALONE
	      jobType: DEFAULT
	      logGroupId: <log_group_id>
	      logId: <log_id>
	    type: dataScienceJob
	  runtime:
	    kind: runtime
	    spec:
	      args:
	      - <first_argument>
	      - <second_argument>
	      scriptPathURI: job_script_argument.py
	    type: python
	""")


Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Similarly, if the script you want to run requires environment
variables, you also pass them in using the 
``.with_environment_variable()`` method. The key-value pair of the environment 
variable are passed in using the ``.with_environment_variable()`` method, 
and are accessed in the Python script using the ``os.environ`` dictionary.

Python
++++++

Suppose you want to run the following python script named ``job_script_env.py``:

.. code:: ipython3

  import os
  import sys
  print("Hello " + os.environ["KEY1"] + " and " + os.environ["KEY2"])""")

This example runs a job with environment variables:

.. code:: ipython3
    
  job = Job()
  job.with_infrastructure(
    DataScienceJob()
    .with_log_group_id(<"log_group_id">)
    .with_log_id(<"log_id">)
  )

  job.with_runtime(
    ScriptRuntime()
    .with_source("job_script_env.py")
    .with_environment_variable(KEY1="<first_value>", KEY2="<second_value>")
  )
  job.create()
  job_run = job.run()

You can watch the progress of the job run using the ``.watch()`` method:

.. code:: ipython3

  job_run.watch()

This job run print sout ``Hello <first_value> and <second_value>``.

YAML
++++

The next example shows the equivalent way to create a job from a YAML string:

.. code:: ipython3
	
	job = Job.from_yaml(f"""
	kind: job
	spec:
	  infrastructure:
	    kind: infrastructure
	    spec:
	      jobInfrastructureType: STANDALONE
	      jobType: DEFAULT
	      logGroupId: <log_group_id>
	      logId: <log_id>
	    type: dataScienceJob
	  name: null
	  runtime:
	    kind: runtime
	    spec:
	      env:
	      - name: KEY1
	        value: <first_value>
	      - name: KEY2
		      value: <second_value>
	      scriptPathURI: job_script_env.py
	    type: python
	""")

**ScriptRuntime YAML Schema**

.. code:: yaml

  kind:
    allowed:
      - runtime
    required: true
    type: string
  spec:
    required: true
    schema:
      args:
        nullable: true
        required: false
        schema:
          type: string
        type: list
      conda:
        nullable: false
        required: false
        schema:
          slug:
            required: true
            type: string
          type:
            allowed:
              - service
            required: true
            type: string
        type: dict
      env:
        required: false
        schema:
          type: dict
        type: list
      freeform_tag:
        required: false
        type: dict
      scriptPathURI:
        required: true
        type: string
      entrypoint:
        required: false
        type: string
    type: dict
  type:
    allowed:
      - script
    required: true
    type: string
