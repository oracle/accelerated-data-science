Run a Notebook
**************

In some cases, you may want to run an existing JupyterLab notebook as a job. You can do this using the ``NotebookRuntime()`` object.

The next example shows you how to run an the `TensorFlow 2 quick start for beginner <https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb>`__ notebook from the internet and save the results to OCI Object Storage. The notebook path points to the raw file link from GitHub.  To run the following example, ensure that you have internet access to retrieve the notebook:

Python
======

.. code-block:: python3

    from ads.jobs import Job, DataScienceJob, NotebookRuntime

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
            NotebookRuntime()
            .with_notebook(
                path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
                encoding='utf-8'
            )
            .with_service_conda(tensorflow26_p37_cpu_v2")
            .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
            .with_output("oci://bucket_name@namespace/path/to/dir")
        )
    )

    job.create()
    run = job.run().watch()

After the notebook finishes running, the notebook with results are saved to ``oci://bucket_name@namespace/path/to/dir``.  You can download the output by calling the ``download()`` method.

.. code-block:: python3

    run.download("/path/to/local/dir")

The ``NotebookRuntime`` also allows you to use exclusion tags, which lets you exclude cells from a job run. For example, you could use these tags to do exploratory data analysis, and then train and evaluate your model in a notebook. Then you could use that same notebook to only build future models that are trained on a different dataset. So the job run only has to execute the cells that are related to training the model, and not the exploratory data analysis or model evaluation.

You tag the cells in the notebook, and then specify the tags using the ``.with_exclude_tag()`` method. Cells with any matching tags are excluded from the job run.  For example, if you tagged cells with ``ignore`` and ``remove``, you can pass in a list of the two tags to the method and those cells are excluded from the code that is executed as part of the job run. To tag cells in a notebook, see `Adding tags using notebook interfaces <https://jupyterbook.org/content/metadata.html#adding-tags-using-notebook-interfaces>`__.

.. code-block:: python3

    job.with_runtime(
        NotebookRuntime()
        .with_notebook("path/to/notebook")
        .with_exclude_tag(["ignore", "remove"])
    )

YAML
====

You could use the following YAML to create the job:

.. code-block:: yaml

	kind: job
	spec:
	  infrastructure:
	    kind: infrastructure
        type: dataScienceJob
	    spec:
	      jobInfrastructureType: STANDALONE
	      jobType: DEFAULT
	      logGroupId: <log_group_id>
	      logId: <log.id>
	  runtime:
	    kind: runtime
        type: notebook
	    spec:
          notebookPathURI: /path/to/notebook
	      conda:
            slug: tensorflow26_p37_cpu_v1
            type: service

**NotebookRuntime Schema**

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
        - notebook
    spec:
      required: true
      type: dict
      schema:
        excludeTags:
          required: false
          type: list
        notebookPathURI:
          required: false
          type: string
        notebookEncoding:
          required: false
          type: string
        outputUri:
          required: false
          type: string
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
              required: true
              type: string
              allowed:
                - service
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

