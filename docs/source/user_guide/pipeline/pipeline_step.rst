.. _pipeline_step:

Pipeline Step
*************

Pipeline step is a task in a pipeline. A pipeline step can be one of two types:

- Data Science Job: the OCID of an existing Data Science Job must be provided.
- Custom Script: the artifact of the Python script and the execution configuration must be specified.

This section shows how you can use the ADS Pipeline APIs to create pipeline steps. 

Data Science Job Step
=====================

Create a Data Science Job step with the OCID of an existing Job.

.. tabs::

  .. code-tab:: YAML

    kind: pipeline
    spec:
      ...
      stepDetails:
      ...
      - kind: dataScienceJob
        spec:
          description: <pipeline_step_description>
          jobId: ocid1.datasciencejob..<unique_id>	
          name: <pipeline_step_name>
      ...
    type: pipeline
    
  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import PipelineStep

    pipeline_step = (
        PipelineStep("<pipeline_step_name>")
        .with_description("<pipeline_step_description>")
        .with_job_id("<job_id>")
    )


Custom Script Step
==================

To create a Custom Script step, ``infrastructure`` and ``runtime`` must be specified.

The Custom Script step ``infrastructure`` is defined by a ``CustomScriptStep`` instance. 
When constructing a Custom Scrip step ``infrastructure``, you specify the Compute shape, Block Storage size in the ``CustomScriptStep`` instance. For example:


.. tabs::

  .. code-tab:: YAML


    kind: pipeline
    spec:
      ...
      stepDetails:
      - kind: customScript
        spec:
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: Python_Script_Step
       ...
    type: pipeline
    

  .. code-tab:: Python3
    :caption: Python
  
    from ads.pipeline import CustomScriptStep

    infrastructure = (
      CustomScriptStep()
      .with_block_storage_size(200)
      .with_shape_name("VM.Standard3.Flex")
      .with_shape_config_details(ocpus=4, memory_in_gbs=32)
    )



A Custom Script step can have different types of ``runtime`` depending on the source code you run:

.. include:: ../jobs/components/runtime_non_byoc.rst

All of these runtime options allow you to configure a `Data Science Conda Environment <https://docs.oracle.com/iaas/data-science/using/conda_understand_environments.htm>`__ for running your code. 

To define a Custom Script step with ``GitPythonRuntime`` you can use:


.. tabs::

  .. code-tab:: YAML

	kind: runtime
	spec:
	  conda:
		slug: pytorch19_p37_gpu_v1
		type: service
	  entrypoint: beginner_source/examples_nn/polynomial_nn.py
	  env:
	  - name: GREETINGS
		value: Welcome to OCI Data Science
	  outputDir: ~/Code/tutorials/beginner_source/examples_nn
	  outputUri: oci://<bucket_name>@<namespace>/<prefix>
	  url: https://github.com/pytorch/tutorials.git
	type: gitPython


  .. code-tab:: Python3
    :caption: Python
  
    from ads.pipeline import GitPythonRuntime

    runtime = (
        GitPythonRuntime()
        .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
        .with_service_conda("pytorch19_p37_gpu_v1")
        .with_source("https://github.com/pytorch/tutorials.git")
        .with_entrypoint("beginner_source/examples_nn/polynomial_nn.py")
        .with_output(
        output_dir="~/Code/tutorials/beginner_source/examples_nn",
        output_uri="oci://<bucket_name>@<namespace>/<prefix>"
      )
    )

To define a Custom Script step with ``NotebookRuntime`` you can use:


.. tabs::

  .. code-tab:: YAML

	kind: runtime
	spec:
	  conda:
		slug: tensorflow26_p37_cpu_v2
		type: service
	  env:
	  - name: GREETINGS
		value: Welcome to OCI Data Science
	  notebookEncoding: utf-8
	  notebookPathURI: https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb
	  outputURI: oci://bucket_name@namespace/path/to/dir
	type: notebook

  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import NotebookRuntime

    runtime = (
        NotebookRuntime()
          .with_notebook(
              path="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/customization/basics.ipynb",
              encoding='utf-8'
          )
          .with_service_conda("tensorflow26_p37_cpu_v2")
          .with_environment_variable(GREETINGS="Welcome to OCI Data Science")
          .with_output("oci://bucket_name@namespace/path/to/dir")
    )

To define a Custom Script step with ``PythonRuntime`` you can use:


.. tabs::

  .. code-tab:: YAML

	kind: runtime
	spec:
	  conda:
		slug: pytorch19_p37_cpu_v1
		type: service
	  entrypoint: zip_or_dir/my_package/entry.py
	  outputDir: output
	  outputUri: oci://bucket_name@namespace/path/to/dir
	  pythonPath:
	  - my_python_packages
	  scriptPathURI: local/path/to/zip_or_dir
	  workingDir: zip_or_dir
	type: python

  .. code-tab:: Python3
    :caption: Python
  
    from ads.pipeline import PythonRuntime

    runtime = (
        PythonRuntime()
        .with_service_conda("pytorch19_p37_cpu_v1")
        # The job artifact directory is named "zip_or_dir"
        .with_source("local/path/to/zip_or_dir", entrypoint="zip_or_dir/my_package/entry.py")
        # Change the working directory to be inside the job artifact directory
        # Working directory a relative path from the parent of the job artifact directory
        # Working directory is also added to Python paths
        .with_working_dir("zip_or_dir")
        # Add an additional Python path
        # The "my_python_packages" folder is under "zip_or_dir" (working directory)
        .with_python_path("my_python_packages")
        # Files in "output" directory will be copied to OCI object storage once the job finishes
        # Here we assume "output" is a folder under "zip_or_dir" (working directory)
        .with_output("output", "oci://bucket_name@namespace/path/to/dir")
    )

To define a Custom Script step with ``ScriptRuntime`` you can use:

.. tabs::

  .. code-tab:: YAML
  
	kind: runtime
	spec:
	  conda:
		slug: tensorflow26_p37_cpu_v2
		type: service
	  scriptPathURI: oci://<bucket_name>@<namespace>/<prefix>/<script.py>
	type: script

  .. code-tab:: Python3
    :caption: Python
  
    from ads.pipeline import ScriptRuntime

    runtime = (
        ScriptRuntime()
        .with_source("oci://<bucket_name>@<namespace>/<prefix>/<script.py>")
        .with_service_conda("tensorflow26_p37_cpu_v2")
    )

With ``Infrastructure`` and ``runtime`` provided, create a pipeline step of the Custom Script type.

.. tabs::

  .. code-tab:: YAML

    kind: pipeline
    spec:
      ...
      stepDetails:
      ...
      - kind: customScript
        spec:
          description: <pipeline_step_description>
          infrastructure:
            kind: infrastructure
            spec:
              blockStorageSize: 200
              shapeConfigDetails:
                memoryInGBs: 32
                ocpus: 4
              shapeName: VM.Standard3.Flex
          name: <pipeline_step_name>
          runtime:
            kind: runtime
            spec:
              conda:
                slug: <slug>
                type: service
              scriptPathURI: oci://<bucket_name>@<namespace>/<prefix>/<script.py>
            type: script
      ...
    type: pipeline
    
  .. code-tab:: Python3
    :caption: Python

    from ads.pipeline import PipelineStep

    pipeline_step = (
        PipelineStep("<pipeline_step_name>")
        .with_description("<pipeline_step_description>")
        .with_infrastructure(infrastructure)
        .with_runtime(runtime)
    )


