++++++++++++++++++++++++
Local Pipeline Execution
++++++++++++++++++++++++

Your pipeline can be executed locally to facilitate development and troubleshooting. Each pipeline step is executed in its own local container.

-------------
Prerequisites
-------------

1. :doc:`Install ADS CLI<../../quickstart>`
2. :doc:`Build Development Container Image<./jobs_container_image>` and :doc:`install a conda environment<./condapack>`

------------
Restrictions
------------

Your pipeline steps are subject to the :doc:`same restrictions as local jobs<./local_jobs>`.

They are also subject to these additional restrictions:

  - Pipeline steps must be of kind ``customScript``.
  - Custom container images are not yet supported. You must use the development container image with a conda environment.

---------------------------------------
Configuring Local Pipeline Orchestrator
---------------------------------------

Use ``ads opctl configure``. Refer to the ``local_backend.ini`` description in the configuration :doc:`instructions<../configure>`.

Most importantly, ``max_parallel_containers`` controls how many pipeline steps may be executed in parallel on your machine. Your pipeline DAG may allow multiple steps to be executed in parallel,
but your local machine may not have enough cpu cores / memory to effectively run them all simultaneously.

---------------------
Running your Pipeline
---------------------

Local pipeline execution requires you to define your pipeline in a yaml file. Refer to the YAML examples :doc:`here<../../../pipeline/examples>`.

Then, invoke the following command to run your pipeline.

.. code-block:: shell

  ads opctl run --backend local --file my_pipeline.yaml --source-folder /path/to/my/pipeline/step/files

Parameter explanation:
  - ``--backend local``: Run the pipeline locally using docker containers.
  - ``--file my_pipeline.yaml``: The yaml file defining your pipeline.
  - ``--source-folder /path/to/my/pipeline/step/files``: The local directory containing the files used by your pipeline steps. This directory is mounted into the container as a volume.
    Defaults to the current working directory if no value is provided.

Source folder and relative paths
================================
If your pipeline step runtimes are of type ``script`` or ``notebook``, the paths in your yaml files must be relative to the ``--source-folder``.

Pipeline steps using a runtime of type ``python`` are able to define their own working directory that will be mounted into the step's container instead.

For example, suppose your yaml file looked like this:

.. code-block:: yaml

  kind: pipeline
  spec:
    displayName: example
    dag:
    - (step_1, step_2) >> step_3
    stepDetails:
    - kind: customScript
      spec:
        description: A step running a notebook
        name: step_1
        runtime:
          kind: runtime
          spec:
            conda:
              slug: myconda_p38_cpu_v1
              type: service
            notebookEncoding: utf-8
            notebookPathURI: step_1_files/my-notebook.ipynb
            type: notebook
    - kind: customScript
      spec:
        description: A step running a shell script
        name: step_2
        runtime:
          kind: runtime
          spec:
            conda:
              slug: myconda_p38_cpu_v1
              type: service
            scriptPathURI: step_2_files/my-script.sh
            type: script
    - kind: customScript
      spec:
        description: A step running a python script
        name: step_3
        runtime:
          kind: runtime
          spec:
            conda:
              slug: myconda_p38_cpu_v1
              type: service
            workingDir: /step_3/custom/working/dir
            scriptPathURI: my-python.py
            type: python
  type: pipeline

And suppose the pipeline is executed locally with the following command:

.. code-block:: shell

  ads opctl run --backend local --file my_pipeline.yaml --source-folder /my/files

``step_1`` uses a ``notebook`` runtime. The container for ``step_1`` will mount the ``/my/files`` directory into the container. The ``/my/files/step_1_files/my-notebook.ipynb`` notebook file
will be converted into a python script and executed in the container.

``step_2`` uses a ``script`` runtime. The container for ``step_2`` will mount the ``/my/files`` directory into the container. The ``/my/files/step_2_files/my-script.sh`` shell script will
be executed in the container.

``step_3`` uses a ``python`` runtime. Instead of mounting the ``/my/files`` directory specified by ``--source-folder``, the ``/step_3/custom/working/dir`` directory will be mounted into the
container. The ``/step_3/custom/working/dir/my-python.py`` script will be executed in the container.

Viewing container output and orchestration messages
===================================================
When a container is running, you can use the ``docker logs`` command to view its output. See https://docs.docker.com/engine/reference/commandline/logs/

Alternatively, you can use the ``--debug`` parameter to print each container's stdout/stderr messages to your shell. Note that Python buffers output by default, so you may see output written
to the shell in bursts. If you want to see output displayed in real-time for a particular step, specify a non-zero value for the ``PYTHONUNBUFFERED`` environment variable in your step's runtime
specification. For example:

.. code-block:: yaml

  - kind: customScript
    spec:
      description: A step running a shell script
      name: step_1
      runtime:
        kind: runtime
        spec:
          conda:
            slug: myconda_p38_cpu_v1
            type: service
          scriptPathURI: my-script.sh
          env:
            PYTHONUNBUFFERED: 1
        type: script


Pipeline steps can run in parallel. You may want your pipeline steps to prefix their log output to easily distinguish which lines of output are coming from which step.

When the ``--debug`` parameter is specified, the CLI will also output pipeline orchestration messages. These include messages about which steps are being started and a summary of each
step's result when the pipeline finishes execution.

