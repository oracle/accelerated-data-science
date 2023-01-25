+++++++++++++++++++
Local Job Execution
+++++++++++++++++++

Your job can be executed in a local container to facilitate development and troubleshooting.

-------------
Prerequisites
-------------

1. :doc:`Install ADS CLI<../../quickstart>`
2. Build a container image.
    - :doc:`Build Development Container Image<./jobs_container_image>` and :doc:`install a conda environment<./condapack>`
    - :doc:`Build Your Own Container (BYOC)<./jobs>`

------------
Restrictions
------------

When running locally, your job is subject to the following restrictions:
  - The job must use API Key auth. Resource Principal auth is not supported in a local container. See https://docs.oracle.com/iaas/Content/API/Concepts/apisigningkey.htm
  - You can only use conda environment published to your own Object Storage bucket. See :doc:`Working with Conda packs<./condapack>`
  - Your job files must be present on your local machine.
  - Any network calls must be reachable by your local machine. (i.e. Your job cannot connect to an endpoint that is only reachable within the job's subnet.)
  - Your local machine meets the hardware requirements of your job.

----------------
Running your Job
----------------

Using a conda environment
=========================

This example below demonstrates how to run a local job using an installed conda environment:

.. code-block:: shell

  ads opctl run --backend local --conda-slug myconda_p38_cpu_v1 --source-folder /path/to/my/job/files/ --entrypoint bin/my_script.py --cmd-args "--some-arg" --env-var "MY_VAR=12345"

Parameter explanation:
  - ``--backend local``: Run the job locally in a docker container.
  - ``--conda-slug myconda_p38_cpu_v1``: Use the ``myconda_p38_cpu_v1`` conda environment. Note that you must install this conda environment locally first.
    The local conda environment directory will be automatically mounted into the container and activated before the entrypoint is executed.
  - ``--source-folder /path/to/my/job/files/``: The local directory containing your job files. This directory is mounted into the container as a volume.
  - ``--entrypoint bin/my_script.py``: Set the container entrypoint to ``bin/my_script.py``. Note that this path is relative to the path specified with the ``--source-folder`` parameter.
  - ``--cmd-args "--some-arg"``: Pass ``--some-arg`` to the container entrypoint.
  - ``--env-var "MY_VAR=12345": Define envrionment variable ``MY_VAR`` with value ``12345``.

Using a custom image
====================

This example below demonstrates how to run a local job using a custom container image:

.. code-block:: shell

  ads opctl run --backend local --image my_image --entrypoint /path/to/my/binary --command my_cmd --env-var "MY_VAR=12345"

Parameter explanation:
  - ``--backend local``: Run the job locally in a docker container.
  - ``--image my_image``: Use the custom container image named ``my_image``.
  - ``--entrypoint /path/to/my/binary``: Set the container entrypoint to ``/path/to/my/binary``. Note that this path is within the container image.
  - ``--command my_cmd``: Set the container command to ``my_cmd``.
  - ``--env-var "MY_VAR=12345": Define envrionment variable ``MY_VAR`` with value ``12345``.

Viewing container output
========================
When the container is running, you can use the ``docker logs`` command to view its output. See https://docs.docker.com/engine/reference/commandline/logs/

Alternatively, you can use the ``--debug`` parameter to print the container stdout/stderr messages to your shell. Note that Python buffers output by default, so you may see output written
to the shell in bursts. If you want to see output displayed in real-time, specify ``--env-var PYTHONUNBUFFERED=1``.

.. code-block:: shell

  ads opctl run --backend local --conda-slug myconda_p38_cpu_v1 --source-folder /path/to/my/job/files/ --entrypoint my_script.py --env-var "PYTHONUNBUFFERED=1" --debug

