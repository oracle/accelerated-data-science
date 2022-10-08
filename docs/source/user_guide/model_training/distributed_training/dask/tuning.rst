Dask Cluster Tuning
-------------------

Configuring dask startup options
++++++++++++++++++++++++++++++++

Dask scheduler
~~~~~~~~~~~~~~

Dask scheduler is launched with ``dask-scheduler`` command. By default no arguments are supplied to ``dask-scheduler``.
You could influence the startup option by adding them to ``startOptions`` under ``cluster/spec/main/config`` section of the cluster YAML definition

Eg. Here is how you could change the scheduler port number:

.. code-block:: yaml

  # Note only portion of the yaml file is shown here for brevity.
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: region.ocir.io/my-tenancy/image:tag
      workDir: "oci://my-bucket@my-namespace/daskcluster-testing/005"
      ephemeral: True
      name: My Precious
      main:
        config:
          startOptions:
            - --port 8788


Dask worker
~~~~~~~~~~~

Dask worker is launched with ``dask-worker`` command. By default no arguments are supplied to ``dask-worker``.
You could influence the startup option by adding them to ``startOptions`` under ``cluster/spec/worker/config`` section of the cluster YAML definition

Eg. Here is how you could change the worker port, nanny port, number of workers per host and number of threads per process:

.. code-block:: yaml

  # Note only portion of the yaml file is shown here for brevity.
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: region.ocir.io/my-tenancy/image:tag
      workDir: "oci://my-bucket@my-namespace/daskcluster-testing/005"
      ephemeral: True
      name: My Precious
      main:
        config:
      worker:
        config:
          startOptions:
            - --worker-port 8700:8800
            - --nanny-port 3000:3100
            - --nworkers 8
            - --nthreads 2

`Refer to the complete list <https://docs.dask.org/en/latest/deploying-cli.html#cli-options>`_

Configuration through Environment Variables
+++++++++++++++++++++++++++++++++++++++++++

You could set configuration parameters that Dask recognizes by add it to ``cluster/spec/config/env`` or ``cluster/spec/main/config/env`` or ``cluster/spec/worker/config/env``
If a configuration value is some for both ``scheduler`` and ``worker`` section, then set it at ``cluster/spec/config/env`` section.

.. code-block:: yaml

 # Note only portion of the yaml file is shown here for brevity.
  cluster:
    kind: dask
    apiVersion: v1.0
    spec:
      image: region.ocir.io/my-tenancy/image:tag
      workDir: "oci://my-bucket@my-tenancy/daskcluster-testing/005"
      ephemeral: True
      name: My Precious
      config:
        env:
          - name: DASK_ARRAY__CHUNK_SIZE
            value: 128 MiB
          - name: DASK_DISTRIBUTED__WORKERS__MEMORY__SPILL
            value: 0.85
          - name: DASK_DISTRIBUTED__WORKERS__MEMORY__TARGET
            value: 0.75
          - name: DASK_DISTRIBUTED__WORKERS__MEMORY__TERMINATE
            value: 0.98



Refer `here <https://docs.dask.org/en/stable/configuration.html#environment-variables>`_ for more information
