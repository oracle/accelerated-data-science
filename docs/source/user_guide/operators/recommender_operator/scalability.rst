===========
Scalability
===========

Cloud-Native Execution
----------------------

You can promote the same ``recommender.yaml`` from local development to OCI Data Science Jobs without rewriting your configuration.

.. code-block:: bash

    # run locally for quick validation
    ads operator run -f recommender.yaml

    # submit to OCI Data Science Jobs (serverless)
    ads operator run -f recommender.yaml -b job

The ``-b job`` flag uses your default job backend profile. Override shape, block storage, or networking by merging a backend config, for example:

.. code-block:: bash

    ads operator run -f recommender.yaml -b backend_job_python_config.yaml

For detailed backend options see :doc:`../common/run`.

Data Throughput and Storage
---------------------------

- Use Object Storage (``oci://`` URIs) for large interaction logs. The operator streams data through ADS I/O utilities, so you are limited primarily by network bandwidth.
- For database sources, push filtering and aggregation into the ``sql`` statement to minimise data transfer. Supply ``connect_args`` such as ``wallet_dir`` or ``dsn`` for Autonomous Database connectivity.
- When writing outputs back to Object Storage, point ``spec.output_directory.url`` to an ``oci://`` URI so downstream AI Skills or Jobs can consume the artifacts.

Batch Size and Latency
----------------------

Surprise ``SVD`` trains in-memory on the interaction matrix. To keep runs tractable:

- Start with filtered cohorts (for example, a single region or product line) to validate signal before scaling out.
- Increase compute shape (more OCPUs / memory) in the job backend when interaction counts grow beyond hundreds of thousands.
- Consider sharding your audience and running the operator multiple times if you need very large coverage; you can merge the resulting recommendation CSVs downstream.

Operational Tips
----------------

- Set ``spec.generate_report`` to ``false`` for automated batch runs to reduce artifact size.
- Version control your YAML files and backend configs alongside infrastructure-as-code scripts so intake reviews can track exactly how the operator is used.
- Monitor job logs in OCI Data Science to confirm the operator runs within expected time windows and to capture Surprise training diagnostics.
