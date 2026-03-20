===========
Recommender
===========

The Recommender Operator is a low-code template built around the Surprise ``SVD`` algorithm for collaborative filtering use cases. It currently focuses on matrix factorization scenarios, and additional capabilities will be documented as they become available.

Current Capabilities
--------------------

- **Data inputs**: expects three tabular sources named ``users``, ``items``, and ``interactions``. Each source can be loaded from local files, OCI Object Storage (``oci://`` URIs), or database queries using the standard ADS ``InputData`` configuration.
- **Model**: wraps Surprise ``SVD`` with sensible defaults. ``spec.model_name`` is reserved for future extensibility and is pinned to ``svd`` internally.
- **Outputs**: generates a recommendations CSV (``recommendations.csv`` by default) and, when enabled, an HTML summary report.
- **Configuration essentials**: ``top_k``, ``user_column``, ``item_column``, and ``interaction_column`` are mandatory and map your datasets to the operator.
- **Deployment targets**: supports local execution and OCI Data Science Jobs; see :doc:`./quickstart` for the CLI flow and :doc:`./scalability` for production guidance.

Future Updates
--------------

New capabilities—such as alternative algorithms, advanced tuning controls, or expanded deployment guidance—will be documented in this guide as they are released.

.. versionadded:: 2.11.14

.. toctree::
  :maxdepth: 1

  ./quickstart
  ./yaml_schema
  ./scalability
