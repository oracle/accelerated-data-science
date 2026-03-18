===========
Recommender
===========

The Recommender Operator is an *experimental* low-code template that packages a collaborative filtering workflow around a single algorithm (matrix factorization via Surprise ``SVD``). It is intended to validate customer interest quickly; as a result, the feature set is intentionally narrow and will evolve based on intake requests from strategic customers.

Current Capabilities
--------------------

- **Data inputs**: expects three tabular sources named ``users``, ``items``, and ``interactions``. Each source can be loaded from local files, OCI Object Storage (``oci://`` URIs), or database queries using the standard ADS ``InputData`` configuration.
- **Model**: wraps Surprise ``SVD`` with sensible defaults. ``spec.model_name`` is reserved for future extensibility and is pinned to ``svd`` internally.
- **Outputs**: generates a recommendations CSV (``recommendations.csv`` by default) and, when enabled, an HTML summary report.
- **Configuration essentials**: ``top_k``, ``user_column``, ``item_column``, and ``interaction_column`` are mandatory and map your datasets to the operator.
- **Deployment targets**: supports local execution and OCI Data Science Jobs; see :doc:`./quickstart` for the CLI flow and :doc:`./scalability` for production guidance.

Roadmap and Feedback
--------------------

Because this operator is experimental, we prioritize enhancements that come through the formal intake process. If your customers need alternative algorithms, richer tuning controls, or production blueprints, please log an intake request so we can size and prioritize the work.

.. versionadded:: 2.11.14

.. toctree::
  :maxdepth: 1

  ./quickstart
  ./yaml_schema
  ./scalability
