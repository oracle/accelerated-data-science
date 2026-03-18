===========
YAML Schema
===========

The ``recommender.yaml`` file orchestrates data access, configuration, and output options for the Recommender Operator. This section walks through every top-level field so you can adapt the template to your environment.

Example Configuration
---------------------

.. code-block:: yaml

    kind: operator
    type: recommender
    version: v1
    spec:
      user_data:
        url: oci://my-bucket@my-namespace/users.csv
      item_data:
        url: oci://my-bucket@my-namespace/items.csv
      interactions_data:
        sql: |
          SELECT user_id, movie_id, rating, event_ts
          FROM MOVIE_RECS.INTERACTIONS
        connect_args:
          wallet_dir: /home/datascience/oci_wallet
      top_k: 10
      user_column: user_id
      item_column: movie_id
      interaction_column: rating
      recommendations_filename: recommendations.csv
      generate_report: true

Configuration Reference
-----------------------

.. list-table:: Recommender Operator Specification
   :widths: 20 10 10 20 40
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Default
     - Description

   * - user_data
     - dict
     - Yes
     - {"url": "user_data.csv"}
     - Source for user attributes. Accepts the standard ADS ``InputData`` options such as ``url``, ``sql``, ``table_name``, ``connect_args``, and column filters. Remote URIs (``oci://``) and database queries are both supported.

   * - item_data
     - dict
     - Yes
     - {"url": "item_data.csv"}
     - Source for item attributes. Shares the same structure and connectivity options as ``user_data``.

   * - interactions_data
     - dict
     - Yes
     - {"url": "interactions_data.csv"}
     - Historical interactions between users and items. Use this to supply implicit or explicit feedback (for example, ratings or click events). Supports the same loaders as ``user_data``.

   * - top_k
     - integer
     - Yes
     - 1
     - Number of recommendations returned per user. Increase this when downstream applications (such as AI Skills) need a wider candidate list.

   * - user_column
     - string
     - Yes
     - user_id
     - User identifier column present in both ``user_data`` and ``interactions_data``.

   * - item_column
     - string
     - Yes
     - item_id
     - Item identifier column present in both ``item_data`` and ``interactions_data``.

   * - interaction_column
     - string
     - Yes
     - rating
     - Interaction strength column used to train Surprise ``SVD``. For implicit feedback, convert events to a numeric score before loading.

   * - output_directory
     - dict
     - No
     - Auto-generated temp path
     - Controls where artifacts are written. Provide ``url`` (local path or ``oci://``) and optional ``name`` to organize outputs. Leave unset to let ADS create a timestamped local directory.

   * - recommendations_filename
     - string
     - No
     - recommendations.csv
     - Customise the recommendations artifact name inside ``output_directory``.

   * - generate_report
     - boolean
     - No
     - true
     - Toggles HTML report creation. Disable when running headless jobs where only CSV output is required.

   * - report_filename
     - string
     - No
     - report.html
     - Name of the HTML summary report file saved under ``output_directory``.

   * - model_name
     - string
     - No
     - svd
     - Reserved for future model expansion. The only supported value today is ``svd``; other values raise ``UnSupportedModelError``.

.. note::

   The operator validates the schema before execution. If you pass extra keys, they will be ignored or trigger a validation error. Use the ``ads operator validate -f recommender.yaml`` command to catch issues early.
