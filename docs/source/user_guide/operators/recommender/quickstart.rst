===========
Quick Start
===========

Install
--------

Install ads using pip (shown below) or OCI Conda Packs

.. code-block:: bash

    python3 -m pip install "oracle_ads"

Initialize
----------

Initialize your recommender job through the ads cli command:

.. code-block:: bash

   ads operator init -t recommender


Input Data
-----------

Within the ``recommender`` folder created above there will be a ``recommender.yaml`` file. This file should be updated to contain the details about your data and recommender.

.. code-block:: bash

   cd recommender
   vi recommender.yaml

.. code-block:: yaml

    kind: operator
    type: recommendation
    version: v1
    spec:
      user_data:
        url: users.csv
      item_data:
        url: items.csv
      interactions_data:
        url: interactions.csv
      top_k: 4
      user_column: user_id
      item_column: movie_id
      interaction_column: rating


Run
---

Now run the recommender job locally:

.. code-block:: bash

    ads operator run -f recommender.yaml


Results
-------

If not specified in the YAML, all results will be placed in a new folder called ``results``. Performance is summarized in the ``report.html`` file.

.. code-block:: bash

    open results/report.html