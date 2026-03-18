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


Prepare Input Data
-------------------

The Recommender Operator requires three essential input files:

1. **Users File**: Contains user information.
2. **Items File**: Contains item information.
3. **Interactions File**: Interactions between users and items.

.. note::

   You can keep these sources local during prototyping and later point to Object Storage or Oracle databases without changing the operator code—only the YAML configuration.

Remote Data Sources
-------------------

Point to Object Storage by swapping the local ``url`` with an ``oci://`` URI:

.. code-block:: yaml

    user_data:
      url: oci://my-bucket@my-namespace/users.csv
    item_data:
      url: oci://my-bucket@my-namespace/items.csv

Read directly from Autonomous Database (or another Oracle database) using ``sql`` and ``connect_args``:

.. code-block:: yaml

    interactions_data:
      sql: |
        SELECT user_id, movie_id, rating, event_ts
        FROM MOVIE_RECS.INTERACTIONS
      connect_args:
        wallet_dir: /home/datascience/oci_wallet

Sample Data
===========

**users.csv**:

=========  ===  ======  ============  =========
user_id    age  gender  occupation    zip_code
=========  ===  ======  ============  =========
1          24   M       technician    85711
2          53   F       other         94043
3          23   M       writer        32067
4          24   M       technician    43537
5          33   F       other         15213
=========  ===  ======  ============  =========

**items.csv**:

===========  =================  ============  ======  =========  ==========  ========
movie_id     movie_title        release_date  Action  Adventure  Animation   Children
===========  =================  ============  ======  =========  ==========  ========
1            Toy Story (1995)    01-Jan-1995   0       0          1          1
2            GoldenEye (1995)    01-Jan-1995   1       1          0          0
3            Four Rooms (1995)   01-Jan-1995   0       0          0          0
4            Get Shorty (1995)   01-Jan-1995   1       0          0          0
===========  =================  ============  ======  =========  ==========  ========

**interactions.csv**:

=======  =========  ======  ============
user_id  movie_id   rating  timestamp
=======  =========  ======  ============
2        1          3       881250949
4        2          3       891717742
3        3          1       878887116
1        4          2       880606923
5        2          1       886397596
2        3          4       884182806
4        1          2       881171488
=======  =========  ======  ============


Configure the YAML File
----------------------

Within the ``recommender`` folder created above there will be a ``recommender.yaml`` file. This file should be updated to contain the details about your data and recommender.

.. code-block:: bash

   cd recommender
   vi recommender.yaml

.. code-block:: yaml

    kind: operator
    type: recommender
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
      output_directory:
        url: results
      recommendations_filename: recommendations.csv
      generate_report: true

Run the Recommender Operator
----------------------------

Validate the YAML and run locally:

.. code-block:: bash

    ads operator validate -f recommender.yaml
    ads operator run -f recommender.yaml

Run as an OCI Data Science Job
------------------------------

When you are ready to scale, submit the same YAML to a managed job backend:

.. code-block:: bash

    ads operator run -f recommender.yaml -b job

Use ``-b`` with a backend config (for example, ``backend_job_python_config.yaml``) to specify shape, subnet, or other runtime controls. See :doc:`../common/run` for backend details.

Results
-------

If not specified in the YAML, all results will be placed in a new folder called ``results``. Performance is summarized in the ``report.html`` file, and the recommendation results can be found in ``results/recommendations.csv``.

.. code-block:: bash

    vi results/recommendations.csv
    open results/report.html

Example Output (recommendations.csv):
====================================

=======  =========  ======
user_id  movie_id   rating
=======  =========  ======
1        1          4.9424
1        2          4.7960
1        3          4.7314
1        4          4.6951
2        1          4.7893
2        2          4.7870
2        3          4.7624
2        4          4.6802
=======  =========  ======
