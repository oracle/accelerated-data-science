===========
Quick Start
===========

Install
--------

Install ads using pip (shown below) or OCI Conda Packs (see :doc:`Installation <./install>`)

.. code-block:: bash

    python3 -m pip install "oracle_ads[anomaly]"

Initialize
----------

Initialize your anomaly detection job through the ads cli command:

.. code-block:: bash

   ads operator init -t anomaly


Input Data 
-----------

Within the ``anomaly`` folder created above there will be a ``anomaly.yaml`` file. This file should be updated to contain the details about your data and anomaly. Prophet's Yosemite Temperature dataset is provided as an example below:

.. code-block:: bash

   cd anomaly
   vi anomaly.yaml

.. code-block:: yaml

    kind: operator
    type: anomaly
    version: v1
    spec:
        datetime_column:
            name: timestamp
        target_category_columns:
            - series_id
        input_data:
            url: https://raw.githubusercontent.com/oracle/accelerated-data-science/refs/heads/main/ads/opctl/operator/common/data/synthetic.csv
        model: autots
        target_column: target

There are many more options in this :doc:`YAML file <./yaml_schema>`.


Run
---

Now run the anomaly detection job locally:

.. code-block:: bash

    ads operator run -f anomaly.yaml


Results
-------

If not specified in the YAML, all results will be placed in a new folder called ``results``. Performance is summarized in the ``report.html`` file.

.. code-block:: bash

    open results/report.html
