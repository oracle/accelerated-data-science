===========
Quick Start
===========

Install (locally)
-----------------

Install ads using pip (shown below) or OCI Conda Packs (see :doc:`Installation <./install>`)

.. code-block:: bash

    python3 -m pip install "oracle_ads[forecast]"

Install (Notebook Session)
--------------------------

Go to ``Envirnoment Explorer`` and search for ``AI Forecasting``. Install that conda pack. Activate the conda pack. For example:

.. code-block:: bash

    odsc conda install -s forecast_v3
    conda activate /home/datascience/forecast_v3


Initialize
----------

Initialize your forecast through the ads cli command. This will create several configuration files which can later be used to run the operators on OCI Data Science Jobs.

.. code-block:: bash

   ads operator init -t forecast --output my-forecast


Input Data 
-----------

Within the ``forecast`` folder created above there will be a ``forecast.yaml`` file. This file should be updated to contain the details about your data and forecast. Prophet's Yosemite Temperature dataset is provided as an example below:

.. code-block:: bash

   cd my-forecast
   vi forecast.yaml

.. code-block:: yaml

    kind: operator
    type: forecast
    version: v1
    spec:
        datetime_column:
            name: ds
        historical_data:
            url: https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv
        horizon: 3
        model: prophet
        target_column: y

There are many more options in this :doc:`YAML file <./yaml_schema>`.


Run
---

Now run the forecast locally:

.. code-block:: bash

    ads operator run -f forecast.yaml


Results
-------

If not specified in the YAML, all results will be placed in a new folder called ``results``. Performance is summarized in the ``report.html`` file, and the full forecast is available in the ``forecast.csv`` file.

.. code-block:: bash

    open results/report.html
