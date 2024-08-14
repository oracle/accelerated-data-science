===========
Quick Start
===========

Local Installation
------------------

Install the ADS library using ``pip`` (as shown below) or through OCI Conda Packs (see :doc:`Installation <./install>`).

.. code-block:: bash

    python3 -m pip install "oracle_ads[forecast]"

Installation in a Notebook Session
----------------------------------

1. Open ``Environment Explorer`` and search for ``AI Forecasting``.
2. Install the relevant Conda pack.
3. Activate the Conda environment. For example:

.. code-block:: bash

    odsc conda install -s forecast_v3
    conda activate /home/datascience/forecast_v3


Initialization
--------------

Initialize your forecast project using the ADS CLI command. This command will create several configuration files that can later be used to run the operators on OCI Data Science Jobs.

.. code-block:: bash

    ads operator init -t forecast --output my-forecast


Input Data 
----------

Within the ``my-forecast`` folder created above, you'll find a ``forecast.yaml`` file. Update this file with the details of your data and forecast. Below is an example using Prophet's Yosemite Temperature dataset:

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

There are many more options available in this :doc:`YAML file <./yaml_schema>`.


Running the Forecast
--------------------

Run the forecast locally using the following command:

.. code-block:: bash

    ads operator run -f forecast.yaml


Viewing Results
---------------

If the YAML configuration does not specify an output directory, all results will be placed in a new folder called ``results``. The performance summary is provided in the ``report.html`` file, and the full forecast is available in the ``forecast.csv`` file.

.. code-block:: bash

    open results/report.html
