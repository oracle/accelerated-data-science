====================================
Installing the AI Forecast Operator
====================================

The Forecast Operator can be installed in 2 primary ways: PyPi and Conda Packs.


**Installing Through Conda Packs**

The service recommended environment for using Operators is through Conda Packs within a Job or Notebook Session on OCI.

To install:

1. Open a Notebook Session
2. Go to Environment Explorer (from the Launcher tab)
3. Search for ``forecast``
4. Download the latest version by clicking the download button.
5. Activate the conda environment using the path, for example:

.. code-block:: bash

    conda activate /home/datascience/conda/forecast_py38_v1


That's it. Your Operator is ready to go!


**Installing Through PyPi**

If you are running the operator from outside of a Notebook Session, you may download ``oracle_ads[forecasting]`` from pypi. 

.. code-block:: bash

    python3 -m pip install oracle_ads[forecasting]


After that, the Operator is ready to go!
