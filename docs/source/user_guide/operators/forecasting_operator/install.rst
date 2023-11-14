====================================
Installing the AI Forecast Operator
====================================

The Forecast Operator can be installed in 2 primary ways: PyPi and Conda Packs.


**Installing Through PyPi**

If you are running the operator from outside of a Notebook Session, you may download ``oracle_ads[forecast]`` from pypi. 

.. code-block:: bash

    python3 -m pip install oracle_ads[forecast]==2.9.0rc1


After that, the Operator is ready to go!

In order to run on a job, you will need to create and publish a conda pack with ``oracle_ads[forecast]`` installed. The simplest way to do this is from a Notebook Session, running the following commands:

.. code-block:: bash

    odsc conda create -n forecast -e
    conda activate /home/datascience/conda/forecast_v1_0
    python3 -m pip install oracle-ads[forecast]==2.9.0rc1
    odsc conda publish -s /home/datascience/conda/forecast_v1_0

Ensure that you have properly configured your conda pack namespace and bucket in the Launcher -> Settings -> Object Storage Settings. For more details, see :doc:`ADS Conda Set Up <../../cli/opctl/configure>`


**Installing Through Conda Packs**

*Coming Soon!* The Forecast Conda Pack will be released on December 1, 2023.

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
