============
Installation
============

The Anomaly Detection Operator can be installed in 2 primary ways: PyPi and Conda Packs.


Installing Through PyPi
------------------------

**Notebook Session or local**

If you are running the operator from outside of a Notebook Session, you may download ``oracle_ads[anomaly]`` from pypi. 

*Note: Due to our dependence on Automlx, ``oracle_ads[anomaly]``  only supports Python<=3.8, != 3.9, <= 3.10.7 . Python 3.8 is the recommended version.*

.. code-block:: bash

    python3 -m pip install "oracle_ads[anomaly]"


After that, the Operator is ready to go!

**Job and Model Deployment**

In order to run on a job, you will need to create and publish a conda pack with ``oracle_ads[anomaly]`` installed. The simplest way to do this is from a Notebook Session, running the following commands:

.. code-block:: bash

    odsc conda create -n anomaly -e
    conda activate /home/datascience/conda/anomaly_v1_0
    python3 -m pip install "oracle-ads[anomaly]"
    odsc conda publish -s /home/datascience/conda/anomaly_v1_0

Ensure that you have properly configured your conda pack namespace and bucket in the Launcher -> Settings -> Object Storage Settings. For more details, see :doc:`ADS Conda Set Up <../../cli/opctl/configure>`


Installing Through Conda Packs
------------------------------

The service recommended environment for using Operators is through Conda Packs within a Job or Notebook Session on OCI.

To install:

1. Open a Notebook Session
2. Go to Environment Explorer (from the Launcher tab)
3. Search for ``anomaly``
4. Download the latest version by clicking the download button.
5. Activate the conda environment using the path, for example:

.. code-block:: bash

    conda activate /home/datascience/conda/anomaly_py38_v1


That's it. Your Operator is ready to go!
