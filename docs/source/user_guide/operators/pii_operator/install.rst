===========================
Installing the PII Operator
===========================

The PII Operator can be installed from PyPi.


.. code-block:: bash

    python3 -m pip install "oracle_ads[pii]==2.9"


After that, the Operator is ready to go!

In order to run on a job, you will need to create and publish a conda pack with ``oracle_ads[pii]`` installed. The simplest way to do this is from a Notebook Session, running the following commands:

.. code-block:: bash

    odsc conda create -n ads_pii -e
    conda activate /home/datascience/conda/ads_pii_v1_0
    python3 -m "pip install oracle-ads[pii]==2.9"
    odsc conda publish -s /home/datascience/conda/ads_pii_v1_0

Ensure that you have properly configured your conda pack namespace and bucket in the Launcher -> Settings -> Object Storage Settings. For more details, see :doc:`ADS Conda Set Up <../../cli/opctl/configure>`
