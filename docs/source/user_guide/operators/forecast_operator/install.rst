============
Installation
============

The Forecast Operator can be installed using two primary methods: via PyPi or Conda Packs.

Installing Through PyPi
------------------------

**Notebook Session or Local Environment**

If you are running the Operator outside of a Notebook Session, you can install it by downloading ``oracle_ads[forecast]`` from PyPi.

*Note: Python 3.10 is recommended.*

To install, run the following command:

.. code-block:: bash

    python3 -m pip install "oracle_ads[forecast]"

After installation, the Operator will be ready to use.

**Job and Model Deployment**

To run the Operator in a job, you need to create and publish a Conda pack with ``oracle_ads[forecast]`` installed. The simplest way to do this is from a Notebook Session using the following commands:

.. code-block:: bash

    odsc conda create -n forecast -e
    conda activate /home/datascience/conda/forecast
    python3 -m pip install "oracle-ads[forecast]"
    odsc conda publish -s /home/datascience/conda/forecast

Ensure that you have properly configured your Conda pack namespace and bucket in the Launcher -> Settings -> Object Storage Settings. For more details, see :doc:`ADS Conda Set Up <../../cli/opctl/configure>`.

Installing Through Conda Packs
------------------------------

The recommended environment for using Operators is through Conda Packs within a Job or Notebook Session on OCI.

To install:

1. Open a Notebook Session.
2. Go to the Environment Explorer (from the Launcher tab).
3. Search for ``forecast``.
4. Download the latest version by clicking the download button.
5. Activate the Conda environment using the specified path, for example:

.. code-block:: bash

    conda activate /home/datascience/conda/forecast_p310_cpu_x86_64_v4

That's it. Your Operator is ready to go!
