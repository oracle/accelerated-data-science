============
Installation
============

The Forecast Operator can be installed using two primary methods: via PyPi or Conda Packs.

Installing Through PyPi
------------------------

**Notebook Session or Local Environment**

If you are running the Operator outside of a Notebook Session, you can install it by downloading ``oracle_ads[forecast]`` from PyPi.

*Note: Python 3.11 is recommended.*

To install, run the following command:

.. code-block:: bash

    python3 -m pip install "oracle_ads[forecast]"

After installation, the Operator will be ready to use.

**Job and Model Deployment**

Jobs and Model Deployments can also use conda packs for their workloads. This can be used automatically from a Notebook Session by running:

.. code-block:: bash

    ads operator run -f forecast.yaml -b job

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

    conda activate /home/datascience/conda/forecast_p311_cpu_x86_64_v6

That's it. Your environment is ready to run Operators!
