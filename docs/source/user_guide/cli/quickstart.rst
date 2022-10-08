======================
Installation and Setup
======================

~~~~~~~~~~~~~~~
Install ADS CLI
~~~~~~~~~~~~~~~

**Prerequisites**

* Linux/Mac (Intel CPU)
* For Mac on M series - Experimental.
* For Windows: Use `Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/windows/wsl/about>`_
* python >=3.7, <3.10

``ads`` cli provides a command line interface to Jobs API related features. Set up your development environment, build docker images compliant with Notebook session and Data Science Jobs, build and publish conda pack locally, start distributed training, etc.

.. admonition:: Installation

  Install ADS and enable CLI:

  .. code-block:: shell

    python3 -m pip install "oracle-ads[opctl]"

.. admonition:: Tip

  ``ads opctl`` subcommand lets us setup your local development envrionment for Data Science Jobs. More information can be found by running ``ads opctl -h``


~~~~~~~~~~~~~~~~~~~~~~~~~~
Install ``oracle-ads`` SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~

Data Science Conda Environments
===============================

ADS is installed in the data science conda environments. Upgrade your existing oracle-ads package by running -

.. code-block:: bash

    $ python3 -m pip install oracle-ads --upgrade

Install in Local Environments
=============================

You have various options when installing ADS.

Installing the ``oracle-ads`` base package
++++++++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    $ python3 -m pip install oracle-ads


Installing extras libraries
+++++++++++++++++++++++++++

The ``all-optional`` module will install all optional dependencies.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[all-optional]

To work with gradient boosting models, install the ``boosted`` module. This module includes XGBoost and LightGBM model classes.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[boosted]

For big data use cases using Oracle Big Data Service (BDS), install the ``bds`` module. It includes the following libraries: `ibis-framework[impala]`, `hdfs[kerberos]` and `sqlalchemy`.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[bds]

To work with a broad set of data formats (for example, Excel, Avro, etc.) install the ``data`` module. It includes the following libraries: `fastavro`, `openpyxl`, `pandavro`, `asteval`, `datefinder`, `htmllistparse`, and `sqlalchemy`.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[data]

To work with geospatial data install the ``geo`` module. It includes the `geopandas` and libraries from the `viz` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[geo]

Install the ``notebook`` module to use ADS within the Oracle Cloud Infrastructure Data Science service `Notebook Session <https://docs.oracle.com/en-us/iaas/data-science/using/manage-notebook-sessions.htm>`_. This module installs `ipywidgets` and `ipython` libraries.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[notebook]

To work with ONNX-compatible run times and libraries designed to maximize performance and model portability, install the ``onnx`` module. It includes the following libraries, `onnx`, `onnxruntime`, `onnxmltools`, `skl2onnx`, `xgboost`, `lightgbm` and libraries from the `viz` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[onnx]

For infrastructure tasks, install the ``opctl`` module. It includes the following libraries, `oci-cli`, `docker`, `conda-pack`, `nbconvert`, `nbformat`, and `inflection`.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[opctl]

For hyperparameter optimization tasks install the ``optuna`` module. It includes the `optuna` and libraries from the `viz` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[optuna]

For Spark tasks install the ``spark`` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[spark]

Install the ``tensorflow`` module to include `tensorflow` and libraries from the ``viz`` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[tensorflow]

For text related tasks, install the ``text`` module. This will include the `wordcloud`, `spacy` libraries.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[text]

Install the ``torch`` module to include `pytorch` and libraries from the ``viz`` module.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[torch]

Install the ``viz`` module to include libraries for visualization tasks. Some of the key packages are `bokeh`, `folium`, `seaborn` and related packages.

.. code-block:: bash

    $ python3 -m pip install oracle-ads[viz]

**Note**

Multiple extra dependencies can be installed together. For example:

.. code-block:: bash

    $ python3 -m pip install  oracle-ads[notebook,viz,text]


