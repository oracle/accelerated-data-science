
.. meta::
    :description lang=en:
        Oracle Accelerated Data Science SDK (ORACLE-ADS)
        is a Python library that is part of the Oracle Cloud Infrastructure Data Science service. ORACLE-ADS is the client
        library and CLI for Machine learning engineers to work with Cloud Infrastructure (CPU and GPU VMs, Storage etc, Spark) for Data, Models,
        Notebooks, Pipelines and Jobs.

Oracle Accelerated Data Science (ADS)
=====================================
|PyPI|_ |Python|_ |Notebook Examples|_

.. |PyPI| image:: https://img.shields.io/pypi/v/oracle-ads.svg?style=for-the-badge&logo=pypi&logoColor=white
..  _PyPI: https://pypi.org/project/oracle-ads/
.. |Python| image:: https://img.shields.io/pypi/pyversions/oracle-ads.svg?style=for-the-badge&logo=pypi&logoColor=white
..  _Python: https://pypi.org/project/oracle-ads/
.. |Notebook Examples| image:: https://img.shields.io/badge/docs-notebook--examples-blue?style=for-the-badge&logo=pypi&logoColor=white
..  _Notebook Examples: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Getting Started

   release_notes
   user_guide/quick_start/quick_start

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Installation and Configuration

   user_guide/cli/quickstart
   user_guide/cli/authentication
   user_guide/cli/opctl/configure
   user_guide/cli/opctl/local-development-setup

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: AI Quick Actions API Server

   user_guide/aqua/apiserver

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Low-Code AI Operators

   user_guide/operators/index
   user_guide/operators/forecast_operator/index
   user_guide/operators/anomaly_detection_operator/index
   user_guide/operators/pii_operator/index
   user_guide/operators/recommender_operator/index
   user_guide/operators/common/index

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Tasks

   user_guide/loading_data/connect
   user_guide/data_labeling/index
   user_guide/data_transformation/data_transformation
   user_guide/data_visualization/visualization
   user_guide/model_training/index
   user_guide/model_registration/introduction

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Integrations

   user_guide/apachespark/spark
   user_guide/big_data_service/index
   user_guide/jobs/index
   user_guide/logs/logs
   user_guide/pipeline/index
   user_guide/large_language_model/index
   user_guide/secrets/index


.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Classes

   modules

.. admonition:: Introducing AI Quick Actions
   :class: note

   Deploy, Fine Tune and Evaluate Large language models such as `Mistral-7B-Instruct-v0.2 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__, `CodeLlama-13b-Instruct-hf <https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf>`__, etc with just a couple of clicks.

   Visit `AI Quick Actions documentation <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/ai-quick-actions>`__ for the latest information
  
.. admonition:: Oracle Accelerated Data Science (ADS)
   :class: note

   Oracle Accelerated Data Science (ADS) is maintained by the Oracle Cloud Infrastructure Data Science service team. It speeds up common data science activities by providing tools that automate and/or simplify common data science tasks, along with providing a data scientist friendly pythonic interface to Oracle Cloud Infrastructure (OCI) services, most notably OCI Data Science, Data Flow, Object Storage, and the Autonomous Database. ADS gives you an interface to manage the lifecycle of machine learning models, from data acquisition to model evaluation, interpretation, and model deployment.

   With ADS you can:

   - Read datasets from Oracle Object Storage, Oracle RDBMS (ATP/ADW/On-prem), AWS S3, and other sources into Pandas dataframes.
   - Easily compute summary statistics on your dataframes and perform data profiling.
   - Tune models using hyperparameter optimization with the ADSTuner tool.
   - Generate detailed evaluation reports of your model candidates with the ADSEvaluator module.
   - Save machine learning models to the OCI Data Science Models.
   - Deploy those models as HTTPS endpoints with Model Deployment.
   - Launch distributed ETL, data processing, and model training jobs in Spark with OCI Data Flow.
   - Train machine learning models in OCI Data Science Jobs.
   - Manage the lifecycle of conda environments through the ads conda command line interface (CLI).
   - Distributed Training with PyTorch, Horovod and Dask


.. admonition:: Installation
   :class: note

   python3 -m pip install oracle-ads


.. admonition:: Source Code
   :class: note

   `https://github.com/oracle/accelerated-data-science <https://github.com/oracle/accelerated-data-science>`_

.. code-block:: python3

   >>> import ads
   >>> ads.hello()

.. runblock:: pycon

   >>> import ads
   >>> ads.hello()


Additional Documentation
++++++++++++++++++++++++

  - `OCI Data Science and AI services Examples <https://github.com/oracle/oci-data-science-ai-samples>`_
  - `Oracle AI & Data Science Blog <https://blogs.oracle.com/ai-and-datascience/>`_
  - `OCI Documentation <https://docs.oracle.com/en-us/iaas/data-science/using/data-science.htm>`_
  - `OCIFS Documentation <https://ocifs.readthedocs.io/en/latest/>`_
  - `Example Notebooks <https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples>`_

Examples
++++++++

Load data from Object Storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python3

   import ads
   import oci
   import pandas as pd

   ads.set_auth(
      auth="api_key", oci_config_location=oci.config.DEFAULT_LOCATION, profile="DEFAULT"
   )
   bucket_name = "<bucket_name>"
   path = "<path>"
   namespace = "<namespace>"
   df = pd.read_csv(
      f"oci://{bucket_name}@{namespace}/{path}", storage_options=ads.auth.default_signer()
   )



Load data from Autonomous DB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses SQL injection safe binding variables.

.. code-block:: python3

   import ads
   import pandas as pd

   connection_parameters = {
      "user_name": "<user_name>",
      "password": "<password>",
      "service_name": "<tns_name>",
      "wallet_location": "<file_path>",
   }

   df = pd.DataFrame.ads.read_sql(
      """
      SELECT *
      FROM SH.SALES
      WHERE ROWNUM <= :max_rows
      """,
      bind_variables={ max_rows : 100 },
      connection_parameters=connection_parameters,
   )

More Examples
~~~~~~~~~~~~~

See :doc:`quick start<user_guide/quick_start/quick_start>` guide for more examples

Contributing
++++++++++++

This project welcomes contributions from the community. Before submitting a pull request, please review our contribution guide `CONTRIBUTING.md <https://github.com/oracle/accelerated-data-science/blob/main/CONTRIBUTING.md>`_.

Find Getting Started instructions for developers in `README-development.md <https://github.com/oracle/accelerated-data-science/blob/main/README-development.md>`_

Security
++++++++

Consult the security guide `SECURITY.md <https://github.com/oracle/accelerated-data-science/blob/main/SECURITY.md>`_ for our responsible security vulnerability disclosure process.

License
+++++++

Copyright (c) 2020, 2022 Oracle and/or its affiliates. Licensed under the `Universal Permissive License v1.0 <https://oss.oracle.com/licenses/upl/>`_
