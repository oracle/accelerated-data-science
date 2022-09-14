Oracle Accelerated Data Science SDK (ADS)
=========================================

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: History:

   release_notes

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Configuration:

   user_guide/overview/overview
   user_guide/quick_start/quick_start
   user_guide/configuration/index

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Tasks:

   user_guide/loading_data/index
   user_guide/data_catalog_metastore/index
   user_guide/data_transformation/data_transformation
   user_guide/data_visualization/visualization
   user_guide/model_training/automl

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Components:

   user_guide/big_data_service/index
   user_guide/data_flow/dataflow
   user_guide/data_labeling/index
   user_guide/feature_type/index
   user_guide/jobs/index
   user_guide/logs/logs
   user_guide/model_catalog/model_catalog
   user_guide/model_deployment/index
   user_guide/model_evaluation/index
   user_guide/model_explainability/index
   user_guide/model_serialization/index
   user_guide/secrets/index
   user_guide/ADSString/index
   user_guide/text_extraction/text_dataset

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Classes:

   modules

.. admonition:: Oracle Accelerated Data Science (ADS) SDK

   The Oracle Accelerated Data Science (ADS) SDK is a Python library that is included as part of the Oracle Cloud Infrastructure Data Science service. ADS offers a friendly user interface, with objects and methods that cover all the steps involved in the lifecycle of machine learning models, from data acquisition to model evaluation and interpretation.

.. admonition:: Installation

   python3 -m pip install oracle-ads

.. admonition:: Source Code

   `https://github.com/oracle/accelerated-data-science <https://github.com/oracle/accelerated-data-science>`_

.. code:: ipython3

   >>> import ads
   >>> ads.hello()

     O  o-o   o-o
    / \ |  \ |
   o---o|   O o-o
   |   ||  /     |
   o   oo-o  o--o

   ADS SDK version: X.Y.Z
   Pandas version: x.y.z
   Debug mode: False
