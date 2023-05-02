########
Overview
########


The Oracle Accelerated Data Science (ADS) SDK is a Python library that is included as part of the Oracle Cloud Infrastructure Data Science service. ADS offers a friendly user interface with objects and methods that describe the steps involved in the lifecycle of machine learning models, from data acquisition to model evaluation and interpretation.

You access ADS when you launch a ``JupyterLab`` session from the Data Science service. ADS is pre-configured to access Data Science and other Oracle Cloud Infrastructure resources, such as the models in the Data Science model catalog or files in Oracle Cloud  Infrastructure Object Storage.

The ADS SDK is also publicly available on PyPi, and can be installed with ``python3 -m pip install oracle-ads``.

Main Features
=============

* **Connect to Data Sources**

  The Oracle JupyterLab environment is pre-installed with default storage options for reading from and writing to Oracle Cloud Infrastructure Object Storage. However, you can load your datasets into ADS from almost anywhere including:

  - Amazon S3
  - Blob
  - Elastic Search instances
  - Google Cloud Service
  - Hadoop Distributed File System
  - Local files
  - Microsoft Azure
  - MongoDB
  - NoSQL DB instances
  - Oracle Autonomous Data Warehouse
  - Oracle Cloud Infrastructure Object Storage
  - Oracle Database


  These datasets can be numerous formats including:

  - Apache server log files
  - Excel
  - HDF5
  - JSON
  - Parquet
  - SQL
  - XML
  - arff
  - csv
  - libsvm
  - tsv

  .. figure:: figures/open-dataset.png
     :align: center

     **Example of Opening a Dataset**

* **Perform Exploratory Data Analysis**

  The ADS data type discovery supports simple data types like categorical, continuous, ordinal to sophisticated data types. For example, geo data, date time, zip codes, and credit card numbers.

  .. figure:: figures/target-visualization.png
     :align: center

     **Example showing exploring the class imbalance of a target variable**

* **Automatic Data Visualization**

  The ``ADSDataset`` object comes with a comprehensive plotting API. It allows you to explore data visually using automatic plotting or create your own custom plots.

  .. figure:: figures/feature-visualization-1.png
     :align: center

     **Example showing Gaussian Heatmap Visualization**
  .. figure:: figures/feature-visualization-2.png
     :align: center

     **Example showing plotting lat/lon points on a map**

* **Feature Engineering**

  Leverage ADS and the `Pandas API <https://pandas.pydata.org/docs/index.html>`_ to transform the content of a `ADSDataset` object with custom data transformations.

  .. figure:: figures/balance-dataset.png
     :align: center

     **Example showing using ADS to drop columns and apply auto transforms**

* **Data Snapshotting for Training Reproducibility**

  Save and load a copy of any dataset in binary optimized Parquet format. By snapshotting a dataset, a URL is returned that can be used by anyone with access to the resource to load the data exactly how it was at that point with all transforms materialized.


* **Model Evaluations**

  Model evaluation generates a comprehensive suite of evaluation metrics and suitable visualizations to measure model performance against new data, and can rank models over time to ensure optimal behavior in production. Model evaluation goes beyond raw performance to take into account expected baseline behavior. It uses a cost API so that the different impacts of false positives and false negatives can be fully incorporated.

  ADS helps data scientists evaluate ``ADSModel`` instances through the `ADSEvaluator` object. This object provides a comprehensive API that covers regression, binary, and multinomial classification use cases.

  .. figure:: figures/model-evaluation.png
     :align: center

     **Example showing how to evaluate a list of models**

  .. figure:: figures/model-evaluation-performance.png
     :align: center

     **Example showing some model evaluation plots**


* **Interact with the Model Catalog**

  You can upload the models that you create with ADS into the Data Science model catalog directly from ADS. You can save all your models, with their provenance information, in the catalog and make them accessible to anybody who needs to use them. Other users can then load the models and use them as an ``ADSModel`` object. You can also use this feature to help put the models into production with `Oracle Functions <https://docs.cloud.oracle.com/iaas/Content/Functions/Concepts/functionsoverview.htm>`_.

