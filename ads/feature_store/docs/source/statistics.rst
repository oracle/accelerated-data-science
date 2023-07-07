Statistics
*************

Feature Store provides functionality to compute statistics for feature groups & datasets and persist them along with the metadata. These statistics can help you
to derive insights about the data quality.

.. note::

  Feature Store utilizes MLM Insights which is a Python API that helps evaluate & monitor data for entirety of ML Observability lifecycle. It performs data summarization which reduces a dataset into a set of descriptive statistics.


Statistics Configuration
========================
Computation of statistical metrics happens by default for all the features but you can configure it using ``StatisticsConfig`` object. This object can be passed at the creation of
feature group or dataset or it can be later updated as well.

.. code-block:: python3

  # Define statistics configuration for selected features
  stats_config = StatisticsConfig().with_is_enabled(True).with_columns(["column1", "column2"])


This can be used with feature group instance.

.. code-block:: python3

  # Fetch stats results for a feature group job
  from ads.feature_store.feature_group import FeatureGroup

  feature_group_resource = (
    FeatureGroup()
    .with_feature_store_id(feature_store.id)
    .with_primary_keys(["<key>"])
    .with_name("<name>")
    .with_entity_id(entity.id)
    .with_compartment_id(<compartment_id>)
    .with_schema_details_from_dataframe(<dataframe>)
    .with_statistics_config(stats_config)

Similarly for dataset instance.

.. code-block:: python3

  from ads.feature_store.dataset import Dataset

  dataset = (
        Dataset
        .with_name("<dataset_name>")
        .with_entity_id(<entity_id>)
        .with_feature_store_id("<feature_store_id>")
        .with_description("<dataset_description>")
        .with_compartment_id("<compartment_id>")
        .with_dataset_ingestion_mode(DatasetIngestionMode.SQL)
        .with_query('SELECT col FROM <entity_id>.<feature_group_name>')
        .with_statistics_config(stats_config)
  )

Statistics Results
==================
You can call the ``get_statistics()`` method of the FeatureGroup or Dataset instance to fetch validation results for a specific ingestion job.

The ``get_statistics()`` method takes the following optional parameter:

- ``job_id: string``. Id of feature group/dataset job

.. code-block:: python3

  # Fetch stats results for a feature group job
  df = feature_group.get_statistics(job_id).to_pandas()

similarly for dataset instance

.. code-block:: python3

  # Fetch stats results for a dataset job
  df = dataset.get_statistics(job_id).to_pandas()

.. image:: figures/stats_1.png