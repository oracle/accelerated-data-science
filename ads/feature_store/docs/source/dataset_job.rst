.. _Dataset Job:

Dataset Job
***********

A dDataset job is the processing instance of a dataset. Each dataset job includes validation and statistics results.

Define
======

In an ADS feature store module, you can use the Python API or a yaml file to define a dataset job.


The following example defines a dataset job and gives it a name. A ``DatasetJob`` instance is created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.dataset_job import DatasetJob

    dataset_job = (
        DatasetJob
        .with_name("<dataset_job_name>")
        .with_feature_store_id("<feature_store_id>")
        .with_description("<dataset_job_description>")
        .with_compartment_id("<compartment_id>")
    )

  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.dataset_job import DatasetJob

    yaml_string = """
    kind: dataset_job
    spec:
      compartmentId: ocid1.compartment..<unique_id>
      description: <dataset_job_description>
      name: <dataset_job_name>
      featureStoreId: <feature_store_id>
    type: dataset_job
    """

    dataset_job = DatasetJob.from_yaml(yaml_string)


Create
======

Use the ``create()`` method of the ``DatasetJob`` instance to create a dataset job.

.. code-block:: python3

  # Create an dataset_job
  dataset_job.create()


Load
====

Use the ``from_id()`` method from the ``DatasetJob`` class to load an existing dataset job by specifying its OCID. A``DatasetJob`` instance is returned.

.. code-block:: python3

  from ads.feature_store.dataset_job import DatasetJob

  dataset_job = DatasetJob.from_id("ocid1.dataset_job..<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``DatasetJob`` instance to delete a dataset job. A dataset_job can only be deleted when its associated entities are all deleted,

.. code-block:: python3

  dataset_job.delete()
