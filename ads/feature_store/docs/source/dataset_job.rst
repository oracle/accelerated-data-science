.. _Dataset Job:

Dataset Job
***********

Dataset job is the execution instance of a dataset. Each dataset job will include validation results and statistics results.

Define
======

In an ADS feature store module, you can either use the Python API or YAML to define a dataset job.


With the specified way below, you can define a dataset_job and give it a name.
A ``DatasetJob`` instance will be created.

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

You can call the ``create()`` method of the ``DatasetJob`` instance to create an dataset job.

.. code-block:: python3

  # Create an dataset_job
  dataset_job.create()


Load
====

Use the ``from_id()`` method from the ``DatasetJob`` class to load an existing dataset job with its OCID provided. It returns a ``DatasetJob`` instance.

.. code-block:: python3

  from ads.feature_store.dataset_job import DatasetJob

  dataset_job = DatasetJob.from_id("ocid1.dataset_job..<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``DatasetJob`` instance to delete a dataset job.

A dataset_job can only be deleted when its associated entities are all deleted,

.. code-block:: python3

  dataset_job.delete()
