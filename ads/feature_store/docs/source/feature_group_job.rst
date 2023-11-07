.. _Feature Group Job:

Feature Group Job
*****************

A feature group job is the processing instance of a dataset. Each feature group job includes validation and statistics results.

Define
======

In an ADS feature store module, you can use the Python API or a yaml file to define a dataset job.


The following example defines a feature group job and gives it a name. A ``FeatureGroupJob`` instance is created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.feature_group_job import FeatureGroupJob

    feature_group_job = (
        FeatureGroupJob
        .with_name("<feature_group_job_name>")
        .with_feature_store_id("<feature_store_id>")
        .with_description("<feature_group_job_description>")
        .with_compartment_id("<compartment_id>")
    )

  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.feature_group_job import FeatureGroupJob

    yaml_string = """
    kind: feature_group_job
    spec:
      compartmentId: ocid1.compartment..<unique_id>
      description: <feature_group_job_description>
      name: <feature_group_job_name>
      featureStoreId: <feature_store_id>
    type: feature_group_job
    """

    feature_group_job = FeatureGroupJob.from_yaml(yaml_string)


Create
======

Use the ``create()`` method of the ``FeatureGroupJob`` instance to create a dataset job.

.. code-block:: python3

  # Create an feature_group_job
  feature_group_job.create()


Load
====

Use the ``from_id()`` method from the ``FeatureGroupJob`` class to load an existing dataset job by specifyingh its OCID. It returns a ``FeatureGroupJob`` instance.

.. code-block:: python3

  from ads.feature_store.feature_group_job import FeatureGroupJob

  feature_group_job = FeatureGroupJob.from_id("<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``FeatureGroupJob`` instance to delete a dataset job.

A feature group job can only be deleted when its associated entities are all deleted.

.. code-block:: python3

  feature_group_job.delete()
