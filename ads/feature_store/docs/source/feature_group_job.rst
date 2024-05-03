.. _Feature Group Job:

Feature Group Job
*****************

A feature group job is the processing instance of a dataset. Each feature group job includes validation and statistics results.


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
