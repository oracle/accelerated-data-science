.. _Dataset Job:

Dataset Job
***********

A dataset job is the processing instance of a dataset. Each dataset job includes validation and statistics results.


Load
====

Use the ``from_id()`` method from the ``DatasetJob`` class to load an existing dataset job by specifying its OCID. A``DatasetJob`` instance is returned.

.. code-block:: python3

  from ads.feature_store.dataset_job import DatasetJob

  dataset_job = DatasetJob.from_id("<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``DatasetJob`` instance to delete a dataset job. A dataset job can only be deleted when its associated entities are all deleted.

.. code-block:: python3

  dataset_job.delete()
