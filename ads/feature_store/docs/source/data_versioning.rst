Data Versioning
****************

Data versioning is a practice aimed at recording the various data commits integrated into a particular feature group and dataset. This involves tracking changes in data over time while maintaining consistent schema structures and feature definitions within a shared schema version. In the context of feature store, it's important to note that data versioning features are exclusively available for offline feature groups.

.. image:: figures/data_versioning.png


As Of
======

You can call the ``as_of()`` method of the ``FeatureGroup`` or ``Dataset`` instance to get specified point in time and time traveled data.

The ``.as_of()`` method takes the following optional parameter:

- ``commit_timestamp: date-time``. Commit timestamp for feature group
- ``version_number: int``. Version number for feature group

.. code-block:: python3

  # as_of feature group
  df = feature_group.as_of(version_number=1)


History
=======

You can call the ``history()`` method of the ``FeatureGroup`` or ``Dataset`` instance to show history of the feature group.

.. code-block:: python3

  # Show history of feature group
  df = feature_group.history()
