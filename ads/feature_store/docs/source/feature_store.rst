Feature Store
*************

Feature store is the top-level entity for a Feature Store.

Define
======

In an ADS Feature Store module, you can use the Python API or a .yaml file to define a Feature Store.


The following example defines a Feature Store and gives it a name. A ``FeatureStore`` instance is created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.feature_store import FeatureStore

    feature_store = (
            FeatureStore().
            with_description(<feature_store_description>).
            with_compartment_id("ocid1.compartment..<unique_id>").
            with_name(<feature_store_name>).
            with_offline_config(
                metastore_id=metastoreId
            )
    )

  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.feature_store import FeatureStore

    yaml_string = """
    kind: feature_store
    spec:
      compartmentId: ocid1.compartment..<unique_id>
      description: <feature_store_description>
      name: <feature_store_name>
      featureStoreId: <feature_store_id>
    type: feature_store
    """

    feature_store = FeatureStore.from_yaml(yaml_string)


Create
======

Use the ``create()`` method of the ``FeatureStore`` instance to create a Feature Store.

.. code-block:: python3

  # Create an Feature Store
  feature_store.create()


Load
====

Use the ``from_id()`` method from the ``FeatureStore`` class to load an existing Feature Store by specifying its OCID. A ``FeatureStore`` instance is returned.

.. code-block:: python3

  from ads.feature_store.feature_store import FeatureStore

  feature_store = FeatureStore.from_id("<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``FeatureStore`` instance to delete a Feature Store. A Feature Store can only be deleted when its associated entities are all deleted.

.. code-block:: python3

  feature_store.delete()

SQL
===
Use the ``sql()`` method of the ``FeatureStore`` instance to query a Feature Store.

The following example queries a Feature Store using SQL:

.. code-block:: python3

  # Fetch the entity id. Entity id is used as database name in Feature Store
  entity_id = entity.id

  # Form a query with entity id and fetch the results
  sql = (f"SELECT feature_group_a.* "
       f"FROM {entity_id}.feature_group_a "
       f"JOIN {entity_id}.feature_group_b "
       f"ON {entity_id}.feature_group_a.col_1={entity_id}.feature_group_b.col_2 "
       f"JOIN {entity_id}.feature_group_a.col_1={entity_id}.feature_group_b.col_3 ")

  # Run the sql query and fetch the results as data-frame
  df = feature_store.sql(sql)

Create an Entity
================
Use the ``create_entity()`` method of the ``FeatureStore`` instance to create an ``Entity``.

.. code-block:: python3

  # Create a Feature Store entity
  feature_store.create_entity(name="<ENTITY_NAME>")

Create a Transformation
=======================
Transformations in a Feature Store are the operations and processes applied to raw data to create, modify, or derive new features for use as inputs for machine learning models. These transformations are necessary for improving the quality, relevance, and usefulness of features that then enhance the performance of models.
You can call the ``create_transformation()`` method of the FeatureStore instance to create a ``Transformation``.

.. code-block:: python3

  # Create a Feature Store entity
  feature_store.create_transformation(
      source_code_func="<FUNCTION>",
      transformation_mode="SQL|PANDAS"
      name="<TRANSFORMATION NAME>"
  )
