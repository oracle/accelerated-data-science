Feature Store
*************

Feature store is the top level entity for feature store service.

Define
======

In an ADS feature store module, you can either use the Python API or YAML to define a feature_store.


With the specified way below, you can define a feature store and give it a name.
A ``FeatureStore`` instance will be created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.feature_store import FeatureStore

    feature_store = (
        feature_store_resource = FeatureStore().
            with_description(<feature_store_description>)
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

You can call the ``create()`` method of the ``FeatureStore`` instance to create an feature store.

.. code-block:: python3

  # Create an feature store
  feature_store.create()


Load
====

Use the ``from_id()`` method from the ``FeatureStore`` class to load an existing feature store with its OCID provided. It returns a ``FeatureStore`` instance.

.. code-block:: python3

  from ads.feature_store.feature_store import FeatureStore

  feature_store = FeatureStore.from_id("ocid1.feature_store..<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``FeatureStore`` instance to delete a feature store.

A feature store can only be deleted when its associated entities are all deleted,

.. code-block:: python3

  feature_store.delete()

SQL
===
You can call the ``sql()`` method of the FeatureStore instance to query a feature store.

Query a feature store using sql
###############################

.. code-block:: python3

  # Fetch the entity id. Entity id is used as database name in feature store
  entity_id = entity.id

  # Form a query with entity id and fetch the results
  sql = (f"SELECT feature_group_a.* "
       f"FROM {entity_id}.feature_group_a "
       f"JOIN {entity_id}.feature_group_b "
       f"ON {entity_id}.feature_group_a.col_1={entity_id}.feature_group_b.col_2 "
       f"JOIN {entity_id}.feature_group_a.col_1={entity_id}.feature_group_b.col_3 ")

  # Run the sql query and fetch the results as data-frame
  df = feature_store.sql(sql)

Create Entity
=============
You can call the ``create_entity()`` method of the FeatureStore instance to create a ``Entity``.

.. code-block:: python3

  # Create a feature store entity
  feature_store.create_entity(name="<ENTITY_NAME>")

Create Transformation
=====================
Transformations in a feature store refers to the operations and processes applied to raw data to create, modify or derive new features that can be used as inputs for ML Models. These transformations are crucial for improving the quality, relevance and usefulness of features which in turn can enhance the performance of ml models.
You can call the ``create_transformation()`` method of the FeatureStore instance to create a ``Transformation``.

.. code-block:: python3

  # Create a feature store entity
  feature_store.create_transformation(
      source_code_func="<FUNCTION>",
      transformation_mode="SQL|PANDAS"
      display_name="<TRANSFORMATION NAME>"
  )
