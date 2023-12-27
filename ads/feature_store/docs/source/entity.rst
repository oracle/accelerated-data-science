Entity
********

An entity is a group of semantically related features. An entity is an object or concept that is described by its features. The first step when accessing a feature store is typically to list the entities and the entities' associated features. Examples of entities are customer, product, transaction, review, image, aand document.

.. image:: figures/entity.png

Define
======

In an ADS feature store module, you can use the Python API or a .yaml file to define an entity.


The following example defines an entity and gives it a name. An ``Entity`` instance is created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.entity import Entity

    entity = (
        Entity
        .with_name("<entity_name>")
        .with_feature_store_id("<feature_store_id>")
        .with_description("<entity_description>")
        .with_compartment_id("<compartment_id>")
    )

  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.entity import Entity

    yaml_string = """
    kind: entity
    spec:
      compartmentId: ocid1.compartment..<unique_id>
      description: <entity_description>
      name: <entity_name>
      featureStoreId: <feature_store_id>
    type: entity
    """

    entity = Entity.from_yaml(yaml_string)


Create
======

Use the ``create()`` method of the ``Entity`` instance to create an entity.

.. code-block:: python3

  # Create an entity
  entity.create()


Load
====

Use the ``from_id()`` method from the ``Entity`` class to load an existing entity by specifying its OCID. An ``Entity`` instance is returned.

.. code-block:: python3

  from ads.feature_store.entity import Entity

  entity = Entity.from_id("<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``Entity`` instance to delete a entity. A entity can only be deleted when its associated entities are all deleted.

.. code-block:: python3

  entity.delete()
