Entity
********

An entity is a group of semantically related features. The first step a consumer of features would typically do when accessing the feature store service is to list the entities and the entities associated features. Another way to look at it is that an entity is an object or concept that is described by its features. Examples of entities could be customer, product, transaction, review, image, document, etc.


Define
======

In an ADS feature store module, you can either use the Python API or YAML to define a entity.


With the specified way below, you can define a entity and give it a name.
A ``Entity`` instance will be created.

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

You can call the ``create()`` method of the ``Entity`` instance to create an entity.

.. code-block:: python3

  # Create an entity
  entity.create()


Load
====

Use the ``from_id()`` method from the ``Entity`` class to load an existing entity with its OCID provided. It returns a ``Entity`` instance.

.. code-block:: python3

  from ads.feature_store.entity import Entity

  entity = Entity.from_id("<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``Entity`` instance to delete a entity.

A entity can only be deleted when its associated entities are all deleted,

.. code-block:: python3

  entity.delete()
