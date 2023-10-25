Transformation
**************

Transformations in a feature store us the operations and processes applied to raw data to create, modify, or derive new features that can be used as inputs for machine learning models. These transformations are important for improving the quality, relevance, and usefulness of features and can enhance the performance of models. A transformation is an object that represents a transformation applied on the feature group and can be a Pandas transformation or Spark SQL transformation.

Define
======

In an ADS feature store module, you can use the Python API or a yaml file to define a transformation.


The following example defines a transformation and gives it a name. A ``Transformation`` instance is created.

.. tabs::

  .. code-tab:: Python3
    :caption: Python

    from ads.feature_store.transformation import Transformation

    transformation = (
        Transformation
        .with_name("<transformation_name>")
        .with_feature_store_id("<feature_store_id>")
        .with_source_code("<source_code>")
        .with_transformation_mode("<transformation_mode>")
        .with_description("<transformation_description>")
        .with_compartment_id("<compartment_id>")
    )

  .. code-tab:: Python3
    :caption: YAML

    from ads.feature_store.transformation import Transformation

    yaml_string = """
    kind: transformation
    spec:
      compartmentId: ocid1.compartment..<unique_id>
      description: <transformation_description>
      name: <transformation_name>
      featureStoreId: <feature_store_id>
      sourceCode: <source_code>
      transformationMode: <transformation_mode>
    type: transformation
    """

    transformation = Transformation.from_yaml(yaml_string)


Create
======

Use the ``create()`` method of the ``Transformation`` instance to create an transformation.

.. code-block:: python3

  # Create an transformation
  transformation.create()


Load
====

Use the ``from_id()`` method from the ``Transformation`` class to load an existing transformation by specifiying its OCID. A ``Transformation`` instance is returned.

.. code-block:: python3

  from ads.feature_store.transformation import Transformation

  transformation = Transformation.from_id("ocid1.transformation..<unique_id>")

Delete
======

Use the ``.delete()`` method on the ``Transformation`` instance to delete a transformation. A transformation can only be deleted when its associated entities are all deleted.

.. code-block:: python3

  transformation.delete()
