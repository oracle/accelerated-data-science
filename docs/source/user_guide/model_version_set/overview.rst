Overview
________

The normal workflow of a data scientist is to create a model and push it into production. While in production the data scientist learns what the model is doing well and what it isn't. Using this information the data scientist creates an improved model. These models are linked using model version sets. A model version set is a collection of models that are related to each other. A model version set is a way to track the relationships between models. As a container, the model version set takes a collection of models. Those models are assigned a sequential version number based on the order they are entered into the model version set. 

In ADS the class ``ModelVersionSet`` is used to represent the model version set. An object of ``ModelVersionSet`` references a model version set in the Data Science service. The ``ModelVersionSet`` class supports two APIs: the builder pattern and the traditional parameter-based pattern. You can use either of these API frameworks interchangeably and examples for both patterns are included.

Use the ``.create()`` method to create a model version set in your tenancy. If the model version set already exists in the model catalog, use the ``.from_id()`` or ``.from_name()`` method to get a ``ModelVersionSet`` object based on the specified model version set. If you make changes to the metadata associated with the model version set, use the ``.update()`` method to push those changes to the model catalog. The ``.list()`` method lists all model version sets. To add an existing model to a model version set, use the ``.add_model()`` method. The ``.models()`` method lists the models in the model version set. Use the ``.delete()`` method to delete a model version set from the model catalog.


