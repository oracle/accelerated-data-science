List
____

ModelVersionSet
---------------

The ``.list()`` method on the ``ModelVersionSet`` class takes a compartment ID and lists the model version sets in that compartment. If the compartment isn't given, then the compartment of the notebook session is used.

The following example uses context manager to iterate over the collection of model version sets:

.. code-block:: python3

    for model_version_set in ModelVersionSet.list():
        print(model_version_set)
        print("---------")


Model
-----

You can get the list of models associated with a model version set by calling the ``.models()`` method on a ``ModelVersionSet`` object. A list of models that are associated with that model version set is returned. First, you must obtain a ``ModelVersionSet`` object. Use the ``.from_id()`` method if you know the model version set OCID. Alternatively, use the ``.from_name()`` method if you know the name of the model version set.

.. code-block:: python3

    mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
    models = mvs.models()

    for dsc_model in models:
        print(dsc_model.display_name, dsc_model.id, dsc_model.status)
