Associate a Model
_________________

Model version sets are a collection of models. After a model is associated with a model version set, the model can't be associated with a different model version set. Further, the model can't be disassociated with the model version set.

When a model is associated with a model version set, a version label can be assigned to the set. This version is different than the model version that is maintained by the model version set.

There are a number of ways to associate a model with a model version set. Which approach you use depends on the workflow.

ModelVersionSet Object
----------------------

For a model not associated with a model version set,  use the ``.model_add()`` method on a ``ModelVersionSet`` object to associate the model with the model version set. The ``.model_add()`` requires that you provide the model OCID and optionally a version label.

.. code-block:: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")
   mvs.model_add(<your_model_id>, version_label="Version 1")

Model Serialization
-------------------

The :ref:`Model Serialization` classes allow a model to be associated with a model version set at the time that it is saved to the model catalog. You do this with the ``model_version_set`` parameter in the ``.save()`` method. In addition, you can add the model's version label with the ``version_label`` parameter.

The ``model_version_set`` parameter accepts a model version set's OCID or name. The parameter also accepts a ``ModelVersionSet`` object.

In the following, the ``model`` variable is a :ref:`Model Serialization` object that is to be saved to the model catalog, and at the same time associated with a model version set.

.. code-block:: python3

   model.save(
       display_name='Model attached to a model version set',
       version_label = "Version 1",
       model_version_set="<model_version_set_id>"
   )


Context Manager
---------------

To associate several models with a model version set, use a context manager. The ``ads.model.experiment()`` method requires a ``name`` parameter. If the model catalog has a matching model version set name, the model catalog uses that model version set. If the parameter ``create_if_not_exists`` is ``True``, the ``experiment()`` method attempts to match the model version set name with name in the model catalog. If the name does not exist, the method creates a new model version set.

Within the context manager, you can save multiple :ref:`Model Serialization` models without specifying the ``model_version_set`` parameter because it's taken from the model context manager. The following example assumes that ``model_1``, ``model_2``, and ``model_3`` are :ref:`Model Serialization` objects. If the model version set doesn't exist in the model catalog, the example creates a model version set named ``my_model_version_set``.  If the model version set exists in the model catalog, the models are saved to that model version set.

.. code-block:: python3

   with ads.model.experiment(name="my_model_version_set", create_if_not_exists=True):
        # experiment 1
        model_1.save(
            display_name='Generic Model Experiment 1',
            version_label = "Experiment 1"
        )

        # experiment 2
        model_2.save(
            display_name='Generic Model Experiment 2',
            version_label = "Experiment 2"
        )

        # experiment 3
        model_3.save(
            display_name='Generic Model Experiment 3',
            version_label = "Experiment 3"
        )


