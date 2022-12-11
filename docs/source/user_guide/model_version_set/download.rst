Download
________

Create a ``ModelVersionSet`` object by downloading the metadata from the model catalog. The ``ModelVersionSet`` class has a ``.from_id()`` method that accepts the model version set OCID. The ``.from_name()`` method takes the name of the model version set.

``.from_id()``
^^^^^^^^^^^^^^

.. code-block:: python3

   mvs = ModelVersionSet.from_id(id="<model_version_set_id>")


``.from_name()``
^^^^^^^^^^^^^^^^

.. code-block:: python3

    mvs = ModelVersionSet.from_name(name="<model_version_set_name>")


