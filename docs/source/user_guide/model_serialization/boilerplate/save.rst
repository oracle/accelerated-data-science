After you are satisfied with the performance of your model and have verified that the ``score.py`` file is working, use the ``.save()`` method to save the model to the model catalog. The ``.save()`` method bundles up the model artifacts, stores them in the model catalog, and returns the model OCID.

The ``.save()`` method stores the model artifacts in the model catalog. It takes the following parameters:

- ``defined_tags : (Dict(str, dict(str, object)), optional)``: Defaults to ``None``. Defined tags for the model.
- ``description: (str, optional)``: Defaults to ``None``. The description of the model.
- ``display_name: (str, optional)``: Defaults to ``None``. The name of the model.
- ``freeform_tags : Dict(str, str)``: Defaults to ``None``. Free form tags for the model.
- ``ignore_introspection: (bool, optional)``: Defaults to ``None``. Determines whether to ignore the result of model introspection or not. If set to ``True``, then ``.save()`` ignores all model introspection errors.
-  ``**kwargs``:
    - ``compartment_id : (str, optional)``: Compartment OCID. If not specified, the value is taken either from the environment variables or model properties.
    - ``project_id: (str, optional)``: Project OCID. If not specified, the value is taken either from the environment variables or model properties.
    - ``timeout: (int, optional)``: Defaults to 10 seconds. The connection timeout in seconds for the client.

The ``.save()`` method reloads ``score.py`` and ``runtime.yaml`` files from disk. This will pick up any changes that have been made to those files. If ``ignore_introspection=False`` then it conducts an introspection test to determine if the model deployment might have issues. If potential problems are detected, it will suggest possible remedies. Lastly, it uploads the artifacts to the model catalog, and returns the model OCID. You can also call ``.instrospect()`` to conduct the test any time after you call ``.prepare()``.

