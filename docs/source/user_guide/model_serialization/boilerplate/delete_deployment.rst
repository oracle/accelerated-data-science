Use the ``.delete_deployment()`` method on the serialization model object to delete a model deployment. You must delete a model deployment before deleting its associated model from the model catalog.

Each time you call the ``.deploy()`` method, it creates a new deployment. Only the most recent deployment is attached to the object.

The ``.delete_deployment()`` method deletes the most recent deployment and takes the following optional parameter:

- ``wait_for_completion: (bool, optional)``. Defaults to ``False`` and the process runs in the background. If set to ``True``, the method returns when the model deployment is deleted.


