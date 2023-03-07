Delete
******

A model deployment can be deleted using a ``ModelDeployment`` object.

When a model deployment is deleted, it deletes the load balancer instances associated with it. However, it doesn't delete other resources like log group, log, or model.

If you have a ``ModelDeployment`` object, you can use the ``.delete()`` method to delete the model that is associated with that object. The optional ``wait_for_completion`` parameter accepts a boolean value and determines if the process is blocking or not. 

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.from_id()``.

.. code-block:: python3

    deployment = deployment.delete(wait_for_completion=True)

