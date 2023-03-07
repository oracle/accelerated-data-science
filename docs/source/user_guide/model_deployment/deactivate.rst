Deactivate
**********

An activate model deployment can be deactivated using a ``ModelDeployment`` object. Deactivating a model deployment shuts down the instances that are associated with your deployment. See `Deactivate Model Deployments <https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_manage.htm#model_dep_deactivate>`_ for more details.

If you have a ``ModelDeployment`` object, you can use the ``.deactivate()`` method to deactivate the model that is associated with that object.

In the following code snippets, the variable ``deployment`` is a ``ModelDeployment`` object.  This object can be obtained from a call to ``.deploy()`` or ``.from_id()``.

.. code-block:: python3

   deployment.deactivate(wait_for_completion=True)

